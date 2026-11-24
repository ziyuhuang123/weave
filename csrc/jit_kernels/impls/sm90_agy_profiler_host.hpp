#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <tuple>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>

#include <deep_gemm/impls/sm90_agy_profiler.cuh>

namespace deep_gemm::agy_prof {

struct HostLaunchMeta {
    uint64_t host_prep_ns = 0;
    uint64_t host_launch_api_ns = 0;
    uint64_t seq_id = 0;
    int device_idx = 0;
    int rank_idx = 0;
    int num_ranks = 0;
    int num_experts = 0;
    int num_tokens = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    int is_fused = 0;
    int fast_host_hit = 0;
    int prof_mode = 0;
    int l1_bm = 0, l1_bn = 0, l1_bk = 0, l1_stages = 0;
    int l2_bm = 0, l2_bn = 0, l2_bk = 0, l2_stages = 0;
    std::string cfg_str;
};

inline std::atomic<int>& mode_override_ref() {
    // -1 = read the recorder level from the environment; 0/1/2 = explicit override via PyBind
    static std::atomic<int> v{-1};
    return v;
}

// Recorder level from the environment.  WEAVE_PROFILER_LEVEL (the switch name the
// SM103 / MI308X recorders already use) wins; AGY_PROF is the legacy SM90 spelling.
inline int get_env_mode() {
    const char* s = std::getenv("WEAVE_PROFILER_LEVEL");
    if (s == nullptr || *s == '\0') s = std::getenv("AGY_PROF");
    if (s == nullptr || *s == '\0' || *s == '0') return 0;
    const int parsed = std::atoi(s);
    return parsed > 0 ? parsed : 0;
}

inline int get_mode() {
    const int ov = mode_override_ref().load(std::memory_order_relaxed);
    if (ov >= 0) return ov;
    return get_env_mode();
}

inline void reset_current_device();
inline void set_mode_override(int mode) {
    if (mode == -99) {
        reset_current_device();
        return;
    }
    mode_override_ref().store(mode, std::memory_order_relaxed);
}

inline uint64_t steady_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct DeviceProfSlot {
    std::mutex mtx;
    uint64_t* dev_ptr = nullptr;
    size_t nbytes = sizeof(AgyProfDeviceBuffer);
    HostLaunchMeta last_meta{};
    uint64_t t_host_start_ns = 0;
    uint64_t call_counter = 0;
};

inline std::array<DeviceProfSlot, 16>& device_slots() {
    static std::array<DeviceProfSlot, 16> slots{};
    return slots;
}

inline uint64_t* ensure_and_clear_buffer(int device_idx, cudaStream_t stream) {
    if (device_idx < 0 || device_idx >= 16) return nullptr;
    auto& slot = device_slots()[device_idx];
    std::lock_guard<std::mutex> lk(slot.mtx);
    int prev_dev = -1;
    cudaGetDevice(&prev_dev);
    if (prev_dev != device_idx) cudaSetDevice(device_idx);
    if (slot.dev_ptr == nullptr) {
        cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&slot.dev_ptr), slot.nbytes);
        if (e != cudaSuccess) {
            slot.dev_ptr = nullptr;
            if (prev_dev >= 0 && prev_dev != device_idx) cudaSetDevice(prev_dev);
            return nullptr;
        }
    }
    const char* skip_clr = std::getenv("WEAVE_PROFILER_SKIP_AUTO_CLEAR");
    if (skip_clr == nullptr || *skip_clr == '\0') skip_clr = std::getenv("AGY_PROF_SKIP_AUTO_CLEAR");
    if (skip_clr == nullptr || *skip_clr == '\0' || *skip_clr == '0') {
        cudaMemsetAsync(slot.dev_ptr, 0, slot.nbytes, stream);
    }
    if (prev_dev >= 0 && prev_dev != device_idx) cudaSetDevice(prev_dev);
    return slot.dev_ptr;
}

inline uint64_t* get_existing_buffer(int device_idx) {
    if (device_idx < 0 || device_idx >= 16) return nullptr;
    return device_slots()[device_idx].dev_ptr;
}

inline uint64_t* begin_invocation(int device_idx, int prof_mode) {
    if (prof_mode <= 0 || device_idx < 0 || device_idx >= 16) return nullptr;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_idx).stream();
    uint64_t* ptr = ensure_and_clear_buffer(device_idx, stream);
    auto& slot = device_slots()[device_idx];
    slot.t_host_start_ns = steady_now_ns();
    return ptr;
}

inline void record_launch_timing(
    int device_idx,
    uint64_t t_before_launch_ns,
    uint64_t t_after_launch_ns,
    int rank_idx,
    int num_ranks,
    int num_experts,
    int num_tokens,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    bool is_fused,
    bool fast_host_hit,
    int prof_mode,
    int l1_bm, int l1_bn, int l1_bk, int l1_stages,
    int l2_bm, int l2_bn, int l2_bk, int l2_stages,
    const std::string& cfg_str) {
    if (prof_mode <= 0 || device_idx < 0 || device_idx >= 16) return;
    auto& slot = device_slots()[device_idx];
    std::lock_guard<std::mutex> lk(slot.mtx);
    HostLaunchMeta& m = slot.last_meta;
    m.host_prep_ns = (t_before_launch_ns >= slot.t_host_start_ns && slot.t_host_start_ns > 0)
        ? (t_before_launch_ns - slot.t_host_start_ns) : 0ull;
    m.host_launch_api_ns = (t_after_launch_ns >= t_before_launch_ns)
        ? (t_after_launch_ns - t_before_launch_ns) : 0ull;
    m.seq_id = ++slot.call_counter;
    m.device_idx = device_idx;
    m.rank_idx = rank_idx;
    m.num_ranks = num_ranks;
    m.num_experts = num_experts;
    m.num_tokens = num_tokens;
    m.num_topk = num_topk;
    m.hidden = hidden;
    m.intermediate_hidden = intermediate_hidden;
    m.is_fused = is_fused ? 1 : 0;
    m.fast_host_hit = fast_host_hit ? 1 : 0;
    m.prof_mode = prof_mode;
    m.l1_bm = l1_bm; m.l1_bn = l1_bn; m.l1_bk = l1_bk; m.l1_stages = l1_stages;
    m.l2_bm = l2_bm; m.l2_bn = l2_bn; m.l2_bk = l2_bk; m.l2_stages = l2_stages;
    m.cfg_str = cfg_str;
}

inline void maybe_auto_dump(int device_idx, int prof_mode) {
    if (prof_mode <= 0 || device_idx < 0 || device_idx >= 16) return;
    const char* dump_dir = std::getenv("AGY_PROF_DUMP_DIR");
    if (dump_dir == nullptr || *dump_dir == '\0') return;
    auto& slot = device_slots()[device_idx];
    if (slot.dev_ptr == nullptr) return;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_idx).stream();
    cudaStreamSynchronize(stream);
    std::error_code ec;
    std::filesystem::create_directories(dump_dir, ec);
    HostLaunchMeta m;
    {
        std::lock_guard<std::mutex> lk(slot.mtx);
        m = slot.last_meta;
    }
    const std::string tag = std::getenv("AGY_PROF_TAG") ? std::getenv("AGY_PROF_TAG") : "run";
    const std::string base = std::format("{}/{}_rank{}_seq{:04d}", dump_dir, tag, m.rank_idx, m.seq_id);
    std::vector<uint8_t> host_bytes(slot.nbytes);
    if (cudaMemcpy(host_bytes.data(), slot.dev_ptr, slot.nbytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
        if (FILE* fp = std::fopen((base + ".agyprof.bin").c_str(), "wb")) {
            std::fwrite(host_bytes.data(), 1, host_bytes.size(), fp);
            std::fclose(fp);
        }
        if (FILE* jf = std::fopen((base + ".meta.json").c_str(), "wb")) {
            const std::string js = std::format(
                "{{\"host_prep_ns\":{},\"host_launch_api_ns\":{},\"seq_id\":{},\"device_idx\":{},\"rank_idx\":{},\"num_ranks\":{},"
                "\"num_experts\":{},\"num_tokens\":{},\"num_topk\":{},\"hidden\":{},\"intermediate_hidden\":{},"
                "\"is_fused\":{},\"fast_host_hit\":{},\"prof_mode\":{},\"l1_bm\":{},\"l1_bn\":{},\"l1_bk\":{},\"l1_stages\":{},"
                "\"l2_bm\":{},\"l2_bn\":{},\"l2_bk\":{},\"l2_stages\":{},\"cfg\":\"{}\"}}\n",
                m.host_prep_ns, m.host_launch_api_ns, m.seq_id, m.device_idx, m.rank_idx, m.num_ranks,
                m.num_experts, m.num_tokens, m.num_topk, m.hidden, m.intermediate_hidden,
                m.is_fused, m.fast_host_hit, m.prof_mode, m.l1_bm, m.l1_bn, m.l1_bk, m.l1_stages,
                m.l2_bm, m.l2_bn, m.l2_bk, m.l2_stages, m.cfg_str);
            std::fwrite(js.data(), 1, js.size(), jf);
            std::fclose(jf);
        }
    }
}

inline void reset_current_device() {
    const int dev = c10::cuda::current_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    ensure_and_clear_buffer(dev, stream);
}

inline torch::Tensor snapshot_bytes() {
    const int dev = c10::cuda::current_device();
    if (dev < 0 || dev >= 16) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    }
    auto& slot = device_slots()[dev];
    if (slot.dev_ptr == nullptr) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    cudaStreamSynchronize(stream);
    auto out = torch::empty(
        {static_cast<int64_t>(slot.nbytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    cudaMemcpy(out.data_ptr(), slot.dev_ptr, slot.nbytes, cudaMemcpyDeviceToHost);
    return out;
}

inline std::string last_host_meta_json() {
    const int dev = c10::cuda::current_device();
    if (dev < 0 || dev >= 16) return "{}";
    auto& slot = device_slots()[dev];
    std::lock_guard<std::mutex> lk(slot.mtx);
    const auto& m = slot.last_meta;
    return std::format(
        "{{\"host_prep_ns\":{},\"host_launch_api_ns\":{},\"seq_id\":{},\"device_idx\":{},\"rank_idx\":{},\"num_ranks\":{},"
        "\"num_experts\":{},\"num_tokens\":{},\"num_topk\":{},\"hidden\":{},\"intermediate_hidden\":{},"
        "\"is_fused\":{},\"fast_host_hit\":{},\"prof_mode\":{},\"l1_bm\":{},\"l1_bn\":{},\"l1_bk\":{},\"l1_stages\":{},"
        "\"l2_bm\":{},\"l2_bn\":{},\"l2_bk\":{},\"l2_stages\":{},\"cfg\":\"{}\"}}",
        m.host_prep_ns, m.host_launch_api_ns, m.seq_id, m.device_idx, m.rank_idx, m.num_ranks,
        m.num_experts, m.num_tokens, m.num_topk, m.hidden, m.intermediate_hidden,
        m.is_fused, m.fast_host_hit, m.prof_mode, m.l1_bm, m.l1_bn, m.l1_bk, m.l1_stages,
        m.l2_bm, m.l2_bn, m.l2_bk, m.l2_stages, m.cfg_str);
}

}  // namespace deep_gemm::agy_prof
