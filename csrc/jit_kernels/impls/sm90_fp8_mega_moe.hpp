#pragma once

#include <format>
#include <torch/python.h>
#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/nv_moe_workspace.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_mega_moe.hpp"
#include "sm90_agy_profiler_host.hpp"

namespace deep_gemm {

// SM90 uses FP8 operands, float scale factors, register-resident WGMMA
// accumulators, and one CTA per scheduled work item.

class SM90FP8MegaMoERuntime final {
public:
    enum class KernelPhase {
        Linear1,
        Linear2,
        Fused
    };

    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool bf16_scaled_accum;
        bool l1_nmajor_for_fused = false;
        KernelPhase kernel_phase;
        MegaMoESM90Config config;

        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const float* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l2_weights_sf;

        deep_jit::cuda::LaunchOptions options;
        int agy_prof_mode = 0;
        uint64_t* agy_prof_ptr = nullptr;
    };

    using KernelHandleT = std::decay_t<decltype(jit->compile(std::string{}, std::string{}))>;

    static KernelHandleT compile_only(const std::string& tag, const Args& args) {
        const char* kernel_symbol =
            args.kernel_phase == KernelPhase::Linear1 ? "sm90_fp8_mega_moe_l1_impl" :
            (args.kernel_phase == KernelPhase::Linear2 ? "sm90_fp8_mega_moe_l2_impl" :
             "sm90_fp8_mega_moe_fused_impl");
        const auto phase_template_args =
            args.kernel_phase == KernelPhase::Linear1 ?
                std::format(",\n        {}", args.config.nmajor_schedule ? "true" : "false") :
            (args.kernel_phase == KernelPhase::Linear2 ?
                std::format(",\n        {}, {}, {}",
                            args.config.direct_l2_scatter ? "true" : "false",
                            args.config.nmajor_schedule ? "true" : "false",
                            args.config.one_warp_cleanup ? "true" : "false") :
                std::format(",\n        {}, {}, {}, {}",
                            args.config.direct_l2_scatter ? "true" : "false",
                            args.config.nmajor_schedule ? "true" : "false",
                            args.config.one_warp_cleanup ? "true" : "false",
                            args.l1_nmajor_for_fused ? "true" : "false"));
        const Sm90MoeHeuristicInput compile_heur_input {
            args.config.num_sms,
            args.num_ranks, args.num_experts, args.num_experts / std::max(1, args.num_ranks),
            args.num_max_tokens_per_rank, args.num_tokens, args.num_topk,
            args.hidden, args.intermediate_hidden,
            args.config.num_padded_sf_pool_tokens
        };
        const auto agy_spec = get_agy_sm90_spec(compile_heur_input);
        const auto agy_defs = get_agy_sm90_jit_defines(agy_spec);
        const std::string prof_tag = args.agy_prof_mode > 0 ? std::format("_agyprof{}", args.agy_prof_mode) : "";
        const std::string prof_defs = args.agy_prof_mode > 0 ? std::format("#define AGY_PROF {}\n", args.agy_prof_mode) : "";
        return jit->compile(
            tag + get_agy_sm90_tag_suffix(agy_spec) + prof_tag,
            prof_defs + agy_defs + std::format(R"(
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {}{}
    >);
}};
)",
    kernel_symbol,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.sf_pool_stride_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.num_sms, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.config.swap_ab ? "true" : "false",
    args.bf16_scaled_accum ? "true" : "false",
    phase_template_args));

    }

    static void launch_compiled(const KernelHandleT& kernel, const Args& args) {
        if (args.agy_prof_mode > 0) {
            jit->launch(kernel, args.options,
                args.y,
                args.cumulative_local_expert_recv_stats,
                args.num_tokens,
                args.sym_buffer_ptrs,
                args.tensor_map_l1_acts,
                args.tensor_map_l1_acts_sf,
                args.tensor_map_l1_weights,
                args.l1_weights_sf,
                args.tensor_map_l1_output,
                args.tensor_map_l2_acts,
                args.tensor_map_l2_acts_sf,
                args.tensor_map_l2_weights,
                args.l2_weights_sf,
                args.agy_prof_ptr
            );
        } else {
            jit->launch(kernel, args.options,
                args.y,
                args.cumulative_local_expert_recv_stats,
                args.num_tokens,
                args.sym_buffer_ptrs,
                args.tensor_map_l1_acts,
                args.tensor_map_l1_acts_sf,
                args.tensor_map_l1_weights,
                args.l1_weights_sf,
                args.tensor_map_l1_output,
                args.tensor_map_l2_acts,
                args.tensor_map_l2_acts_sf,
                args.tensor_map_l2_weights,
                args.l2_weights_sf
            );
        }
    }

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = compile_only(tag, args);
        launch_compiled(kernel, args);
    }
};

struct AgyFastPlanKey {
    int device_idx = -1;
    int rank_idx = -1;
    int num_ranks = 0;
    int num_experts = 0;
    int num_max_tokens_per_rank = 0;
    int num_tokens = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    uint32_t clamp_bits = 0;
    bool fast_math = false;
    const void* sym_buf_ptr = nullptr;
    const void* l1_w_ptr = nullptr;
    const void* l2_w_ptr = nullptr;
    std::array<int64_t, layout::kNumMaxRanks> sym_ptrs{};
    uint64_t agy_hash = 0;
    int prof_mode = 0;

    bool operator==(const AgyFastPlanKey&) const = default;
};

struct AgyFastPlanEntry {
    bool valid = false;
    AgyFastPlanKey key{};
    bool is_fused = false;
    layout::SymBuffer<> sym_buffer_arg{};
    CUtensorMap tensor_map_l1_acts{};
    CUtensorMap tensor_map_l1_acts_sf{};
    CUtensorMap tensor_map_l1_weights{};
    CUtensorMap tensor_map_l1_output{};
    CUtensorMap tensor_map_l2_acts{};
    CUtensorMap tensor_map_l2_acts_sf{};
    CUtensorMap tensor_map_l2_weights{};
    std::optional<SM90FP8MegaMoERuntime::KernelHandleT> kernel_0;
    deep_jit::cuda::LaunchOptions options_0{};
    std::optional<SM90FP8MegaMoERuntime::KernelHandleT> kernel_1;
    deep_jit::cuda::LaunchOptions options_1{};
    int l1_bm = 0, l1_bn = 0, l1_bk = 0, l1_stages = 0;
    int l2_bm = 0, l2_bn = 0, l2_bk = 0, l2_stages = 0;
};

static inline std::pair<std::array<AgyFastPlanEntry, 32>&, size_t&> get_agy_fast_plan_tl_cache() {
    thread_local std::array<AgyFastPlanEntry, 32> cache{};
    thread_local size_t write_cursor = 0;
    return {cache, write_cursor};
}

static inline AgyFastPlanKey make_agy_fast_plan_key(
    const AgySm90KnobSpec& agy_spec,
    const int device_idx,
    const int rank_idx,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const void* sym_buf_ptr,
    const void* l1_w_ptr,
    const void* l2_w_ptr,
    const int num_max_tokens_per_rank,
    const int num_experts_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden,
    const float activation_clamp,
    const bool fast_math,
    const int prof_mode = 0
) {
    AgyFastPlanKey k{};
    k.device_idx = device_idx;
    k.rank_idx = rank_idx;
    k.num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    k.num_experts = num_experts_per_rank * k.num_ranks;
    k.num_max_tokens_per_rank = num_max_tokens_per_rank;
    k.num_tokens = num_tokens;
    k.num_topk = num_topk;
    k.hidden = hidden;
    k.intermediate_hidden = intermediate_hidden;
    std::memcpy(&k.clamp_bits, &activation_clamp, sizeof(uint32_t));
    k.fast_math = fast_math;
    k.sym_buf_ptr = sym_buf_ptr;
    k.l1_w_ptr = l1_w_ptr;
    k.l2_w_ptr = l2_w_ptr;
    for (size_t i = 0; i < sym_buffer_ptrs.size() and i < k.sym_ptrs.size(); ++ i)
        k.sym_ptrs[i] = sym_buffer_ptrs[i];
    k.agy_hash = agy_spec.raw_hash;
    k.prof_mode = prof_mode;
    return k;
}

static inline bool try_launch_agy_sm90_fast_plan(
    const AgySm90KnobSpec& agy_spec,
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int rank_idx,
    const int num_max_tokens_per_rank,
    const int num_experts_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden,
    const float activation_clamp,
    const bool fast_math,
    const int agy_prof_mode = 0,
    uint64_t* const agy_prof_dev_ptr = nullptr
) {
    auto [cache, cursor] = get_agy_fast_plan_tl_cache();
    (void)cursor;
    const auto key = make_agy_fast_plan_key(
        agy_spec, y.get_device(), rank_idx, sym_buffer_ptrs,
        sym_buffer.data_ptr(), l1_weights.data_ptr(), l2_weights.data_ptr(),
        num_max_tokens_per_rank, num_experts_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, activation_clamp, fast_math, agy_prof_mode);

    for (const auto& entry : cache) {
        if (entry.valid and entry.key == key) {
            void* y_ptr = y.data_ptr();
            int* stats_ptr = cumulative_local_expert_recv_stats.has_value()
                ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
            const float* l1_wsf = l1_weights_sf.data_ptr<float>();
            const float* l2_wsf = l2_weights_sf.data_ptr<float>();
            const uint64_t t_before_launch = agy_prof_mode > 0 ? agy_prof::steady_now_ns() : 0ull;
            if (agy_prof_mode > 0) {
                jit->launch(*entry.kernel_0, entry.options_0,
                    y_ptr, stats_ptr, num_tokens, entry.sym_buffer_arg,
                    entry.tensor_map_l1_acts, entry.tensor_map_l1_acts_sf,
                    entry.tensor_map_l1_weights, l1_wsf,
                    entry.tensor_map_l1_output,
                    entry.tensor_map_l2_acts, entry.tensor_map_l2_acts_sf,
                    entry.tensor_map_l2_weights, l2_wsf,
                    agy_prof_dev_ptr);
                if (not entry.is_fused) {
                    jit->launch(*entry.kernel_1, entry.options_1,
                        y_ptr, stats_ptr, num_tokens, entry.sym_buffer_arg,
                        entry.tensor_map_l1_acts, entry.tensor_map_l1_acts_sf,
                        entry.tensor_map_l1_weights, l1_wsf,
                        entry.tensor_map_l1_output,
                        entry.tensor_map_l2_acts, entry.tensor_map_l2_acts_sf,
                        entry.tensor_map_l2_weights, l2_wsf,
                        agy_prof_dev_ptr);
                }
            } else {
                jit->launch(*entry.kernel_0, entry.options_0,
                    y_ptr, stats_ptr, num_tokens, entry.sym_buffer_arg,
                    entry.tensor_map_l1_acts, entry.tensor_map_l1_acts_sf,
                    entry.tensor_map_l1_weights, l1_wsf,
                    entry.tensor_map_l1_output,
                    entry.tensor_map_l2_acts, entry.tensor_map_l2_acts_sf,
                    entry.tensor_map_l2_weights, l2_wsf);
                if (not entry.is_fused) {
                    jit->launch(*entry.kernel_1, entry.options_1,
                        y_ptr, stats_ptr, num_tokens, entry.sym_buffer_arg,
                        entry.tensor_map_l1_acts, entry.tensor_map_l1_acts_sf,
                        entry.tensor_map_l1_weights, l1_wsf,
                        entry.tensor_map_l1_output,
                        entry.tensor_map_l2_acts, entry.tensor_map_l2_acts_sf,
                        entry.tensor_map_l2_weights, l2_wsf);
                }
            }
            if (agy_prof_mode > 0) {
                const uint64_t t_after_launch = agy_prof::steady_now_ns();
                agy_prof::record_launch_timing(
                    y.get_device(), t_before_launch, t_after_launch,
                    rank_idx, key.num_ranks, key.num_experts, num_tokens, num_topk,
                    hidden, intermediate_hidden, entry.is_fused, true, agy_prof_mode,
                    entry.l1_bm, entry.l1_bn, entry.l1_bk, entry.l1_stages,
                    entry.l2_bm, entry.l2_bn, entry.l2_bk, entry.l2_stages,
                    agy_spec.raw);
            }
            return true;
        }
    }
    return false;
}

static void sm90_fp8_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const torch::Tensor* sym_buffer_opt = nullptr,
    const int agy_prof_mode = 0,
    uint64_t* const agy_prof_dev_ptr = nullptr
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Resolve hardware, generic fallback, phase schedules, and numerical modes
    // once. The runtime only consumes the resulting complete launch config.
    const int num_sms = runtime->get_num_sms();
    const Sm90MoeHeuristicInput heuristic_input {
        num_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        num_padded_sf_pool_tokens
    };
    const auto launch_config = select_mega_moe_sm90(heuristic_input);
    const auto& l1_config = launch_config.l1;
    const auto& l2_config = launch_config.l2;
    const auto combine_smem_before_barriers =
        get_sm90_moe_pre_barrier_smem_size_for_combine(
            heuristic_input, l2_config);
    if (not layout::is_sm90_moe_combine_vectorization_legal(
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(l2_config.num_epilogue_threads / 32),
            combine_smem_before_barriers,
            static_cast<uint32_t>(sizeof(nv_bfloat16)),
            static_cast<uint32_t>(sizeof(nv_bfloat16))))
        DG_HOST_UNREACHABLE(
            "SM90 FP8 MegaMoE hidden size is incompatible with the selected combine vectorization");

    // Carbon: resolve the AGY knob spec before tensormap construction.
    const auto agy_spec = get_agy_sm90_spec(heuristic_input);
    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    constexpr int kTMATileK = 128;
    constexpr int kMaxTMABoxN = 256;
    const int num_epilogue_warpgroups_h = l1_config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        l1_config.block_m, l1_config.block_n, num_epilogue_warpgroups_h);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const int wg_l1_out_block_n = wg_block_n / 2;
    const bool use_adapt_bm_l1 = agy_spec.enabled and agy_spec.get_int("adapt_bm", 0) > 0 and
                                 l1_config.swap_ab and l1_config.block_m == 64 and
                                 layout::get_sm90_moe_warpgroup_layout(
                                     l1_config.block_m, l1_config.block_n,
                                     l1_config.num_epilogue_threads / 128).split_n;
    const bool use_adapt_bm_l2 = agy_spec.enabled and agy_spec.get_int("adapt_bm", 0) > 0 and
                                 l2_config.swap_ab and l2_config.block_m == 64 and
                                 layout::get_sm90_moe_warpgroup_layout(
                                     l2_config.block_m, l2_config.block_n,
                                     l2_config.num_epilogue_threads / 128).split_n;
    // A BK256 pipeline stage is represented in shared memory as two adjacent
    // independently-swizzled BK128 TMA tiles. Keep the tensor-map box at 128
    // and issue two copies; the kernel config/scheduler still advances by 256.
    const int l1_tma_block_k = std::min(l1_config.block_k, kTMATileK);
    const int l2_tma_block_k = std::min(l2_config.block_k, kTMATileK);
    const int l1_tma_block_n = std::min(l1_config.block_n, kMaxTMABoxN);
    const int l2_tma_block_n = std::min(l2_config.block_n, kMaxTMABoxN);
    DG_HOST_ASSERT(l1_config.block_k % l1_tma_block_k == 0 and
                   l2_config.block_k % l2_tma_block_k == 0);
    DG_HOST_ASSERT(l1_config.block_n <= 2 * kMaxTMABoxN and
                   l2_config.block_n <= 2 * kMaxTMABoxN and
                   l1_config.block_n % l1_tma_block_n == 0 and
                   l2_config.block_n % l2_tma_block_n == 0);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, l1_config.num_max_pool_tokens,
                                                     l1_tma_block_k, use_adapt_bm_l1 ? 16 : l1_config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     128);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        l1_config.sf_pool_stride_tokens, hidden,
                                                        l1_config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        l1_tma_block_k, l1_tma_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        128);
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The default TMA store is issued per warpgroup, each writing a WG_BLOCK_M
    // row tile. In split-N mode, two WGs produce different N halves of the same
    // M rows, then one TMA store writes the full 64x128 post-SwiGLU tile.
    const int l1_output_box_n = wg_layout.split_n ? l1_config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m = wg_layout.split_n ? l1_config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, l1_config.num_max_pool_tokens,
                                                       l1_output_box_n, use_adapt_bm_l1 ? 16 : l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, l2_config.num_max_pool_tokens,
                                                     l2_tma_block_k, use_adapt_bm_l2 ? 16 : l2_config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     128);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        l2_config.sf_pool_stride_tokens, intermediate_hidden,
                                                        l2_config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        l2_tma_block_k, l2_tma_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        128);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const bool bf16_scaled_accum = launch_config.numerical.bf16_scaled_accum;
    const SM90FP8MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .bf16_scaled_accum = bf16_scaled_accum,
        .kernel_phase = SM90FP8MegaMoERuntime::KernelPhase::Linear1,
        .config = l1_config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = l1_weights_sf.data_ptr<float>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = l2_weights_sf.data_ptr<float>(),
        .options = {},
        .agy_prof_mode = agy_prof_mode,
        .agy_prof_ptr = agy_prof_dev_ptr
    };
    const auto make_phase_args = [&](const SM90FP8MegaMoERuntime::KernelPhase kernel_phase) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        split_args.l1_nmajor_for_fused = l1_config.nmajor_schedule;
        const bool use_l2_cfg =
            kernel_phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2 or
            kernel_phase == SM90FP8MegaMoERuntime::KernelPhase::Fused;
        split_args.config = use_l2_cfg ? l2_config : l1_config;
        const bool use_mcast2 =
            kernel_phase == SM90FP8MegaMoERuntime::KernelPhase::Fused and
            agy_spec.enabled and agy_spec.get_int("mcast", 1) == 2 and
            (split_args.config.num_sms % 2 == 0);
        split_args.options = {
            .num_smem_bytes = split_args.config.smem_size,
            .grid_dim = dim3(split_args.config.num_sms, 1, 1),
            .block_dim = dim3(split_args.config.num_dispatch_threads +
                              split_args.config.num_non_epilogue_threads +
                              split_args.config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(use_mcast2 ? 2 : 1, 1, 1),
            .enable_pdl = false,
        };
        return split_args;
    };

    const bool record_fast_plan =
        sym_buffer_opt != nullptr and agy_spec.enabled and agy_spec.get_int("fast_host", 0) > 0;

    if (launch_config.fused_single_kernel) {
        const auto fused_args = make_phase_args(SM90FP8MegaMoERuntime::KernelPhase::Fused);
        const auto k_fused = SM90FP8MegaMoERuntime::compile_only("sm90_fp8_mega_moe_fused_impl", fused_args);
        if (record_fast_plan) {
            auto [cache, cursor] = get_agy_fast_plan_tl_cache();
            auto& dst = cache[cursor & 31u];
            cursor = (cursor + 1u) & 31u;
            dst.valid = true;
            dst.key = make_agy_fast_plan_key(
                agy_spec, y.get_device(), rank_idx, sym_buffer_ptrs,
                sym_buffer_opt->data_ptr(), l1_weights.data_ptr(), l2_weights.data_ptr(),
                num_max_tokens_per_rank, num_experts_per_rank, num_tokens, num_topk,
                hidden, intermediate_hidden, activation_clamp, fast_math, agy_prof_mode);
            dst.is_fused = true;
            dst.sym_buffer_arg = fused_args.sym_buffer_ptrs;
            dst.tensor_map_l1_acts = tensor_map_l1_acts;
            dst.tensor_map_l1_acts_sf = tensor_map_l1_acts_sf;
            dst.tensor_map_l1_weights = tensor_map_l1_weights;
            dst.tensor_map_l1_output = tensor_map_l1_output;
            dst.tensor_map_l2_acts = tensor_map_l2_acts;
            dst.tensor_map_l2_acts_sf = tensor_map_l2_acts_sf;
            dst.tensor_map_l2_weights = tensor_map_l2_weights;
            dst.kernel_0 = k_fused;
            dst.options_0 = fused_args.options;
            dst.kernel_1.reset();
            dst.l1_bm = l1_config.block_m; dst.l1_bn = l1_config.block_n; dst.l1_bk = l1_config.block_k; dst.l1_stages = l1_config.num_stages;
            dst.l2_bm = l2_config.block_m; dst.l2_bn = l2_config.block_n; dst.l2_bk = l2_config.block_k; dst.l2_stages = l2_config.num_stages;
        }
        const uint64_t t_before_launch = agy_prof_mode > 0 ? agy_prof::steady_now_ns() : 0ull;
        SM90FP8MegaMoERuntime::launch_compiled(k_fused, fused_args);
        if (agy_prof_mode > 0) {
            const uint64_t t_after_launch = agy_prof::steady_now_ns();
            agy_prof::record_launch_timing(
                y.get_device(), t_before_launch, t_after_launch,
                rank_idx, num_ranks, num_experts, num_tokens, num_topk,
                hidden, intermediate_hidden, true, false, agy_prof_mode,
                l1_config.block_m, l1_config.block_n, l1_config.block_k, l1_config.num_stages,
                l2_config.block_m, l2_config.block_n, l2_config.block_k, l2_config.num_stages,
                agy_spec.enabled ? agy_spec.raw : "__OFFICIAL__");
        }
    } else {
        const auto l1_args = make_phase_args(SM90FP8MegaMoERuntime::KernelPhase::Linear1);
        const auto l2_args = make_phase_args(SM90FP8MegaMoERuntime::KernelPhase::Linear2);
        const auto k_l1 = SM90FP8MegaMoERuntime::compile_only("sm90_fp8_mega_moe_l1_impl", l1_args);
        const auto k_l2 = SM90FP8MegaMoERuntime::compile_only("sm90_fp8_mega_moe_l2_impl", l2_args);
        if (record_fast_plan) {
            auto [cache, cursor] = get_agy_fast_plan_tl_cache();
            auto& dst = cache[cursor & 31u];
            cursor = (cursor + 1u) & 31u;
            dst.valid = true;
            dst.key = make_agy_fast_plan_key(
                agy_spec, y.get_device(), rank_idx, sym_buffer_ptrs,
                sym_buffer_opt->data_ptr(), l1_weights.data_ptr(), l2_weights.data_ptr(),
                num_max_tokens_per_rank, num_experts_per_rank, num_tokens, num_topk,
                hidden, intermediate_hidden, activation_clamp, fast_math, agy_prof_mode);
            dst.is_fused = false;
            dst.sym_buffer_arg = l1_args.sym_buffer_ptrs;
            dst.tensor_map_l1_acts = tensor_map_l1_acts;
            dst.tensor_map_l1_acts_sf = tensor_map_l1_acts_sf;
            dst.tensor_map_l1_weights = tensor_map_l1_weights;
            dst.tensor_map_l1_output = tensor_map_l1_output;
            dst.tensor_map_l2_acts = tensor_map_l2_acts;
            dst.tensor_map_l2_acts_sf = tensor_map_l2_acts_sf;
            dst.tensor_map_l2_weights = tensor_map_l2_weights;
            dst.kernel_0 = k_l1;
            dst.options_0 = l1_args.options;
            dst.kernel_1 = k_l2;
            dst.options_1 = l2_args.options;
            dst.l1_bm = l1_config.block_m; dst.l1_bn = l1_config.block_n; dst.l1_bk = l1_config.block_k; dst.l1_stages = l1_config.num_stages;
            dst.l2_bm = l2_config.block_m; dst.l2_bn = l2_config.block_n; dst.l2_bk = l2_config.block_k; dst.l2_stages = l2_config.num_stages;
        }
        const uint64_t t_before_launch = agy_prof_mode > 0 ? agy_prof::steady_now_ns() : 0ull;
        SM90FP8MegaMoERuntime::launch_compiled(k_l1, l1_args);
        SM90FP8MegaMoERuntime::launch_compiled(k_l2, l2_args);
        if (agy_prof_mode > 0) {
            const uint64_t t_after_launch = agy_prof::steady_now_ns();
            agy_prof::record_launch_timing(
                y.get_device(), t_before_launch, t_after_launch,
                rank_idx, num_ranks, num_experts, num_tokens, num_topk,
                hidden, intermediate_hidden, false, false, agy_prof_mode,
                l1_config.block_m, l1_config.block_n, l1_config.block_k, l1_config.num_stages,
                l2_config.block_m, l2_config.block_n, l2_config.block_k, l2_config.num_stages,
                agy_spec.enabled ? agy_spec.raw : "__OFFICIAL__");
        }
    }
}

} // namespace deep_gemm
