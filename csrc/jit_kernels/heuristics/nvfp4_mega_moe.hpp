#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>

#include <deep_gemm/layout/nvfp4_mega_moe.cuh>
#include <deep_gemm/scheduler/nvfp4_mega_moe.cuh>
#include <deep_gemm/scheduler/weave_cost_model.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"
#include "weave_knob_config.hpp"

namespace deep_gemm::nvfp4 {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    // Unified Weave knob request resolved for this launch.
    weave::WeaveKnobs knobs;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull
           << ", dispatch_warps=" << config.knobs.dispatch_warps
           << ", ring_blocks=" << config.knobs.ring_blocks << ")";
        return os;
    }
};

static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int, int> {
        float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        if (num_expected_tokens_per_expert <= 8.5) {
            // Really small token-per-expert (e.g. RL long-tail rollout), use the smallest block_m and larger BLOCK_K for less synchronization
            return {2, 16, 8, 256, 2};
        } else if (num_expected_tokens_per_expert <= 16.5) {
            // Small batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 128
            return {2, 32, 16, 128, 2};
        } else if (num_expected_tokens_per_expert <= 32.5) {
            // Medium batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 256
            return {2, 64, 32, 128, 1};
        } else if (num_expected_tokens_per_expert <= 64.5) {
            // Large batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 512
            return {2, 96, 16, 128, 2};
        } else if (num_expected_tokens_per_expert <= 96.5) {
            // Medium batch size, Medium EP, decoding, e.g. 6/384 experts, EP16, bsz 256, or EP32, bsz128
            return {2, 128, 32, 128, 2};
        } else {
            // Prefill, or large EP decoding
            return {2, 192, 32, 128, 2};
        }
    }();
    // `block_k` above is in bytes; each byte stores two E2M1 elements.
    block_k *= 2;

    DG_HOST_ASSERT(std::any_of(
        layout::nvfp4::kCandidateBlockM, layout::nvfp4::kCandidateBlockM + layout::nvfp4::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int load_block_m = block_m / 2;
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 4) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::nvfp4::TaskInfo<true>));
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;
    const int smem_weave_policy = align(
        static_cast<int>(sizeof(weave::PolicyDecision)) + 5 * static_cast<int>(sizeof(uint32_t)),
        16);
    const int smem_tmem_ptr = 4;
    const int smem_sfa_per_stage = sf_block_m * (block_k / gran_k);
    const int smem_sfb_per_stage = sf_block_n * (block_k / gran_k);
    const int smem_a_size_per_stage = load_block_m * block_k / 2;
    const int smem_b_size_per_stage = block_n * block_k / 2;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers + smem_task_info +
        smem_weave_policy + smem_tmem_ptr;
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

// The NVFP4 kernel compiles one BLOCK_M envelope for both GEMM phases, so
// distinct L1/L2 requests compile at max(L1, L2); the per-phase requests stay
// visible to the CTA-0 cost model through the JIT defines.
static int resolve_block_m_envelope(const weave::WeaveKnobs& knobs, const int block_m_auto) {
    const int req_bm_l1 = knobs.block_m_l1 != 0u ? static_cast<int>(knobs.block_m_l1) : 0;
    const int req_bm_l2 = knobs.block_m_l2 != 0u ? static_cast<int>(knobs.block_m_l2) : 0;
    const int block_m = (req_bm_l1 > 0 and req_bm_l2 > 0)
        ? std::max(req_bm_l1, req_bm_l2)
        : static_cast<int>(weave::resolve_block_m(
              knobs, knobs.block_m_l1, knobs.block_m_l2, static_cast<uint32_t>(block_m_auto)));
    DG_HOST_ASSERT(std::any_of(
        layout::nvfp4::kCandidateBlockM, layout::nvfp4::kCandidateBlockM + layout::nvfp4::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return block_m;
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens) {
    const auto [cluster_size, block_m_auto, store_block_m_auto, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    DG_HOST_ASSERT(num_ring_tokens % block_m_auto == 0);

    // Unified WEAVE_* knob request.
    const auto knobs = weave_knobs::from_env(
        weave::HardwareArch::B300_SM100, weave::PrecisionMode::NVFP4);
    const int block_m = resolve_block_m_envelope(knobs, block_m_auto);
    int store_block_m = store_block_m_auto;
    if (block_m != block_m_auto) {
        store_block_m = block_m <= 16 ? 8 : block_m <= 64 ? 16 : 32;
    }
    // One shared store height: asymmetric L1/L2 store requests cannot be
    // honored by the single STORE_BLOCK_M template and are rejected.
    DG_HOST_ASSERT(knobs.store_block_m_l1 == 0u or knobs.store_block_m_l2 == 0u or
                   knobs.store_block_m_l1 == knobs.store_block_m_l2);
    if (knobs.store_block_m_l1 != 0u)
        store_block_m = static_cast<int>(knobs.store_block_m_l1);
    else if (knobs.store_block_m_l2 != 0u)
        store_block_m = static_cast<int>(knobs.store_block_m_l2);
    const int num_epi_wgs = std::max(1, num_epilogue_threads / 128);
    const int wg_bm = block_m / num_epi_wgs;
    while (store_block_m > 8 and (store_block_m > wg_bm or wg_bm % store_block_m != 0))
        store_block_m -= 8;
    store_block_m = std::max(8, store_block_m);
    DG_HOST_ASSERT(num_ring_tokens % block_m == 0);

    const int block_n = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = align(block_n, 128);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = 16;
    const uint32_t dispatch_warps = weave::resolve_dispatch_warps(knobs, 4u);
    DG_HOST_ASSERT(dispatch_warps == 4u or dispatch_warps == 8u);
    const int num_dispatch_threads = static_cast<int>(dispatch_warps) * 32;
    const int num_non_epilogue_threads = 128;

    constexpr int kPullThreshold = 4096;
    int num_bytes_per_pull = hidden / 2;
    while (num_bytes_per_pull > kPullThreshold) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_ring_tokens, num_sf_ring_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull,
        knobs
    };

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "NVFP4MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm::nvfp4
