#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/weave_knobs.cuh>
#include <deep_gemm/scheduler/weave_cost_model.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"

namespace deep_gemm {

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
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    DG_HOST_ASSERT(mma_type_str == "fp8xfp4");
    return MmaKind::MXFP8FP4;
}

static int get_num_mma_elem_bytes(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::BF16 ? 2 : 1;
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4;
}

static int get_tune_env(const char* primary, const char* fallback, const int& default_val = 0) {
    const int v = get_env<int>(primary, -999999);
    if (v != -999999)
        return v;
    if (fallback != nullptr)
        return get_env<int>(fallback, default_val);
    return default_val;
}

static bool is_d3_opt_enabled(const MmaKind& mma_kind) {
    if (mma_kind != MmaKind::MXFP8FP4)
        return false;
    return get_env<int>("DG_ISOLATED_D3_OPT", 0) != 0 or
           get_tune_env("DG_TUNE_D3_BM", "DG_D3_BM", 0) > 0 or
           get_tune_env("DG_TUNE_D3_BN", "DG_D3_BN", 0) > 0 or
           get_tune_env("DG_TUNE_D3_BK", "DG_D3_BK", 0) > 0 or
           get_tune_env("DG_TUNE_D3_STORE_M", "DG_D3_STORE_M", 0) > 0 or
           get_tune_env("DG_TUNE_D3_DYN_EPI", "DG_D3_DYN_EPI", 0) > 0 or
           get_tune_env("DG_TUNE_D3_DYN_TMA_A", "DG_D3_DYN_TMA_A", 0) > 0 or
           get_tune_env("DG_TUNE_D3_OVERLAP_SMEM", "DG_D3_OVERLAP_EXPERT_SMEM", 0) > 0;
}

static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind,
    const int& hidden = 0,
    const int& intermediate_hidden = 0) {
    const bool d3_opt = is_d3_opt_enabled(mma_kind);

    auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int, int> {
        float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        if (not d3_opt) {
            if (num_expected_tokens_per_expert <= 8.5) {
                return {2, 16, 8, 256, 2};
            } else if (num_expected_tokens_per_expert <= 16.5) {
                return {2, 32, 16, 128, 2};
            } else if (num_expected_tokens_per_expert <= 32.5) {
                return {2, 64, 32, 128, 1};
            } else if (num_expected_tokens_per_expert <= 64.5) {
                return {2, 96, 16, 128, 2};
            } else if (num_expected_tokens_per_expert <= 96.5) {
                return {2, 128, 32, 128, 2};
            } else {
                return {2, 192, 32, 128, 2};
            }
        } else {
            if (num_expected_tokens_per_expert <= 16.5f) {
                return {2, 32, 16, 128, 2};
            } else if (num_expected_tokens_per_expert <= 32.5f) {
                return {2, 64, 32, 128, 1};
            } else if (num_expected_tokens_per_expert <= 96.5f) {
                return {2, 128, 32, 128, 2};
            } else {
                return {2, 256, 32, 128, 2};
            }
        }
    }();
    block_k /= get_num_mma_elem_bytes(mma_kind);

    block_m = get_env<int>("DG_MEGA_BLOCK_M", block_m);
    store_block_m = get_env<int>("DG_MEGA_STORE_BLOCK_M", store_block_m);
    block_k = get_env<int>("DG_MEGA_BLOCK_K", block_k);
    num_epilogue_warpgroups = get_env<int>("DG_MEGA_EPILOGUE_WARPGROUPS", num_epilogue_warpgroups);

    const int d3_bm = get_tune_env("DG_TUNE_D3_BM", "DG_D3_BM", 0);
    const int d3_store_m = get_tune_env("DG_TUNE_D3_STORE_M", "DG_D3_STORE_M", 0);
    const int d3_bk = get_tune_env("DG_TUNE_D3_BK", "DG_D3_BK", 0);
    const int d3_epi_wg = get_tune_env("DG_TUNE_D3_EPI_WG", "DG_D3_EPI_WG", 0);
    if (d3_bm > 0) block_m = d3_bm;
    if (d3_epi_wg > 0) num_epilogue_warpgroups = d3_epi_wg;
    if (d3_store_m > 0) {
        store_block_m = d3_store_m;
    } else if (store_block_m * num_epilogue_warpgroups > block_m) {
        store_block_m = std::max(8, block_m / num_epilogue_warpgroups);
    }
    if (d3_bk > 0) block_k = d3_bk;

    // Axis 2 (D2): Modes 1..8 use 2 epilogue warpgroups (8E = 256 threads); Modes 11..15 keep default
    const int d2_mode = get_tune_env("DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
    if (1 <= d2_mode and d2_mode <= 8) {
        num_epilogue_warpgroups = 2;
        if (store_block_m * num_epilogue_warpgroups > block_m)
            store_block_m = std::max(8, block_m / num_epilogue_warpgroups);
    }

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
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
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind) {
    constexpr int kSmemAlignment = 1024;
    const int num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);

    const int default_accum_stages = ((block_n / 128) * block_m >= 256) ? 1 : 2;
    const int num_epilogue_stages = get_tune_env("DG_D3_ACCUM_STAGES", "DG_MEGA_ACCUM_STAGES", default_accum_stages);
    DG_HOST_ASSERT(1 <= num_epilogue_stages and num_epilogue_stages <= 4);
    const int num_tma_store_stages = get_tune_env("DG_D3_TMA_STORE_STAGES", "DG_MEGA_TMA_STORE_STAGES", 2);
    DG_HOST_ASSERT(1 <= num_tma_store_stages and num_tma_store_stages <= 8);
    const int kNumScheduleStages = get_tune_env("DG_TUNE_D3_SCHED_STAGES", "DG_D3_SCHED_STAGES", 2);

    const int d2_mode = get_tune_env("DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
    const int d2_reg_pull_bytes = get_tune_env("DG_TUNE_D2_REG_PULL_BYTES", "DG_ISOLATED_D2_REG_PULL_BYTES", 0);
    const bool d2_all_reg_pull = is_mma_with_sf(mma_kind) and (d2_reg_pull_bytes > 0);
    const bool d2_split_hybrid = is_mma_with_sf(mma_kind) and (not d2_all_reg_pull) and
        ((5 <= d2_mode and d2_mode <= 8) or d2_mode == 14);
    const int num_tma_dispatch_warps =
        d2_all_reg_pull ? 0 : (d2_split_hybrid ? std::min(4, num_dispatch_warps) : num_dispatch_warps);
    const int num_dispatch_barrier_slots =
        d2_all_reg_pull ? 1 : num_tma_dispatch_warps;

    const int d3_overlap_smem = get_tune_env("DG_TUNE_D3_OVERLAP_SMEM", "DG_D3_OVERLAP_EXPERT_SMEM", 0);
    const int d7_scatter_mode = get_tune_env("DG_TUNE_D7_SCATTER_MODE", "DG_ISOLATED_D7_SCATTER", 0);
    const int d7_smem_alias = get_env<int>("DG_TUNE_D7_SMEM_ALIAS", 1);
    const bool d7_handoff_active = is_mma_with_sf(mma_kind) and (d7_scatter_mode == 1 or
        (get_env<int>("DG_TUNE_D7_SCATTER_MODE", -1) == -1 and d7_scatter_mode != 0));

    const int load_block_m = block_m / 2;
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = (num_tma_dispatch_warps > 0 and num_bytes_per_pull > 0) ? align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_tma_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment) : 0;

    int smem_dispatch_size = 0;
    if (d3_overlap_smem != 0 and (not d7_handoff_active or d7_smem_alias == 0)) {
        smem_dispatch_size = std::max(smem_expert_count_size, smem_send_buffers_size);
    } else if (d2_all_reg_pull and d7_handoff_active and d7_smem_alias != 0) {
        smem_dispatch_size = smem_expert_count_size + 8192;
    } else {
        smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;
    }

    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_d7_dedicated = (d7_handoff_active and d7_smem_alias == 0)
        ? (2 * num_epilogue_warpgroups * 16 * 256)
        : 0;

    const int l1_out_block_n = is_mma_with_sf(mma_kind) ? 64 : (block_n / 2);
    const int l2_out_block_n = is_mma_with_sf(mma_kind) ? ((block_n == 64) ? 64 : 128) : block_n;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * l1_out_block_n * num_tma_store_stages * num_mma_elem_bytes;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * l2_out_block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));
    const int smem_barriers = (num_dispatch_barrier_slots + num_epilogue_stages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    const int smem_d2_meta = (is_mma_with_sf(mma_kind) and d2_mode >= 1) ? 8 : 0;
    const int smem_d7_meta = d7_handoff_active ? 96 : 0;
    const int smem_weave_policy = align(
        static_cast<int>(sizeof(weave::PolicyDecision)) + 2 * static_cast<int>(sizeof(uint32_t)),
        16);
    const int smem_tmem_ptr = 4 + smem_d2_meta + smem_d7_meta;

    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;
    const int smem_a_size_per_stage = load_block_m * block_k * num_mma_elem_bytes;
    const int smem_b_size_per_stage = block_n * block_k * num_mma_elem_bytes;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    const int smem_fixed = smem_dispatch_size + smem_d7_dedicated + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_weave_policy + smem_tmem_ptr;

    const int max_num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(max_num_stages >= 2);
    const int d3_stages = get_tune_env("DG_TUNE_D3_STAGES", "DG_D3_STAGES", 0);
    const int num_stages = (d3_stages > 0) ? std::min(d3_stages, max_num_stages) : get_env<int>("DG_MEGA_STAGES", max_num_stages);
    DG_HOST_ASSERT(num_stages >= 2);
    DG_HOST_ASSERT(smem_fixed + num_stages * smem_size_per_stage <= smem_capacity);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {

    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
            num_tokens, mma_kind, hidden, intermediate_hidden);

    int block_n = 128;
    const int d3_bn = get_tune_env("DG_TUNE_D3_BN", "DG_D3_BN", 0);
    if (d3_bn > 0) {
        block_n = d3_bn;
        DG_HOST_ASSERT(block_n == 64 or block_n == 128 or block_n == 256);
    }

    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4) : std::pair(0, 0);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = 32;

    // Thread layout (Axis 2 D2 & Axis 7 D7)
    const int d2_mode = get_tune_env("DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
    const int d7_warps = get_tune_env("DG_TUNE_D7_WARPS", "DG_TUNE_D7_SCATTER_WARPS", get_env<int>("DG_SCATTER_WARPS", 0));
    int num_dispatch_threads = get_env<int>("DG_MEGA_DISPATCH_THREADS", 128);
    if (d7_warps > 0)
        num_dispatch_threads = d7_warps * 32;
    if ((2 <= d2_mode and d2_mode <= 8) or (11 <= d2_mode and d2_mode <= 15))
        num_dispatch_threads = 256;
    DG_HOST_ASSERT(num_dispatch_threads != 192 and
                   (num_dispatch_threads == 128 or num_dispatch_threads == 256));
    const int num_non_epilogue_threads = get_env<int>("DG_MEGA_NON_EPILOGUE_THREADS", 128);

    // Pull bytes (Axis 2 D2 & Axis 3 D3)
    constexpr int kPullThreshold = 4096;
    int num_bytes_per_pull = hidden * get_num_mma_elem_bytes(mma_kind);
    while (num_bytes_per_pull > kPullThreshold) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }
    if ((2 <= d2_mode and d2_mode <= 4) or (11 <= d2_mode and d2_mode <= 13)) {
        num_bytes_per_pull = 2048;
    }
    const int d2_reg_pull_bytes = get_tune_env("DG_TUNE_D2_REG_PULL_BYTES", "DG_ISOLATED_D2_REG_PULL_BYTES", 0);
    if (d2_reg_pull_bytes > 0) {
        num_bytes_per_pull = d2_reg_pull_bytes;
    }
    num_bytes_per_pull = get_env<int>("DG_MEGA_NUM_BYTES_PER_PULL", num_bytes_per_pull);
    DG_HOST_ASSERT(num_bytes_per_pull > 0);
    DG_HOST_ASSERT(hidden * get_num_mma_elem_bytes(mma_kind) % num_bytes_per_pull == 0);

    // Pipeline
    auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
