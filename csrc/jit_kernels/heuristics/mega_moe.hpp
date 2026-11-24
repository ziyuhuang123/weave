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
#include "weave_knob_config.hpp"

namespace deep_gemm {

// Code-generation switches resolved once per launch from the unified WEAVE_*
// knobs (plus the legacy env aliases).  The JIT generator, the SMEM planner and
// the TMA-descriptor builder all read this single copy, so the host SMEM budget
// always matches the kernel's SharedStorage and the declared kernel parameters.
struct MegaMoECodegen {
    // D7 scatter executor: 0 epilogue stores, 1 SMEM handoff to the comm warps,
    // 2 epilogue local bf16x2 dedup, 3 local dedup + HBM scatter queue.
    int d7_scatter_mode = 0;
    int d7_smem_alias = 1;
    int d7_handoff_wait = 1;
    // D3 dynamic epilogue / dynamic TMA-A boxes and the per-expert BM ladder.
    int dyn_epi = 0;
    int dyn_tma_a = 0;
    bool has_extra_tmas = false;
    int expert_step_mode = 0;
    // D6 fine-grained combine: N-tile group size and combine reduce warps.
    int d6_g = 0;
    int d6_rw = 0;
    // Fused SharedLinear2 + INT7 tail combine, derived from the
    // existing public WEAVE_* preset.  Disables D6 and the final combine.
    bool fused_shared_int7 = false;

    bool d7_handoff_active() const { return d7_scatter_mode == 1 and d7_handoff_wait != 0; }
    bool d7_queue_active() const { return d7_scatter_mode == 3; }
};

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m_l1, store_block_m_l2;

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
    uint32_t tmem_pipeline_mode;

    // Single-source codegen switches (see MegaMoECodegen).
    MegaMoECodegen codegen;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m_l1=" << config.store_block_m_l1
           << ", store_block_m_l2=" << config.store_block_m_l2
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull
           << ", tmem_pipeline_mode=" << config.tmem_pipeline_mode
           << ", d7_scatter_mode=" << config.codegen.d7_scatter_mode
           << ", dyn_epi=" << config.codegen.dyn_epi
           << ", dyn_tma_a=" << config.codegen.dyn_tma_a
           << ", has_extra_tmas=" << config.codegen.has_extra_tmas
           << ", expert_step_mode=" << config.codegen.expert_step_mode
           << ", d6_g=" << config.codegen.d6_g << ", d6_rw=" << config.codegen.d6_rw
           << ", dispatch_warps=" << config.knobs.dispatch_warps
           << ", locality_domains=" << config.knobs.locality_domains
           << ", ring_blocks=" << config.knobs.ring_blocks << ")";
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

static bool is_d3_opt_enabled(const MmaKind& mma_kind) {
    if (mma_kind != MmaKind::MXFP8FP4)
        return false;
    return get_weave_env<int>("WEAVE_ENABLE_BLOCK_TUNING", "DG_ISOLATED_D3_OPT", nullptr, 0) != 0 or
           get_weave_env<int>("WEAVE_BLOCK_M", "DG_TUNE_D3_BM", "DG_D3_BM", 0) > 0 or
           get_weave_env<int>("WEAVE_BLOCK_N", "DG_TUNE_D3_BN", "DG_D3_BN", 0) > 0 or
           get_weave_env<int>("WEAVE_BLOCK_K", "DG_TUNE_D3_BK", "DG_D3_BK", 0) > 0 or
           get_weave_env<int>("WEAVE_STORE_BLOCK_M", "DG_TUNE_D3_STORE_M", "DG_D3_STORE_M", 0) > 0 or
           get_weave_env<int>("WEAVE_DYN_EPILOGUE", "DG_TUNE_D3_DYN_EPI", "DG_D3_DYN_EPI", 0) > 0 or
           get_weave_env<int>("WEAVE_DYN_TMA_A", "DG_TUNE_D3_DYN_TMA_A", "DG_D3_DYN_TMA_A", 0) > 0 or
           get_weave_env<int>("WEAVE_OVERLAP_EXPERT_SMEM", "DG_TUNE_D3_OVERLAP_SMEM", "DG_D3_OVERLAP_EXPERT_SMEM", 0) > 0;
}

static std::tuple<int, int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind,
    const int& hidden = 0,
    const int& intermediate_hidden = 0) {
    const bool d3_opt = is_d3_opt_enabled(mma_kind);

    auto [cluster_size, block_m, store_block_m_l1, block_k, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int, int> {
        const float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        // Reference SM103 MegaMoE producer: one M block costs roughly 128 extra rows
        // (each task re-reads its weight tiles). Use the fewest blocks per expert:
        // one block up to 192 rows, then blocks of up to 240 rows. A single 240-row
        // block only pays off with enough local experts (about 14 or more per rank).
        const float num_covered_tokens = num_expected_tokens_per_expert + std::sqrt(num_expected_tokens_per_expert);
        if (not d3_opt) {
            if (num_expected_tokens_per_expert <= 8.5) {
                return {2, 16, 8, 256, 2};
            } else if (num_expected_tokens_per_expert <= 16.5) {
                return {2, 32, 16, 128, 2};
            } else if (num_expected_tokens_per_expert <= 32.5) {
                // With more than 64 experts per rank, a 32-row tile that already
                // covers the expected rows (+1 sigma) in one block follows the
                // upstream rule (Qwen3.8-Max EP4, S_rank=256, 20 tokens/expert:
                // BM=32 593.9 us = upstream 593.9 us vs BM=64 614.5 us).
                if (num_covered_tokens <= 32.0f and num_experts / num_ranks > 64)
                    return {2, 32, 16, 128, 2};
                // Reference producer: 64-row tiles store 16 rows per warpgroup and
                // keep both epilogue warpgroups (512-thread geometry).
                return {2, 64, 16, 128, 2};
            } else if (num_expected_tokens_per_expert <= 64.5) {
                if (num_covered_tokens <= 80.0f and mma_kind == MmaKind::MXFP8FP4)
                    return {2, 80, 8, 128, 2};
                return {2, 96, 16, 128, 2};
            } else if (num_expected_tokens_per_expert <= 96.5) {
                // Verified SM103 auto tier (results/sm103_ep4_prior_verified.json):
                // BM=80 with 8/8 epilogue stores (arm B8, 710.53 us = 1.2405x vs
                // BM=240 and 1.054x vs BM=128) beats BM=96/16/8 (B6, 716.67 us)
                // on the EP4 fine-grained-D6 sweep, so the pure 21-knob default
                // picks it directly; 96 stays the BF16 fallback.
                // An 80-row tile only covers <= 80 rows in one block; between 80
                // and 96 covered rows most experts need two blocks, i.e. two
                // full weight reloads (cost ~ blocks * (128 + BM) rows: 2 * 208
                // vs 1 * 256 at BM=128).  Keep the verified BM=80 choice on the
                // 64-expert/rank EP4 shape it was measured on, and follow the
                // upstream single-block rule (BM=128) when each rank holds more
                // experts (Qwen3.8-Max EP4: 128 experts/rank, 80 tokens/expert:
                // BM=128 727.0 us vs BM=80 837.7 us, measured on 4x SM103 EP4).
                if (mma_kind == MmaKind::MXFP8FP4 and
                    (num_covered_tokens <= 80.0f or
                     (num_covered_tokens <= 96.0f and num_experts / num_ranks <= 64)))
                    return {2, 80, 8, 128, 2};
                return {2, 128, 32, 128, 2};
            } else {
                int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
                if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
                    num_blocks = 2;
                int chosen = 240;
                for (const int& candidate: {64, 80, 96, 128, 192, 240}) {
                    chosen = candidate;
                    if (num_blocks * candidate >= num_covered_tokens)
                        break;
                }
                const int chosen_store_m = chosen <= 64 ? 16 :
                                           (chosen == 80 and mma_kind == MmaKind::MXFP8FP4) ? 8 :
                                           (chosen == 96 and mma_kind == MmaKind::MXFP8FP4) ? 16 :
                                           chosen <= 192 ? 32 :
                                           mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;
                return {2, chosen, chosen_store_m, 128, 2};
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

    block_m = get_weave_env<int>("WEAVE_BLOCK_M", "DG_MEGA_BLOCK_M", nullptr, block_m);
    store_block_m_l1 = get_weave_env<int>("WEAVE_STORE_BLOCK_M", "DG_MEGA_STORE_BLOCK_M", nullptr, store_block_m_l1);
    block_k = get_weave_env<int>("WEAVE_BLOCK_K", "DG_MEGA_BLOCK_K", nullptr, block_k);
    num_epilogue_warpgroups = get_weave_env<int>("WEAVE_EPILOGUE_WARPGROUPS", "DG_MEGA_EPILOGUE_WARPGROUPS", nullptr, num_epilogue_warpgroups);

    // Asymmetric L1/L2 store split for FP8xFP4:
    //   - BM=80 (8/8) & BM=96 (16/8) unlock 8 TMA stages on SM103;
    //   - BM=240 (24/8) unlocks 6 TMA stages inside 232,448B SMEM.
    int store_block_m_l2 = ((block_m == 80 or block_m == 96 or block_m == 240) and
                            mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1;

    const int d3_bm = get_weave_env<int>("WEAVE_BLOCK_M", "DG_TUNE_D3_BM", "DG_D3_BM", 0);
    const int d3_store_m = get_weave_env<int>("WEAVE_STORE_BLOCK_M", "DG_TUNE_D3_STORE_M", "DG_D3_STORE_M", 0);
    const int d3_store_m_l2 = get_weave_env<int>("WEAVE_STORE_BLOCK_M_L2", "DG_TUNE_D3_STORE_M_L2", "DG_D3_STORE_M_L2", 0);
    const int d3_bk = get_weave_env<int>("WEAVE_BLOCK_K", "DG_TUNE_D3_BK", "DG_D3_BK", 0);
    const int d3_epi_wg = get_weave_env<int>("WEAVE_EPILOGUE_WARPGROUPS", "DG_TUNE_D3_EPI_WG", "DG_D3_EPI_WG", 0);
    if (d3_bm > 0) block_m = d3_bm;
    if (d3_epi_wg > 0) num_epilogue_warpgroups = d3_epi_wg;
    if (d3_store_m > 0) {
        store_block_m_l1 = d3_store_m;
    } else {
        if (mma_kind == MmaKind::MXFP8FP4) {
            if (block_m == 80) store_block_m_l1 = 8;
            else if (block_m == 96) store_block_m_l1 = 16;
            else if (block_m == 240) store_block_m_l1 = 24;
            else if (block_m == 256) store_block_m_l1 = 32;
        }
        if (store_block_m_l1 * num_epilogue_warpgroups > block_m)
            store_block_m_l1 = std::max(8, block_m / num_epilogue_warpgroups);
    }
    if (d3_store_m_l2 > 0) {
        store_block_m_l2 = d3_store_m_l2;
    } else if (d3_store_m > 0 or (block_m != 80 and block_m != 96 and block_m != 240)) {
        store_block_m_l2 = store_block_m_l1;
    }
    if (store_block_m_l2 * num_epilogue_warpgroups > block_m)
        store_block_m_l2 = std::max(8, block_m / num_epilogue_warpgroups);
    if (d3_bk > 0) block_k = d3_bk;

    // Axis 2 (D2): Modes 1..8 use 2 epilogue warpgroups (8E = 256 threads); Modes 11..15 keep default
    const int d2_mode = get_weave_env<int>("WEAVE_WARP_SPLIT_MODE", "DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
    if (1 <= d2_mode and d2_mode <= 8) {
        num_epilogue_warpgroups = 2;
        if (store_block_m_l1 * num_epilogue_warpgroups > block_m)
            store_block_m_l1 = std::max(8, block_m / num_epilogue_warpgroups);
        if (store_block_m_l2 * num_epilogue_warpgroups > block_m)
            store_block_m_l2 = std::max(8, block_m / num_epilogue_warpgroups);
    }

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    return {cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_warpgroups * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m_l1, const int& store_block_m_l2,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind,
    const weave::WeaveKnobs& knobs,
    const MegaMoECodegen& codegen = MegaMoECodegen{}) {
    constexpr int kSmemAlignment = 1024;
    const int num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);

    // Mirrors the JIT codegen default (WEAVE_ACCUM_STAGES -> 2): the kernel sizes
    // tmem_full/empty_barriers from that emitted value (clamped by TMEM capacity,
    // which still keeps 2 stages for the overlapped BM=256 layout), so planning 1
    // stage at BM=256 (e.g. the WEAVE_ADAPT_BM pow2 envelope at S_rank=2048)
    // under-allocated the SharedStorage by 2 barriers = 16 B.
    const int default_accum_stages = 2;
    const int num_epilogue_stages = get_weave_env<int>("WEAVE_ACCUM_STAGES", "DG_D3_ACCUM_STAGES", "DG_MEGA_ACCUM_STAGES", default_accum_stages);
    DG_HOST_ASSERT(1 <= num_epilogue_stages and num_epilogue_stages <= 4);
    const int num_tma_store_stages = get_weave_env<int>("WEAVE_TMA_STORE_STAGES", "DG_D3_TMA_STORE_STAGES", "DG_MEGA_TMA_STORE_STAGES", 2);
    DG_HOST_ASSERT(1 <= num_tma_store_stages and num_tma_store_stages <= 8);
    const int kNumScheduleStages = get_weave_env<int>("WEAVE_SCHED_STAGES", "DG_TUNE_D3_SCHED_STAGES", "DG_D3_SCHED_STAGES", 2);

    const int d2_mode = get_weave_env<int>("WEAVE_WARP_SPLIT_MODE", "DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
    const int d2_reg_pull_bytes = get_weave_env<int>("WEAVE_REG_PULL_BYTES", "DG_TUNE_D2_REG_PULL_BYTES", "DG_ISOLATED_D2_REG_PULL_BYTES", 0);
    const bool d2_all_reg_pull = is_mma_with_sf(mma_kind) and (d2_reg_pull_bytes > 0);
    const bool d2_split_hybrid = is_mma_with_sf(mma_kind) and (not d2_all_reg_pull) and
        ((5 <= d2_mode and d2_mode <= 8) or d2_mode == 14);
    const int num_tma_dispatch_warps =
        d2_all_reg_pull ? 0 : (d2_split_hybrid ? std::min(4, num_dispatch_warps) : num_dispatch_warps);
    const int num_dispatch_barrier_slots =
        d2_all_reg_pull ? 1 : num_tma_dispatch_warps;

    const int d3_overlap_smem = get_weave_env<int>("WEAVE_OVERLAP_EXPERT_SMEM", "DG_TUNE_D3_OVERLAP_SMEM", "DG_D3_OVERLAP_EXPERT_SMEM", 0);
    // D7 terms follow the resolved executor (owner/staging/local-dedup knobs or an
    // explicit WEAVE_SCATTER_MODE), exactly as the kernel's SharedStorage does.
    const int d7_smem_alias = codegen.d7_smem_alias;
    const bool d7_handoff_active = is_mma_with_sf(mma_kind) and codegen.d7_handoff_active();
    const bool d7_queue_active = is_mma_with_sf(mma_kind) and codegen.d7_queue_active();

    const int load_block_m = block_m / 2;
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int num_smem_pull_bytes = codegen.fused_shared_int7
        ? std::max(num_bytes_per_pull, 5120) : num_bytes_per_pull;
    const int smem_send_buffers_size = (num_tma_dispatch_warps > 0 and num_bytes_per_pull > 0) ? align(
        static_cast<int>(layout::Buffer(layout::Data(num_smem_pull_bytes), num_tma_dispatch_warps, 1).get_num_bytes()),
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
    // Mirrors the kernel's DG_D7_RING_SLOTS: 4 slots for strict comm_only with
    // the register-direct pull, otherwise the 2-slot ping-pong.
    const int d7_ring_slots = (codegen.d7_handoff_wait == 3 and d2_all_reg_pull) ? 4 : 2;
    const int smem_d7_dedicated = (d7_handoff_active and d7_smem_alias == 0)
        ? (d7_ring_slots * num_epilogue_warpgroups * ((codegen.d7_handoff_wait == 3 and not d2_all_reg_pull) ? 8 : 16) * 256)
        : 0;

    const int l1_out_block_n = is_mma_with_sf(mma_kind) ? 64 : (block_n / 2);
    const int l2_out_block_n = is_mma_with_sf(mma_kind) ? ((block_n == 64) ? 64 : 128) : block_n;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m_l1 * l1_out_block_n * num_tma_store_stages * num_mma_elem_bytes;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * l2_out_block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));
    const int smem_barriers = (num_dispatch_barrier_slots + num_epilogue_stages * 2 + 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m_l1 * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    const int smem_d2_meta = (is_mma_with_sf(mma_kind) and d2_mode >= 1) ? 8 : 0;
    // Handoff: scatter_jobs[2] + 4 barriers + epi_scatter_mode[4] + dispatch_pull_done.
    // Queue: dispatch_pull_done only (padded to 16 B).
    const int smem_d7_meta = d7_handoff_active ? (32 * d7_ring_slots + 32) : (d7_queue_active ? 16 : 0);
    // The kernel compiles its CTA-0 PolicyDecision block iff WEAVE_RUNTIME_POLICY,
    // which the JIT generator emits from the resolved knobs.sm0_stream_sched
    // (explicit WEAVE_SM0_STREAM_SCHED / DG_SM0_STREAM_SCHED, or implied by
    // WEAVE_READY_QUEUE=sm0_fifo); plan from that same resolved value.
    const int smem_weave_policy = knobs.sm0_stream_sched != 0u
        ? align(static_cast<int>(sizeof(weave::PolicyDecision)) + 5 * static_cast<int>(sizeof(uint32_t)), 16)
        : 0;
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
    const int d3_stages = get_weave_env<int>("WEAVE_GEMM_STAGES", "DG_TUNE_D3_STAGES", "DG_D3_STAGES", 0);
    // WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG STAGES field (launch-wide, both
    // tuples must agree); clamped to the largest legal count like WEAVE_GEMM_STAGES.
    const int tuple_stages1 = weave_knobs::gemm1_config_env().stages;
    const int tuple_stages2 = weave_knobs::gemm2_config_env().stages;
    DG_HOST_ASSERT(tuple_stages1 == 0 or tuple_stages2 == 0 or tuple_stages1 == tuple_stages2);
    const int tuple_stages = tuple_stages1 != 0 ? tuple_stages1 : tuple_stages2;
    const int num_stages = (d3_stages > 0) ? std::min(d3_stages, max_num_stages) :
        (tuple_stages > 0) ? std::min(tuple_stages, max_num_stages) :
        get_weave_env<int>("WEAVE_GEMM_STAGES", "DG_MEGA_STAGES", nullptr, max_num_stages);
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
    const MmaKind& mma_kind,
    const int& num_shared_experts = 0) {

    auto [cluster_size, block_m_auto, store_block_m_l1_auto, store_block_m_l2_auto,
          block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
            num_tokens, mma_kind, hidden, intermediate_hidden);
    int block_m = block_m_auto;
    int store_block_m_l1 = store_block_m_l1_auto;
    int store_block_m_l2 = store_block_m_l2_auto;

    const auto arch = weave::HardwareArch::B300_SM100;
    const auto precision = mma_kind == MmaKind::BF16
        ? weave::PrecisionMode::BF16 : weave::PrecisionMode::FP8xFP4;
    auto knobs = weave_knobs::from_env(arch, precision);

    // ---- WEAVE_ADAPT_GEMM_CONFIG / WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG ----
    // The whole GEMM config tuple (BM, BN, BK, store_BM_L1, store_BM_L2, smem
    // stages, TMEM pipeline mode, epilogue register stages, setmaxnreg split) is
    // a HOST decision taken once per launch from
    //   (M_expected = num_tokens * num_ranks * num_topk / num_experts, H, I).
    //   * ADAPT_GEMM_CONFIG = 1 (default when unset; legacy alias WEAVE_ADAPT_BM):
    //     the matched tuple from get_block_config_for_mega_moe (BM ladder
    //     {16, 32, 64, 80, 96, 128, 192, 240}, BK 128/256, asymmetric L1/L2
    //     store heights, auto max legal stages, epi reg stages 2 for 80/96/240)
    //     plus the 64/56/192 setmaxnreg rebalance when WEAVE_DEDUP_MODE=1.
    //     It NEVER chops a pool block into device-side sub-tiles: the old
    //     ADAPT_BM per-expert 64/32-row ladder (expert_step_mode / dyn_epi /
    //     dyn_tma_a) re-loaded every expert's weights once per sub-tile and
    //     measured +174..+719 us; it is only reachable through the internal
    //     WEAVE_EXPERT_STEP_MODE diagnostic now.
    //   * ADAPT_GEMM_CONFIG = 0 (and no GEMM1/GEMM2 tuple pinned): one fixed
    //     canonical tuple for every shape: BM=128, BN=128, BK=128, store 32/32,
    //     TMEM mode 200, epi reg stages 1, max legal stages, default setmaxnreg.
    //   * WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG pin the per-phase tuple
    //     "BM[:STORE_BM[:BK[:STAGES[:EPI_REG_STAGES]]]]" (legacy aliases
    //     WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2 = BM only).  Equal BMs with
    //     different store heights (e.g. 96:16 / 96:8) are a clean asymmetric
    //     single-envelope build; different BMs keep the legacy decoupled
    //     max(L1, L2) envelope.
    const int adapt_gemm_config = weave_knobs::adapt_gemm_config_env();
    const auto gemm1_cfg = weave_knobs::gemm1_config_env();
    const auto gemm2_cfg = weave_knobs::gemm2_config_env();
    const bool fixed_gemm_tuple = adapt_gemm_config == 0 and not gemm1_cfg.set and not gemm2_cfg.set;
    const auto tuple_field = [](const int a, const int b, const char* what) {
        if (a != 0 and b != 0 and a != b) {
            printf("WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG disagree on the launch-wide %s (%d vs %d)\n", what, a, b);
            DG_HOST_ASSERT(false and "GEMM1/GEMM2 tuples disagree on a launch-wide field");
        }
        return a != 0 ? a : b;
    };
    const int tuple_bk = tuple_field(gemm1_cfg.bk, gemm2_cfg.bk, "BK");
    const int tuple_epi = tuple_field(gemm1_cfg.epi_reg_stages, gemm2_cfg.epi_reg_stages, "EPI_REG_STAGES");
    const bool epi_reg_env_set = std::getenv("WEAVE_EPI_REG_STAGES") != nullptr;
    const bool reg_balance_env_set = std::getenv("WEAVE_REG_BALANCE") != nullptr;
    if (fixed_gemm_tuple) {
        block_m = 128;
        store_block_m_l1 = store_block_m_l2 = 32;
        block_k = 128 / get_num_mma_elem_bytes(mma_kind);
        if (not epi_reg_env_set)
            knobs.epi_reg_stages = 1u;
        if (knobs.tmem_pipeline_mode == 0u)
            knobs.tmem_pipeline_mode = 200u;
    } else {
        if (tuple_bk > 0)
            block_k = tuple_bk / get_num_mma_elem_bytes(mma_kind);
        if (tuple_epi > 0 and not epi_reg_env_set)
            knobs.epi_reg_stages = static_cast<uint32_t>(tuple_epi);
        // Adaptive setmaxnreg fold: the dedup pull path (WEAVE_DEDUP_MODE=1)
        // keeps few live registers in the comm / TMA warps, so the 64/56/192
        // split (measured -37.1 us at Qwen3.5-35B S=16384) is the matched
        // choice.  An explicit WEAVE_REG_BALANCE still pins it.
        if (not reg_balance_env_set and adapt_gemm_config != 0 and mma_kind == MmaKind::MXFP8FP4 and
            knobs.dedup_mode == 1u)
            knobs.reg_balance = 1u;
    }
    // Per-GEMM envelopes: when L1 and L2 request distinct block_m profiles,
    // compile the CTA envelope at max(L1, L2) and bind each phase's synchronous
    // store height (store_block_m_l1 vs store_block_m_l2) and dynamic TMA/epi
    // sub-tile heights independently from its own profile.
    const int req_bm_l1 = knobs.block_m_l1 != 0u ? static_cast<int>(knobs.block_m_l1) : 0;
    const int req_bm_l2 = knobs.block_m_l2 != 0u ? static_cast<int>(knobs.block_m_l2) : 0;
    int block_m_resolved = (req_bm_l1 > 0 and req_bm_l2 > 0)
        ? std::max(req_bm_l1, req_bm_l2)
        : static_cast<int>(weave::resolve_block_m(
              knobs, knobs.block_m_l1, knobs.block_m_l2, static_cast<uint32_t>(block_m)));
    // A fixed 32-row sub-tile inside a larger envelope puts every second tile at
    // an odd 32-row offset, i.e. an odd block-scale TMEM column, which faults
    // with `misaligned address` on SM103 (measured for L1 and L2).  Reject it on
    // the host (the internal per-expert ladder only uses 32-row tiles as single tiles).
    if ((req_bm_l1 == 32 or req_bm_l2 == 32) and block_m_resolved > 32) {
        printf("WEAVE_BLOCK_M_L1/L2=32 inside a %d-row envelope is not supported (odd SF TMEM column)\n",
               block_m_resolved);
        DG_HOST_ASSERT(false and "32-row sub-tiles need a 32-row BLOCK_M envelope");
    }
    block_m = block_m_resolved;
    const int eff_bm_l1 = req_bm_l1 > 0 ? req_bm_l1 : block_m;
    const int eff_bm_l2 = req_bm_l2 > 0 ? req_bm_l2 : block_m;
    if (block_m != block_m_auto or req_bm_l1 != req_bm_l2) {
        const int num_epi_wgs = std::max(1, num_epilogue_threads / 128);
        const bool fp8fp4 = mma_kind == MmaKind::MXFP8FP4;
        auto derive_l1_store = [&](const int bm) {
            return bm <= 16 ? 8 : bm <= 64 ? 16 :
                   (bm == 80 and fp8fp4) ? 8 :
                   (bm == 96 and fp8fp4) ? 16 :
                   bm <= 192 ? 32 :
                   bm == 256 ? 32 : fp8fp4 ? 24 : 40;
        };
        store_block_m_l1 = derive_l1_store(eff_bm_l1);
        store_block_m_l2 = ((eff_bm_l2 == 80 or eff_bm_l2 == 96 or eff_bm_l2 == 240) and fp8fp4)
            ? 8 : derive_l1_store(eff_bm_l2);
        auto fix_divisibility = [&](int s_bm) {
            const int wg_bm = block_m / num_epi_wgs;
            while (s_bm > 8 and (s_bm > wg_bm or wg_bm % s_bm != 0))
                s_bm -= 8;
            return std::max(8, s_bm);
        };
        store_block_m_l1 = fix_divisibility(store_block_m_l1);
        store_block_m_l2 = fix_divisibility(store_block_m_l2);
    }
    if (knobs.store_block_m_l1 != 0u)
        store_block_m_l1 = static_cast<int>(knobs.store_block_m_l1);
    if (knobs.store_block_m_l2 != 0u)
        store_block_m_l2 = static_cast<int>(knobs.store_block_m_l2);

    // ---- Single-source codegen resolution --------------------------------------
    // Everything below is consumed verbatim by the JIT generator (macros and the
    // kernel signature), the SMEM planner and the TMA-descriptor builder.
    MegaMoECodegen codegen;
    const bool decoupled_l1_l2_bm = req_bm_l1 > 0 and req_bm_l2 > 0 and req_bm_l1 != req_bm_l2;
    if (mma_kind == MmaKind::MXFP8FP4) {
        // D7 executor.  Local dedup is epilogue-owned (mode 2) when the epilogue
        // explicitly owns the scatter, otherwise comm-owned with the HBM queue (mode 3).
        const int derived_scatter_mode =
            knobs.scatter_owner == 0u ? 0 : (knobs.scatter_staging == 2u ? 3 : 1);
        codegen.d7_scatter_mode = get_weave_env<int>(
            "WEAVE_SCATTER_MODE", "DG_ISOLATED_D7_SCATTER", "DG_TUNE_D7_SCATTER_MODE", derived_scatter_mode);
        if (knobs.local_scatter_dedup != 0u)
            codegen.d7_scatter_mode = knobs.scatter_owner == 0u ? 2 : 3;
        DG_HOST_ASSERT(codegen.d7_scatter_mode >= 0 and codegen.d7_scatter_mode <= 3);
        // comm_only + smem defaults to the strict handoff (wait = 3): the
        // epilogue blocks on a free slot and never issues a remote store, the
        // dispatch warps drain the ring between pulls, so the ring needs its
        // own SMEM (alias = 0).  wait = 1 / 2 are the opportunistic hybrids
        // (epilogue scatters while the pull runs), kept as diagnostics only.
        codegen.d7_handoff_wait = get_weave_env<int>("WEAVE_SCATTER_HANDOFF_WAIT", "DG_TUNE_D7_HANDOFF_WAIT", nullptr,
                                                     codegen.d7_scatter_mode == 1 ? 3 : 1);
        codegen.d7_smem_alias = get_weave_env<int>("WEAVE_SCATTER_SMEM_ALIAS", "DG_TUNE_D7_SMEM_ALIAS", nullptr,
                                                   (codegen.d7_scatter_mode == 1 and codegen.d7_handoff_wait == 3) ? 0 : 1);
        if (codegen.d7_scatter_mode == 1) {
            DG_HOST_ASSERT(codegen.d7_handoff_wait >= 1 and codegen.d7_handoff_wait <= 3);
            DG_HOST_ASSERT(codegen.d7_handoff_wait != 3 or codegen.d7_smem_alias == 0);
        }
        // The D5 dedup pull (WEAVE_DEDUP_MODE=1, i.e. d5_dedup == 2) rendezvouses all
        // dispatch warps grid-wide mid-pull, which the strict handoff's between-pull
        // drain cannot cover, so it falls back to wait = 2.  Resolve the fallback HERE
        // (not only in the JIT generator) so the SMEM planner below sizes the same ring
        // (16-row slots for wait 2) as the emitted kernel; otherwise the planner sized the
        // 8-row strict ring and the kernel static_assert "Host SMEM planner under-allocates"
        // failed for comm_only + dedup=1 + register pull 0.
        if (codegen.d7_scatter_mode == 1 and codegen.d7_handoff_wait == 3) {
            const int public_dedup = get_env<int>("WEAVE_DEDUP_MODE", -1);
            const int d5_dedup = (public_dedup >= 0)
                ? (public_dedup == 1 ? 2 : 0)
                : get_weave_env<int>("DG_TUNE_D5_DEDUP", "DG_ISOLATED_D5_DEDUP", nullptr, 0);
            if (d5_dedup == 2) {
                printf("WEAVE_SCATTER_OWNER=comm_only: WEAVE_DEDUP_MODE=1 falls back to handoff wait 2\n");
                codegen.d7_handoff_wait = 2;
            }
        }

        // D3 dynamic epilogue: decoupled L1/L2 tiles need 8-row granular stores.
        codegen.dyn_epi = get_weave_env<int>(
            "WEAVE_DYN_EPILOGUE", "DG_TUNE_D3_DYN_EPI", "DG_D3_DYN_EPI", decoupled_l1_l2_bm ? 2 : 0);
        DG_HOST_ASSERT(codegen.dyn_epi >= 0 and codegen.dyn_epi <= 2);
        codegen.dyn_tma_a = get_weave_env<int>(
            "WEAVE_DYN_TMA_A", "DG_TUNE_D3_DYN_TMA_A", "DG_D3_DYN_TMA_A",
            (decoupled_l1_l2_bm and codegen.dyn_epi == 2) ? 2 : 0);
        DG_HOST_ASSERT(codegen.dyn_tma_a >= 0 and codegen.dyn_tma_a <= 2);
        codegen.has_extra_tmas = (codegen.dyn_epi == 2) or (codegen.dyn_tma_a > 0);
        codegen.expert_step_mode = get_weave_env<int>(
            "WEAVE_EXPERT_STEP_MODE", "DG_TUNE_D3_EXPERT_STEP_MODE", "DG_D3_EXPERT_STEP_MODE", 0);

        // D6 fine-grained combine (COMBINE_SYNC_CHUNK -> N-tile group size).
        // 896 * 16 B = 14336 B = one whole 7168-column combine row: a single
        // readiness flag (and a single sys fence) per pool M-block.  The device
        // clamps the group to L2_SHAPE_N / BLOCK_N, so 896 means "full row" at
        // any hidden size.
        const int chunk_to_d6_g = (knobs.combine_sync_chunk >= 896u) ? 56 :
                                  ((knobs.combine_sync_chunk >= 128u) ? 8 :
                                  ((knobs.combine_sync_chunk >= 64u) ? 4 :
                                  ((knobs.combine_sync_chunk >= 32u) ? 2 :
                                  ((knobs.combine_sync_chunk >= 16u) ? 1 : 0))));
        const int auto_d6_g = (chunk_to_d6_g > 0) ? chunk_to_d6_g :
                              ((knobs.early_stream_reduce != 0u or knobs.reduce_warps > 0u) ? 2 : 0);
        codegen.d6_g = get_weave_env<int>("WEAVE_COMBINE_GROUP_TILES", "DG_TUNE_D6_G", "DG_ISOLATED_D6_G", auto_d6_g);
        // Without dedicated Middle reduce warps (WEAVE_REDUCE_WARPS=0) the tail
        // Combine/Reduce is always the all-warp (W00..W15) combine: with
        // COMBINE_SYNC_CHUNK > 0 it waits on the D6 group flags, with 0 on the
        // per-peer grid-index readiness tags.  WEAVE_COMBINE_REDUCE_WARPS=0 is
        // the internal ablation switch back to the 8-warp epilogue-only combine.
        const int default_d6_rw = (knobs.reduce_warps == 0u) ? 16 : 0;
        codegen.d6_rw = get_weave_env<int>(
            "WEAVE_COMBINE_REDUCE_WARPS", "DG_TUNE_D6_REDUCE_WARPS", "DG_ISOLATED_D6_REDUCE_WARPS", default_d6_rw);
    }

    // Fused SharedLinear2 + INT7 request.  This is a pure function of the
    // existing 21-knob preset; the block/thread shape is checked below once the
    // resolved layout is known (a non-matching shape falls back losslessly).
    const bool fused_shared_int7_requested =
        knobs.combine_sync_chunk == 64u and knobs.reduce_warps == 0u and
        knobs.scatter_owner == 0u and knobs.scatter_staging == 0u and
        knobs.local_scatter_dedup == 0u and knobs.early_stream_reduce == 0u and
        knobs.epi_reg_stages == 1u and
        knobs.block_m_l1 == 0u and knobs.block_m_l2 == 0u and
        knobs.store_block_m_l1 == 0u and knobs.store_block_m_l2 == 0u and
        codegen.dyn_epi == 0 and codegen.dyn_tma_a == 0 and
        codegen.d7_scatter_mode == 0 and codegen.expert_step_mode == 0;

    // Per-phase / per-expert sub-tiling envelope (see sched::TaskInfo::get_m_row_offset).
    if (decoupled_l1_l2_bm) {
        const int lo = std::min(req_bm_l1, req_bm_l2), hi = std::max(req_bm_l1, req_bm_l2);
        if (mma_kind != MmaKind::MXFP8FP4 or hi % lo != 0 or lo % 32 != 0) {
            printf("WEAVE_BLOCK_M_L1=%d / WEAVE_BLOCK_M_L2=%d: decoupled tiles need FP8xFP4, a 32-row "
                   "multiple and the smaller tile dividing the larger one\n", req_bm_l1, req_bm_l2);
            DG_HOST_ASSERT(false and "unsupported decoupled WEAVE_BLOCK_M_L1/L2 pair");
        }
    }
    const bool sub_tile_l2 = codegen.expert_step_mode > 0 or (decoupled_l1_l2_bm and req_bm_l2 < block_m);
    if (codegen.expert_step_mode > 0 or decoupled_l1_l2_bm) {
        if (block_m % 32 != 0) {
            printf("WEAVE_ADAPT_BM / decoupled BLOCK_M_L1/L2 need a 32-row aligned BLOCK_M envelope (got %d)\n", block_m);
            DG_HOST_ASSERT(false and "unsupported sub-tiled BLOCK_M envelope");
        }
        // Sub-tiling is implemented on the default scheduler path only.
        DG_HOST_ASSERT(get_weave_env<int>("WEAVE_READY_GROUP_SIZE", "DG_TUNE_D1_G", "DG_ISOLATED_D1_G", 0) == 0);
        DG_HOST_ASSERT(get_env<int>("WEAVE_DEDUP_MODE", 0) == 0);
        DG_HOST_ASSERT(weave_knobs::named_env("WEAVE_READY_QUEUE", {"formula", "sm0_fifo"},
                       get_weave_env<int>("DG_TUNE_D4_READYQ", "DG_ISOLATED_D4_READYQ", nullptr, 0)) == 0);
        DG_HOST_ASSERT(get_weave_env<int>("WEAVE_EXPERTS_PER_WAVE", "DG_TUNE_D4_EPW", "DG_ISOLATED_D4_EPW", 0) == 0);
        DG_HOST_ASSERT(get_weave_env<int>("WEAVE_STATIC_EXPERT_ORDER", "DG_TUNE_D4_STATIC", "DG_ISOLATED_D4_STATIC", 0) == 0);
    }
    if (sub_tile_l2 and codegen.d6_g > 1) {
        printf("WEAVE_COMBINE_SYNC_CHUNK > 16 counts N tiles per pool block and cannot be combined with "
               "sub-tiled GEMM2 tasks (WEAVE_ADAPT_BM=1 or WEAVE_BLOCK_M_L2 < WEAVE_BLOCK_M_L1)\n");
        DG_HOST_ASSERT(false and "unsupported COMBINE_SYNC_CHUNK with sub-tiled GEMM2");
    }

    // Effective store heights, exactly as the kernel derives STORE_BLOCK_M and
    // STORE_BLOCK_M_L2: dyn-epi 2 forces 16-row stores, any dyn-epi shares the L1
    // height with L2, and a static store never exceeds the rows of one warpgroup.
    {
        const int wg_block_m = block_m / std::max(1, num_epilogue_threads / 128);
        if (codegen.dyn_epi == 2) {
            store_block_m_l1 = 16;
        } else {
            store_block_m_l1 = std::min(store_block_m_l1, wg_block_m);
        }
        store_block_m_l2 = codegen.dyn_epi > 0 ? store_block_m_l1 : std::min(store_block_m_l2, wg_block_m);
    }

    int block_n = 128;
    const int d3_bn = get_weave_env<int>("WEAVE_BLOCK_N", "DG_TUNE_D3_BN", "DG_D3_BN", 0);
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
    const int d2_mode = get_weave_env<int>("WEAVE_WARP_SPLIT_MODE", "DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
    const int d7_warps = get_weave_env<int>("WEAVE_SCATTER_WARPS", "DG_TUNE_D7_WARPS", "DG_TUNE_D7_SCATTER_WARPS", get_env<int>("DG_SCATTER_WARPS", 0));
    const uint32_t dispatch_warps = weave::resolve_dispatch_warps(knobs, 4u);
    DG_HOST_ASSERT(dispatch_warps == 4u or dispatch_warps == 8u);
    int num_dispatch_threads = static_cast<int>(dispatch_warps) * 32;
    if (knobs.dispatch_warps == 0u)
        num_dispatch_threads = get_weave_env<int>("WEAVE_DISPATCH_THREADS", "DG_MEGA_DISPATCH_THREADS", nullptr, num_dispatch_threads);
    // WEAVE_SCATTER_WARPS is an alias of the dispatch (comm) warp count; reject a
    // contradicting WEAVE_DISPATCH_WARPS instead of silently overriding it.
    if (d7_warps > 0 and knobs.dispatch_warps != 0u and static_cast<uint32_t>(d7_warps) != knobs.dispatch_warps) {
        printf("WEAVE_SCATTER_WARPS=%d contradicts WEAVE_DISPATCH_WARPS=%u (scatter runs on the dispatch warps)\n",
               d7_warps, knobs.dispatch_warps);
        DG_HOST_ASSERT(false and "WEAVE_SCATTER_WARPS != WEAVE_DISPATCH_WARPS");
    }
    if (d7_warps > 0)
        num_dispatch_threads = d7_warps * 32;
    if ((2 <= d2_mode and d2_mode <= 8) or (11 <= d2_mode and d2_mode <= 15))
        num_dispatch_threads = 256;
    DG_HOST_ASSERT(num_dispatch_threads != 192 and
                   (num_dispatch_threads == 128 or num_dispatch_threads == 256));
    const int num_non_epilogue_threads = get_weave_env<int>("WEAVE_NON_EPILOGUE_THREADS", "DG_MEGA_NON_EPILOGUE_THREADS", nullptr, 128);

    // The fused-shared INT7 path needs the final dispatch thread count, so it is decided only after
    // the D2/D7 thread layout above (it was read before its declaration).
    codegen.fused_shared_int7 = fused_shared_int7_requested and mma_kind == MmaKind::MXFP8FP4 and
        num_shared_experts == 1 and hidden % 256 == 0 and block_n == 128 and
        block_m == 240 and store_block_m_l2 == 8 and num_epilogue_threads == 256 and
        num_dispatch_threads == 128 and num_topk <= 8;
    if (codegen.fused_shared_int7) {
        // The fused tail SharedLinear2 epilogue replaces the D6/final combine.
        codegen.d6_g = 0;
        codegen.d6_rw = 0;
    }

    // Pull bytes (Axis 2 D2 & Axis 3 D3)
    // Reference producer pulls a whole 8 KiB-class token row per pull at FP8xFP4
    // (H=7168 -> 7168 B), which needs the 8192 threshold; the 4096 threshold used
    // to halve it to 3584 B and doubled the number of dispatch pulls.
    const int kPullThreshold = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    int num_bytes_per_pull = hidden * get_num_mma_elem_bytes(mma_kind);
    while (num_bytes_per_pull > kPullThreshold) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }
    if ((2 <= d2_mode and d2_mode <= 4) or (11 <= d2_mode and d2_mode <= 13)) {
        num_bytes_per_pull = 2048;
    }
    const int d2_reg_pull_bytes = get_weave_env<int>("WEAVE_REG_PULL_BYTES", "DG_TUNE_D2_REG_PULL_BYTES", "DG_ISOLATED_D2_REG_PULL_BYTES", 0);
    if (d2_reg_pull_bytes > 0) {
        num_bytes_per_pull = d2_reg_pull_bytes;
    }
    num_bytes_per_pull = get_weave_env<int>("WEAVE_NUM_BYTES_PER_PULL", "DG_MEGA_NUM_BYTES_PER_PULL", nullptr, num_bytes_per_pull);
    DG_HOST_ASSERT(num_bytes_per_pull > 0);
    DG_HOST_ASSERT(hidden * get_num_mma_elem_bytes(mma_kind) % num_bytes_per_pull == 0);

    // Pipeline
    auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind, knobs, codegen);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull,
        knobs,
        weave_knobs::resolve_tmem_pipeline_mode(knobs, static_cast<uint32_t>(block_m)),
        codegen
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
