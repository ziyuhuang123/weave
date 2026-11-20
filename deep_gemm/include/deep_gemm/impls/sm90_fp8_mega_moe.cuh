#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/nv_moe_workspace.cuh>
#include <deep_gemm/impls/sm90_mega_moe_math.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/sm90_mega_moe.cuh>
#include <deep_gemm/scheduler/weave_policy_runtime.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/impls/sm90_agy_profiler.cuh>
namespace deep_gemm {

// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * One A+SFA producer warp and one B producer warp feed the pipeline.
//   * Math warpgroups consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over the 64 post-SwiGLU columns of this block, FP8 e4m3
//         quantize, STSM into SMEM, TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the math warps run the COMBINE step using the
//     same BF16 top-k reduction scheme as the SM100 path.

enum class MegaMoEPhaseKind {
    Linear1,
    Linear2,
    Fused
};

template <uint32_t kNumExpertsPerRank, uint32_t kNumRanks = 1, bool kHasRankCounts = false>
struct alignas(16) AgySm90ExpertCache {
    uint32_t recv_tokens[kNumExpertsPerRank];
    uint32_t m_blocks[kNumExpertsPerRank];
    uint32_t expert_bm[kNumExpertsPerRank];
    uint32_t pool_block_offset[kNumExpertsPerRank + 1];
    uint32_t l1_tile_prefix[kNumExpertsPerRank + 1];
    uint32_t l2_tile_prefix[kNumExpertsPerRank + 1];
    uint32_t exec_expert[kNumExpertsPerRank];
    uint32_t dyn_tile_ring[8];
    uint32_t dyn_aux_ring[8];

    CUTLASS_DEVICE void set_rank_count(const uint32_t&, const uint32_t&, const uint16_t&) {}
    CUTLASS_DEVICE uint16_t get_rank_count(const uint32_t&, const uint32_t&) const { return 0u; }
};

template <uint32_t kNumExpertsPerRank, uint32_t kNumRanks>
struct alignas(16) AgySm90ExpertCache<kNumExpertsPerRank, kNumRanks, true> {
    uint32_t recv_tokens[kNumExpertsPerRank];
    uint32_t m_blocks[kNumExpertsPerRank];
    uint32_t expert_bm[kNumExpertsPerRank];
    uint32_t pool_block_offset[kNumExpertsPerRank + 1];
    uint32_t l1_tile_prefix[kNumExpertsPerRank + 1];
    uint32_t l2_tile_prefix[kNumExpertsPerRank + 1];
    uint32_t exec_expert[kNumExpertsPerRank];
    uint32_t dyn_tile_ring[8];
    uint32_t dyn_aux_ring[8];
    uint16_t recv_by_rank[kNumExpertsPerRank][kNumRanks];

    CUTLASS_DEVICE void set_rank_count(const uint32_t& e, const uint32_t& r, const uint16_t& v) {
        recv_by_rank[e][r] = v;
    }
    CUTLASS_DEVICE uint16_t get_rank_count(const uint32_t& e, const uint32_t& r) const {
        return recv_by_rank[e][r];
    }
};

struct alignas(16) AgyDedupDeferredEntry {
    uint32_t pool_token_idx;
    uint32_t leader_pool_idx;
    uint32_t src_rank_idx;
    uint32_t src_token_idx;
    uint32_t packed_topk_meta;
    uint32_t weight_bits;
    uint32_t m_block_pool_idx;
    uint32_t pad0;
};

__forceinline__ __device__ void sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto scaled = make_float2(
        __fmul_rn(amax.x, kScale), __fmul_rn(amax.y, kScale));
    const auto exp_x = sm90_moe_math::fast_log2_ceil(scaled.x);
    const auto exp_y = sm90_moe_math::fast_log2_ceil(scaled.y);
    sf.x = sm90_moe_math::fast_pow2(exp_x), sf_inv.x = sm90_moe_math::fast_pow2(-exp_x);
    sf.y = sm90_moe_math::fast_pow2(exp_y), sf_inv.y = sm90_moe_math::fast_pow2(-exp_y);
}

__forceinline__ __device__ void sm90_fp8_mega_moe_scat_bar_arrive(
    cutlass::arch::ClusterTransactionBarrier* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar)))
                 : "memory");
}

__forceinline__ __device__ void sm90_fp8_mega_moe_scat_bar_wait(
    cutlass::arch::ClusterTransactionBarrier* bar, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "AGY_SCAT_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 0x989680;\n\t"
        "@!P1 bra AGY_SCAT_WAIT_%=;\n\t"
        "}\n\t"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(phase)
        : "memory");
    phase ^= 1u;
}

template <MegaMoEPhaseKind kKind,
          bool kNMajorScheduleRequested,
          bool kDirectL2ScatterRequested = false,
          bool kOneWarpCleanupRequested = false,
          bool kL1NMajorScheduleRequested = false>
struct MegaMoEPhasePolicy {
    static constexpr bool is_fused = kKind == MegaMoEPhaseKind::Fused;
    static constexpr bool runs_linear1 = (kKind == MegaMoEPhaseKind::Linear1) or is_fused;
    static constexpr bool runs_linear2 = (kKind == MegaMoEPhaseKind::Linear2) or is_fused;
    static constexpr bool nmajor_schedule = kNMajorScheduleRequested;
    static constexpr bool l1_nmajor_schedule =
        is_fused ? kL1NMajorScheduleRequested
                 : (kKind == MegaMoEPhaseKind::Linear1 and kNMajorScheduleRequested);
    static constexpr bool l2_nmajor_schedule =
        runs_linear2 and kNMajorScheduleRequested;
    static constexpr bool direct_l2_scatter =
        runs_linear2 and kDirectL2ScatterRequested;
    static constexpr bool one_warp_cleanup =
        runs_linear2 and kOneWarpCleanupRequested;

    template <typename Scheduler, typename Func>
    CUTLASS_DEVICE static void for_each_selected_block(Scheduler& scheduler, Func&& func) {
        if constexpr (runs_linear1 and not is_fused) {
            scheduler.template for_each_phase_block<sched::SM90BlockPhase::Linear1>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(true, local_expert_idx, num_k_blocks, m_block_idx, n_block_idx,
                         scheduler.get_current_pool_block_offset() + m_block_idx,
                         scheduler.template get_valid_m<false>());
                });
        } else if constexpr (runs_linear2 and not is_fused) {
            scheduler.template for_each_phase_block<sched::SM90BlockPhase::Linear2>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(false, local_expert_idx, num_k_blocks, m_block_idx, n_block_idx,
                         scheduler.get_current_pool_block_offset() + m_block_idx,
                         scheduler.template get_valid_m<false>());
                });
        }
    }
};

#define DG_SM90_FP8_MOE_TEMPLATE_PARAMS \
    uint32_t kNumMaxTokensPerRank, \
    uint32_t kHidden, uint32_t kIntermediateHidden, \
    uint32_t kNumExperts, uint32_t kNumTopk, \
    uint32_t kNumExpertsPerWave, \
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, \
    uint32_t kNumMaxPoolTokens, \
    uint32_t kNumPaddedSFPoolTokens, \
    uint32_t kSFPoolStrideTokens, \
    uint32_t kNumStages, \
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads, \
    uint32_t kNumEpilogueThreads, \
    uint32_t kNumSMs, uint32_t kNumRanks, \
    float kActivationClamp, \
    bool kFastMath, \
    bool kFP8SwapAB = false, \
    bool kBF16ScaledAccumRequested = false

#define DG_SM90_FP8_MOE_KERNEL_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights, \
    const float* __restrict__ l1_weights_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights, \
    const float* __restrict__ l2_weights_sf \
    AGY_PROF_KERNEL_ARG_DECL

#define DG_SM90_FP8_MOE_CORE_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const layout::SymBuffer<kNumRanks>& sym_buffer, \
    const cute::TmaDescriptor& tensor_map_l1_acts, \
    const cute::TmaDescriptor& tensor_map_l1_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l1_weights, \
    const float* __restrict__ l1_weights_sf, \
    const cute::TmaDescriptor& tensor_map_l1_output, \
    const cute::TmaDescriptor& tensor_map_l2_acts, \
    const cute::TmaDescriptor& tensor_map_l2_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l2_weights, \
    const float* __restrict__ l2_weights_sf \
    AGY_PROF_CORE_ARG_DECL

#define DG_SM90_FP8_MOE_KERNEL_ARGS \
    y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, \
    tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, \
    l1_weights_sf, tensor_map_l1_output, tensor_map_l2_acts, \
    tensor_map_l2_acts_sf, tensor_map_l2_weights, l2_weights_sf \
    AGY_PROF_KERNEL_ARG_PASS

#define DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS \
    kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, \
    kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, kNumMaxPoolTokens, \
    kNumPaddedSFPoolTokens, kSFPoolStrideTokens, kNumStages, kNumDispatchThreads, \
    kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, \
    kActivationClamp, kFastMath, kFP8SwapAB, kBF16ScaledAccumRequested

template <typename MegaMoEPhase, DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
struct SM90MegaMoECoreRunner {
    template <int kRoleMode>
    __device__ __noinline__ static void
    role3_noinline(DG_SM90_FP8_MOE_CORE_ARGS_DECL) {
        run<kRoleMode>(DG_SM90_FP8_MOE_KERNEL_ARGS);
    }

    template <int kRoleMode>
    CUTLASS_DEVICE static void
    run(DG_SM90_FP8_MOE_CORE_ARGS_DECL) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 or kNumDispatchThreads == 192 or kNumDispatchThreads % 128 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128,
                     "Invalid number of GEMM TMA warps (2 or 4 warps expected)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT((BLOCK_M % 64 == 0) or (kFP8SwapAB and (BLOCK_M == 32 or BLOCK_M == 16)),
                     "BLOCK_M must be a multiple of WGMMA::M (64), or 16/32 under swapAB");
    DG_STATIC_ASSERT(BLOCK_N == 64 or BLOCK_N == 128 or BLOCK_N == 256 or
                     BLOCK_N == 512,
                     "BLOCK_N must be 64/128/256/512 for this SM90 path");
    DG_STATIC_ASSERT(BLOCK_K == 128 or BLOCK_K == 256,
                     "BLOCK_K must be 128 or 256");
    DG_STATIC_ASSERT(kHidden % BLOCK_K == 0 and kIntermediateHidden % BLOCK_K == 0,
                     "GEMM K dimensions must be divisible by BLOCK_K");
    DG_STATIC_ASSERT(BLOCK_K % 128 == 0,
                     "BLOCK_K must comprise whole BK128 TMA tiles");
    DG_STATIC_ASSERT(BLOCK_N <= 512,
                     "BLOCK_N exceeds the two-box TMA staging contract");

    // Thread / warp identification
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    if constexpr (kRoleMode <= 1) {
        AGY_PROF_CTA_ENTRY(MegaMoEPhase, sm_idx, thread_idx, num_tokens);
        // Prefetch the TMA descriptors used by this split phase.
        if (warp_idx == 0 and cute::elect_one_sync()) {
            if constexpr (MegaMoEPhase::runs_linear1) {
                cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
                cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
                cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
                cute::prefetch_tma_descriptor(&tensor_map_l1_output);
            }
            if constexpr (MegaMoEPhase::runs_linear2) {
                cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
                cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
                cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
            }
        }
    }

    // L2 activation SFs use per-64 K granularity in the nv_dev workspace.
    const auto workspace = layout::nv_moe::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank,
        kNumTopk, kNumMaxPoolTokens);

    // The policy header is appended after all existing workspace offsets.
    auto* const weave_control = workspace.get_weave_policy_control_ptr();
    const uint32_t weave_epoch_base =
        *reinterpret_cast<volatile uint32_t*>(&weave_control->published_epoch);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // MN-major pool SFs keep a TMA-aligned 128-token outer stride even when
    // one logical token's SF byte count is not 16-byte aligned.
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32, false);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 16, false);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());
    DG_STATIC_ASSERT(kSFPoolStrideTokens <= kNumPaddedSFPoolTokens,
                     "Logical SF pool stride must fit in the allocated SF pool capacity");

    // Combine input area stores BF16 L2 contributions for the final reduction.
    constexpr uint32_t kCombineElementBytes = sizeof(nv_bfloat16);
    constexpr auto combine_token_layout = layout::Data(kHidden * kCombineElementBytes);
    const auto combine_token_buffer = layout::Buffer(
        combine_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // GEMM data types and shape constants
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr auto kWarpgroupLayout = layout::get_sm90_moe_warpgroup_layout(
        BLOCK_M, BLOCK_N, kNumEpilogueWarpgroups);
    constexpr bool kSplitNWarpgroups = kWarpgroupLayout.split_n;
    constexpr uint32_t kWarpgroupSplitM = kWarpgroupLayout.split_m;
    constexpr uint32_t kWarpgroupSplitN = kWarpgroupLayout.split_n_count;
    constexpr uint32_t WG_BLOCK_M = kWarpgroupLayout.block_m;
    constexpr uint32_t WG_BLOCK_N = kWarpgroupLayout.block_n;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr bool kSwapABActive =
        kFP8SwapAB and kSplitNWarpgroups and (BLOCK_M == 64 or BLOCK_M == 32 or BLOCK_M == 16) and
        WG_BLOCK_N == 64 and
        (kNumEpilogueWarpgroups == 2 or kNumEpilogueWarpgroups == 4);
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    DG_STATIC_ASSERT(not kSwapABActive or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
#if defined(AGY_INTERLEAVE_L1_L2) && (AGY_INTERLEAVE_L1_L2 > 0)
    constexpr int kAgyInterleaveMode = AGY_INTERLEAVE_L1_L2;
#else
    constexpr int kAgyInterleaveMode = 0;
#endif
    DG_STATIC_ASSERT(kAgyInterleaveMode >= 0 and kAgyInterleaveMode <= 6,
                     "AGY_INTERLEAVE_L1_L2 must be 0..6 (4 = epw L1/L2, 5 = B300 epw + L2KBlockDependency, 6 = wave-lagged epw)");
#if defined(AGY_SCATTER_ROLE_DISP) && AGY_SCATTER_ROLE_DISP
    constexpr bool kAgyScatterRoleDispReq = true;
#else
    constexpr bool kAgyScatterRoleDispReq = false;
#endif
    constexpr bool kAgyScatterRoleDisp = kAgyScatterRoleDispReq;
    constexpr bool kScatDispActive = kAgyScatterRoleDisp and MegaMoEPhase::runs_linear2;
    constexpr bool kDirectL2Scatter =
        (!kSwapABActive) && (!kAgyScatterRoleDisp) && MegaMoEPhase::direct_l2_scatter && WG_BLOCK_N == 128;
    constexpr bool kBF16ScaledAccum = kBF16ScaledAccumRequested;
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT((!kSplitNWarpgroups) or
                     ((BLOCK_M == 64 or (kSwapABActive and (BLOCK_M == 32 or BLOCK_M == 16))) and
                      (WG_BLOCK_N == 64 or WG_BLOCK_N == 128)),
                     "Split-N path expects M64N64/M64N128 or swapAB M16/M32N64 consumers");

    // SM90 MegaMoE uses one CTA per work item; A and B are CTA-local.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = 128;
    constexpr uint32_t kSwizzleBMode   = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)
    constexpr uint32_t kTMATileK       = 128;
    constexpr uint32_t kNumTMATilesPerStage = BLOCK_K / kTMATileK;
    constexpr uint32_t kTMATileN = BLOCK_N > 256 ? 256 : BLOCK_N;
    constexpr uint32_t kNumTMANTilesPerStage = BLOCK_N / kTMATileN;
    DG_STATIC_ASSERT(BLOCK_N % kTMATileN == 0,
                     "BLOCK_N must comprise whole TMA N boxes");

    // Shared memory layout
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Combine reuses the pre-barrier SMEM region, so split L2 keeps this
    // dispatch scratch capacity even though it does not run dispatch pull.
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA holds one aligned BLOCK_M-float vector per 64 channels. L1 uses every
    // other vector (per-128); L2 uses all of them (per-64).
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAKGroups = BLOCK_K / kL2ActsSFGranK;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kNumL2SFAKGroups * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 contribution.
#if defined(AGY_FAST_SWAP_L2) && AGY_FAST_SWAP_L2
    constexpr bool kAgyFastSwapL2 = true;
#else
    constexpr bool kAgyFastSwapL2 = false;
#endif
#if defined(AGY_L2_OMIT_DISPATCH_SMEM) && AGY_L2_OMIT_DISPATCH_SMEM
    constexpr uint32_t SMEM_DISPATCH_TOTAL_SIZE =
        MegaMoEPhase::runs_linear1
            ? (SMEM_EXPERT_COUNT_SIZE > SMEM_SEND_BUFFER_SIZE ? SMEM_EXPERT_COUNT_SIZE : SMEM_SEND_BUFFER_SIZE)
            : 0u;
#elif defined(AGY_ALIAS_DISPATCH_SMEM) && AGY_ALIAS_DISPATCH_SMEM
    constexpr uint32_t SMEM_DISPATCH_TOTAL_SIZE =
        SMEM_EXPERT_COUNT_SIZE > SMEM_SEND_BUFFER_SIZE ? SMEM_EXPERT_COUNT_SIZE : SMEM_SEND_BUFFER_SIZE;
#else
    constexpr uint32_t SMEM_DISPATCH_TOTAL_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE;
#endif
    constexpr uint32_t kL2SmemStrideN =
        (kAgyFastSwapL2 and kSwapABActive and not kAgyScatterRoleDisp) ? (WG_BLOCK_N + 8u) : WG_BLOCK_N;
    constexpr uint32_t SMEM_CD_L1_SIZE = MegaMoEPhase::runs_linear1 ?
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t kScatL2RawBytes =
        kNumEpilogueWarpgroups * WG_BLOCK_M * kL2SmemStrideN * kCombineElementBytes;
    constexpr uint32_t SMEM_CD_L2_SIZE = (!MegaMoEPhase::runs_linear2 || kDirectL2Scatter) ? 0u :
        (kAgyScatterRoleDisp
            ? (kScatL2RawBytes > SMEM_DISPATCH_TOTAL_SIZE ? (kScatL2RawBytes - SMEM_DISPATCH_TOTAL_SIZE) : 0u)
            : kScatL2RawBytes);
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABActive ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_WITH_SWAP_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE > SMEM_CD_SWAP_L1_SIZE ?
            SMEM_CD_OUTPUT_BASE_SIZE : SMEM_CD_SWAP_L1_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_WITH_SWAP_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_OUTPUT_SIZE;
    DG_STATIC_ASSERT(not (kAgyFastSwapL2 and kSwapABActive) or
                     SMEM_CD_L2_SIZE <= SMEM_CD_SWAP_L1_SIZE,
                     "Padded SwapAB L2 SMEM slice must fit inside SwapAB L1 staging region");

#if defined(AGY_COMPACT_NOTIFY) && AGY_COMPACT_NOTIFY
    constexpr bool kAgyCompactNotify = true;
#else
    constexpr bool kAgyCompactNotify = false;
#endif
#if defined(AGY_HOT_COLD_L1) && (AGY_HOT_COLD_L1 > 0)
    constexpr int kAgyHotColdMode = AGY_HOT_COLD_L1;
#else
    constexpr int kAgyHotColdMode = 0;
#endif
#if defined(AGY_ALL_IDLE_COMB) && AGY_ALL_IDLE_COMB
    constexpr bool kAgyAllIdleCombReq = true;
#else
    constexpr bool kAgyAllIdleCombReq = false;
#endif
#if (defined(AGY_DYN_SCHED) && AGY_DYN_SCHED) || (defined(AGY_INTERLEAVE_L1_L2) && (AGY_INTERLEAVE_L1_L2 > 0)) || (defined(AGY_HOT_COLD_L1) && (AGY_HOT_COLD_L1 > 0))
    constexpr bool kAgyDynSched = true;
#else
    constexpr bool kAgyDynSched = false;
#endif
    constexpr bool kAgyInterleaveActive =
        (kAgyInterleaveMode > 0) and MegaMoEPhase::is_fused and kAgyDynSched;
#if defined(AGY_CACHE_RANK_COUNTS) && AGY_CACHE_RANK_COUNTS
    constexpr bool kAgyCacheRankCountsReq = true;
#else
    constexpr bool kAgyCacheRankCountsReq = false;
#endif
    constexpr bool kAgyCacheRankCountsSmem =
        kAgyCacheRankCountsReq and MegaMoEPhase::runs_linear1;
    constexpr bool kAgyCacheRankCounts =
        kAgyCacheRankCountsSmem and (kNumMaxTokensPerRank <= 65535u);
    constexpr bool kAgyUseExpertCache =
        kAgyCompactNotify or kAgyDynSched or kAgyCacheRankCountsSmem;
#if defined(AGY_ADAPT_BM) && AGY_ADAPT_BM
    constexpr bool kAgyAdaptBmReq = true;
#else
    constexpr bool kAgyAdaptBmReq = false;
#endif
    constexpr bool kAgyAdaptBm =
        kAgyAdaptBmReq and kSwapABActive and (BLOCK_M == 64u) and kSplitNWarpgroups and kAgyUseExpertCache;
#if defined(AGY_FAST_CLEAN) && AGY_FAST_CLEAN
    constexpr bool kAgyFastClean = true;
#else
    constexpr bool kAgyFastClean = false;
#endif
#if defined(AGY_STATIC_STRIDE) && AGY_STATIC_STRIDE
    constexpr bool kAgyStaticStride = true;
#else
    constexpr bool kAgyStaticStride = false;
#endif
#if defined(AGY_MCAST) && (AGY_MCAST == 2)
    constexpr uint32_t kAgyMcast =
        (MegaMoEPhase::is_fused and kAgyDynSched and (kNumSMs % 2u == 0u) and
         ((L1_SHAPE_N / BLOCK_N) % 2u == 0u) and ((L2_SHAPE_N / BLOCK_N) % 2u == 0u)) ? 2u : 1u;
#else
    constexpr uint32_t kAgyMcast = 1u;
#endif

#if defined(AGY_PIPE_WGMMA) && AGY_PIPE_WGMMA
    constexpr bool kAgyPipeWgmma = true;
#else
    constexpr bool kAgyPipeWgmma = false;
#endif
#if defined(AGY_SWAP_PIPE) && AGY_SWAP_PIPE
    constexpr bool kAgySwapPipe = true;
#else
    constexpr bool kAgySwapPipe = false;
#endif
#if defined(AGY_DEDUP_DISP) && AGY_DEDUP_DISP
    constexpr bool kAgyDedupDispReq = true;
#else
    constexpr bool kAgyDedupDispReq = false;
#endif
    constexpr bool kAgyDedupDisp =
        kAgyDedupDispReq and ((3u * (kNumMaxPoolTokens / 8u)) >= kNumRanks * kNumMaxTokensPerRank);
#if (defined(AGY_FAST_PULL) && AGY_FAST_PULL) || (defined(AGY_DEDUP_DISP) && AGY_DEDUP_DISP)
    constexpr bool kAgyFastPull = true;
#else
    constexpr bool kAgyFastPull = false;
#endif
#if defined(AGY_DEDUP_SCAT) && AGY_DEDUP_SCAT
    constexpr bool kAgyDedupScatReq = true;
#else
    constexpr bool kAgyDedupScatReq = false;
#endif
    constexpr bool kDedupScatActive =
        (not kAgyInterleaveActive) and (not kAgyScatterRoleDisp) and
        kAgyDedupScatReq and (not kDirectL2Scatter) and
        (kNumMaxPoolTokens >= 2u * kNumRanks * kNumMaxTokensPerRank) and
        ((L2_SHAPE_N / WG_BLOCK_N) <= kNumExpertsPerRank * 6u);
#if defined(AGY_D_MASK_MODE) && AGY_D_MASK_MODE
    constexpr int kAgyDMaskMode = AGY_D_MASK_MODE;
#else
    constexpr int kAgyDMaskMode = 0;
#endif
    DG_STATIC_ASSERT(kAgyDMaskMode >= 0 and kAgyDMaskMode <= 5,
                     "Invalid AGY_D_MASK_MODE (expected 0..5)");
    constexpr bool kAgyHeteroD8D4 =
        (kAgyDMaskMode > 0) and
        (kNumDispatchWarps == 2u or kNumDispatchWarps == 6u or kNumDispatchWarps == 8u) and
        MegaMoEPhase::runs_linear1;
#if defined(AGY_REG_COMB) && AGY_REG_COMB
    constexpr int kAgyRegCombMode = AGY_REG_COMB;
#else
    constexpr int kAgyRegCombMode = 0;
#endif
    constexpr bool kAgyRegComb = kAgyRegCombMode > 0;
#if (defined(AGY_FAST_COMB) && AGY_FAST_COMB) || (defined(AGY_REG_COMB) && AGY_REG_COMB)
    constexpr bool kAgyFastComb = true;
#else
    constexpr bool kAgyFastComb = false;
#endif
#if defined(AGY_FAST_SWAP_EPI) && AGY_FAST_SWAP_EPI
    constexpr bool kAgyFastSwapEpi = true;
#else
    constexpr bool kAgyFastSwapEpi = false;
#endif
#if defined(AGY_SWIZZLE_TILES) && AGY_SWIZZLE_TILES
    constexpr bool kAgySwizzleTiles = true;
#else
    constexpr bool kAgySwizzleTiles = false;
#endif
#if defined(AGY_GROUP_M)
    constexpr uint32_t kAgyGroupMDefault = static_cast<uint32_t>(AGY_GROUP_M);
#else
    constexpr uint32_t kAgyGroupMDefault = 4u;
#endif
#if defined(AGY_L1_GROUP_M)
    constexpr uint32_t kAgyL1GroupM = static_cast<uint32_t>(AGY_L1_GROUP_M);
#else
    constexpr uint32_t kAgyL1GroupM = kAgySwizzleTiles ? kAgyGroupMDefault : 0u;
#endif
#if defined(AGY_L2_GROUP_M)
    constexpr uint32_t kAgyL2GroupM = static_cast<uint32_t>(AGY_L2_GROUP_M);
#else
    constexpr uint32_t kAgyL2GroupM = kAgySwizzleTiles ? kAgyGroupMDefault : 0u;
#endif

    constexpr bool kAgyAllIdleCombActive =
        kAgyAllIdleCombReq and MegaMoEPhase::is_fused and kAgyFastComb and kAgyRegComb and
        ((kHidden % 512u) == 0u);
    constexpr uint32_t kExpertCacheBaseBytes =
        (7u * kNumExpertsPerRank + 19u) * static_cast<uint32_t>(sizeof(uint32_t));
    constexpr uint32_t kNumExpertWaves =
        math::constexpr_ceil_div(kNumExpertsPerRank, kNumExpertsPerWave);
    constexpr uint32_t kInterleaveDoneCapWords =
        (kAgyInterleaveMode == 3 or ((kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) and (2u * L1_OUT_BLOCK_N == BLOCK_K)))
            ? (kNumMaxPoolTokens / BLOCK_M)
            : ((kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) ? 0u : (kNumExpertsPerRank + kNumExpertWaves));
    DG_STATIC_ASSERT((not kAgyInterleaveActive) or
                     ((kAgyDedupDisp ? (kNumRanks * kNumMaxTokensPerRank) : 0u) +
                      kInterleaveDoneCapWords + (kNumMaxPoolTokens / BLOCK_M) <=
                      3u * (kNumMaxPoolTokens / 8u)),
                     "Interleave L1/L2 counters + ready queue must fit inside Workspace ring count buffers");
    constexpr uint32_t kExpertCacheRankBytes = kAgyCacheRankCountsSmem
        ? (kNumExpertsPerRank * kNumRanks * static_cast<uint32_t>(sizeof(uint16_t)))
        : 0u;
    constexpr uint32_t SMEM_EXPERT_CACHE_SIZE = kAgyUseExpertCache
        ? math::constexpr_align<uint32_t>(kExpertCacheBaseBytes + kExpertCacheRankBytes, 128u)
        : 0u;
    constexpr uint32_t kDedupFifoCap = 16u;
    constexpr uint32_t SMEM_DEDUP_FIFO_SIZE = (kAgyDedupDispReq and MegaMoEPhase::runs_linear1)
        ? kNumDispatchWarps * kDedupFifoCap * static_cast<uint32_t>(sizeof(AgyDedupDeferredEntry))
        : 0u;

    DG_STATIC_ASSERT(!kScatDispActive or (SMEM_DISPATCH_TOTAL_SIZE + SMEM_CD_SIZE >= kScatL2RawBytes),
                     "Dispatch scatter staging region must fit within pre-A shared memory");
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_DISPATCH_TOTAL_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    constexpr uint32_t kCombineInputHiddenBytes = kHidden * kCombineElementBytes;
    constexpr uint32_t kCombineOutputHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kCombineNumChunks = layout::get_sm90_moe_combine_num_chunks(
        kHidden, kNumEpilogueWarps, SMEM_BEFORE_BARRIER_SIZE,
        kCombineElementBytes, sizeof(nv_bfloat16));
    constexpr uint32_t kCombineChunkElems = kHidden / kCombineNumChunks;
    constexpr uint32_t kCombineInputChunkBytes =
        kCombineInputHiddenBytes / kCombineNumChunks;
    constexpr uint32_t kCombineOutputChunkBytes =
        kCombineOutputHiddenBytes / kCombineNumChunks;
    constexpr uint32_t SMEM_COMBINE_ALIAS_SIZE = MegaMoEPhase::runs_linear2
        ? kNumEpilogueWarps *
            (2 * kCombineInputChunkBytes + kCombineOutputChunkBytes) : 0u;
    DG_STATIC_ASSERT(kHidden % kCombineNumChunks == 0, "Hidden must be divisible by number of combine chunks");
    DG_STATIC_ASSERT(layout::is_sm90_moe_combine_vectorization_legal(
                         kHidden, kNumEpilogueWarps, SMEM_BEFORE_BARRIER_SIZE,
                         kCombineElementBytes, sizeof(nv_bfloat16)),
                     "Hidden is incompatible with the combine vectorization");
    DG_STATIC_ASSERT(SMEM_COMBINE_ALIAS_SIZE <= SMEM_BEFORE_BARRIER_SIZE,
                     "Combine SMEM alias exceeds the pre-barrier scratch region");

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
#if (defined(AGY_ALIAS_DISPATCH_SMEM) && AGY_ALIAS_DISPATCH_SMEM) || (defined(AGY_L2_OMIT_DISPATCH_SMEM) && AGY_L2_OMIT_DISPATCH_SMEM)
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1, smem_buffer);
#else
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
#endif

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_DISPATCH_TOTAL_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 and L2; reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_cd_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_cd_base, SMEM_CD_SWAP_L1_FP32_SIZE));

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    using ExpertCacheT = AgySm90ExpertCache<kNumExpertsPerRank, kNumRanks, kAgyCacheRankCountsSmem>;
    DG_STATIC_ASSERT((not kAgyUseExpertCache) or (sizeof(ExpertCacheT) <= SMEM_EXPERT_CACHE_SIZE),
                     "SMEM_EXPERT_CACHE_SIZE must cover AgySm90ExpertCache");
    auto smem_expert_cache = reinterpret_cast<ExpertCacheT*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto smem_dedup_fifo_base = reinterpret_cast<AgyDedupDeferredEntry*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + SMEM_EXPERT_CACHE_SIZE);
    // Barriers live after the activation-SF stages (+ optional expert cache & dedup FIFO).
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        SMEM_EXPERT_CACHE_SIZE + SMEM_DEDUP_FIFO_SIZE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    constexpr uint32_t kDynBarrierOffset = kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
    auto dyn_ready_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kDynBarrierOffset + i; });
    auto dyn_free_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kDynBarrierOffset + 8 + i; });
    constexpr uint32_t kScatBarrierOffset = kDynBarrierOffset + (kAgyDynSched ? 16u : 0u);
    DG_STATIC_ASSERT(!kScatDispActive or ((kScatBarrierOffset + 2u) % 2u == 0u),
                     "scat_meta_ptr must be 16-byte aligned");
    Barrier* scat_ready_bar = barrier_start_ptr + kScatBarrierOffset;
    Barrier* scat_free_bar  = barrier_start_ptr + kScatBarrierOffset + 1;
    auto* scat_meta_ptr     = reinterpret_cast<uint4*>(barrier_start_ptr + kScatBarrierOffset + 2);
    auto* smem_scat_l2      = reinterpret_cast<nv_bfloat16*>(smem_buffer);

    // Initialization
    if constexpr (kRoleMode <= 1) {
    if (warp_idx == 0) {
        if constexpr (MegaMoEPhase::runs_linear1) {
            // Clean expert-count shared memory
            #pragma unroll
            for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
                ptx::st_shared(smem_expert_count + i, 0u);
        }
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Producer arrivals: A(+SFA) + B.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kAgyMcast * kNumEpilogueWarps);
            }
            if constexpr (MegaMoEPhase::runs_linear2) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                    combine_barriers[i]->init(1);
            }
            if constexpr (kAgyDynSched) {
                #pragma unroll
                for (uint32_t i = 0; i < 8; ++ i) {
                    dyn_ready_barriers[i]->init(1);
                    dyn_free_barriers[i]->init(
                        kAgyMcast == 2u ? (3u + 2u * kNumEpilogueWarps) : (1u + kNumEpilogueWarps));
                }
            }
            if constexpr (kScatDispActive) {
                scat_ready_bar->init(kNumEpilogueWarps);
                scat_free_bar->init(kNumDispatchWarps);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kAgyMcast == 2u) {
        comm::cluster_sync_with_relaxed_arrive();
    } else {
        __syncthreads();
    }
    AGY_PROF_AFTER_CTA_INIT(MegaMoEPhase, sm_idx, thread_idx);
    }

    // Scheduler (cluster=1)
    auto scheduler = sched::SM90MegaMoESchedulerAdapter<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave, kNumSMs, kNumRanks,
        MegaMoEPhase::l2_nmajor_schedule,
        MegaMoEPhase::l1_nmajor_schedule>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register allocations keep each launch within Hopper's quantized CTA launch grant.
    constexpr uint32_t kNumCTAThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kLaunchRegisterGrantUpperBound =
        (65536u / kNumCTAThreads / 8u) * 8u * kNumCTAThreads;
    constexpr bool kCompactFrontendWarpgroup = (kNumMMANonEpilogueWarps == 2);
    constexpr bool kSingleFrontendWarpgroup =
        (kNumDispatchThreads == 64 and kNumNonEpilogueThreads == 64);
    constexpr bool kAgyBoostFrontendRegs =
        kAgyUseExpertCache or kAgyFastPull or kAgyDedupDisp or
        kAgyFastSwapEpi or kAgyFastSwapL2 or kAgyFastClean or kAgyRegComb;
    constexpr uint32_t kDefaultDispatchRegisters =
        kNumEpilogueThreads == 512
            ? (kNumCTAThreads > 640u ? 24u : 32u)
            : ((kNumDispatchThreads == 192)
                   ? 56u
                   : ((kNumDispatchWarps >= 8u)
                          ? (kAgyDedupDispReq ? 56u : 48u)
                          : ((kAgyBoostFrontendRegs and kSingleFrontendWarpgroup) ? 72u :
                             (kAgyBoostFrontendRegs ? 56u : 48u))));
#if defined(AGY_FRONT_REGS) && (AGY_FRONT_REGS > 0)
    constexpr uint32_t kNumDispatchRegisters = static_cast<uint32_t>(AGY_FRONT_REGS);
#else
    constexpr uint32_t kNumDispatchRegisters = kDefaultDispatchRegisters;
#endif
    constexpr uint32_t kDefaultNonEpilogueRegisters = kCompactFrontendWarpgroup
        ? kNumDispatchRegisters
        : (kNumEpilogueThreads == 512 ? 24u : ((kNumDispatchWarps >= 8u) ? 32u : (kAgyBoostFrontendRegs ? 32u : 40u)));
#if defined(AGY_NON_EPI_REGS) && (AGY_NON_EPI_REGS > 0)
    constexpr uint32_t kNumNonEpilogueRegisters = kCompactFrontendWarpgroup
        ? kNumDispatchRegisters
        : static_cast<uint32_t>(AGY_NON_EPI_REGS);
#else
    constexpr uint32_t kNumNonEpilogueRegisters = kDefaultNonEpilogueRegisters;
#endif
    constexpr uint32_t kDefaultEpilogueRegisters =
        kNumEpilogueThreads == 512
            ? (kNumCTAThreads > 640u ? 104u : 112u)
            : ((kNumDispatchThreads == 192) ? 192u :
               ((kNumDispatchWarps >= 8u) ? (kAgyDedupDispReq ? 168u : 176u) : 208u));
#if defined(AGY_MATH_REGS) && (AGY_MATH_REGS > 0)
    constexpr uint32_t kNumEpilogueRegisters = static_cast<uint32_t>(AGY_MATH_REGS);
#else
    constexpr uint32_t kNumEpilogueRegisters = kDefaultEpilogueRegisters;
#endif
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= kLaunchRegisterGrantUpperBound,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    constexpr uint32_t kFusedGridSyncIndex    = 2;
    constexpr uint32_t kFusedPhaseBarrierIdx  = 7;
    constexpr uint32_t kPreGemmSyncThreads    = kAgyUseExpertCache
        ? (kNumDispatchThreads + 64u + kNumEpilogueThreads)
        : (kNumDispatchThreads + kNumEpilogueThreads);

    uint32_t dyn_slot = 0, dyn_phase = 0;
    const auto advance_dyn_ring = [&]() {
        dyn_slot = (dyn_slot + 1u) & 7u;
        dyn_phase ^= (dyn_slot == 0u);
    };
    const auto dsmem_publish_peer_ring = [&](const uint32_t& slot, const uint32_t& peer_val) {
        const uint32_t local_smem_u32 = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_expert_cache->dyn_tile_ring + slot));
        uint32_t peer_dsmem_u32;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;\n"
            "st.shared::cluster.u32 [%0], %3;\n"
            "fence.acq_rel.cluster;\n"
            : "=r"(peer_dsmem_u32)
            : "r"(local_smem_u32), "r"(1u), "r"(peer_val)
            : "memory"
        );
        dyn_ready_barriers[slot]->arrive(1u);
    };

    const auto load_scheduler_from_expert_cache = [&]() {
        #pragma unroll
        for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumExpertsPerRank, 32u); ++ i) {
            const uint32_t e = i * 32u + lane_idx;
            scheduler.stored_num_tokens_per_expert[i] =
                (e < kNumExpertsPerRank) ? smem_expert_cache->recv_tokens[e] : 0u;
        }
        __syncwarp();
    };

    const auto populate_smem_expert_cache = [&]() {
        if (warp_idx == 0) {
            uint32_t run_pool = 0u, run_l1 = 0u, run_l2 = 0u;
            if (lane_idx == 0) {
                smem_expert_cache->pool_block_offset[0] = 0u;
                smem_expert_cache->l1_tile_prefix[0] = 0u;
                smem_expert_cache->l2_tile_prefix[0] = 0u;
            }
            for (uint32_t base_e = 0; base_e < kNumExpertsPerRank; base_e += 32u) {
                const uint32_t e = base_e + lane_idx;
                uint32_t cnt = 0u;
                if (e < kNumExpertsPerRank) {
                    uint64_t v = 0;
                    if constexpr (MegaMoEPhase::runs_linear1) {
                        v = ptx::ld_acq_sys(workspace.get_expert_recv_count_sum_ptr(e));
                    } else {
                        do {
                            v = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(e));
                        } while (static_cast<uint32_t>(v >> 32) != kNumSMs * kNumRanks);
                    }
                    cnt = static_cast<uint32_t>(v);
                }
                const uint32_t mb = math::ceil_div(cnt, BLOCK_M);
                const uint32_t ebm = (kAgyAdaptBm and cnt > 0u) ? (((math::ceil_div(cnt, mb) + 15u) >> 4) << 4) : BLOCK_M;
                const uint32_t l1_t = mb * (L1_SHAPE_N / BLOCK_N);
                const uint32_t l2_t = mb * (L2_SHAPE_N / BLOCK_N);
                uint32_t scan_mb = mb, scan_l1 = l1_t, scan_l2 = l2_t;
                #pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const uint32_t n_mb = __shfl_up_sync(0xffffffff, scan_mb, offset);
                    const uint32_t n_l1 = __shfl_up_sync(0xffffffff, scan_l1, offset);
                    const uint32_t n_l2 = __shfl_up_sync(0xffffffff, scan_l2, offset);
                    if (static_cast<int>(lane_idx) >= offset) {
                        scan_mb += n_mb;
                        scan_l1 += n_l1;
                        scan_l2 += n_l2;
                    }
                }
                if (e < kNumExpertsPerRank) {
                    smem_expert_cache->recv_tokens[e] = cnt;
                    smem_expert_cache->m_blocks[e] = mb;
                    smem_expert_cache->expert_bm[e] = ebm;
                    smem_expert_cache->pool_block_offset[e + 1] = run_pool + scan_mb;
                    if constexpr (kAgyHotColdMode == 0 or kNumExpertsPerRank > 64u) {
                        smem_expert_cache->exec_expert[e] = e;
                        smem_expert_cache->l1_tile_prefix[e + 1] = run_l1 + scan_l1;
                        smem_expert_cache->l2_tile_prefix[e + 1] = run_l2 + scan_l2;
                    }
                }
                run_pool += __shfl_sync(0xffffffff, scan_mb, 31);
                run_l1   += __shfl_sync(0xffffffff, scan_l1, 31);
                run_l2   += __shfl_sync(0xffffffff, scan_l2, 31);
            }
            __syncwarp();
            if constexpr (kAgyHotColdMode > 0 and kNumExpertsPerRank <= 64u) {
                // In-register 64-element bitonic sort (< 0.2 us) for LPT / Hot-Cold expert pairing!
                const uint32_t i0 = 2u * lane_idx;
                const uint32_t i1 = 2u * lane_idx + 1u;
                uint32_t kv0 = (i0 < kNumExpertsPerRank)
                    ? ((smem_expert_cache->m_blocks[i0] << 16) | (63u - i0))
                    : 0u;
                uint32_t kv1 = (i1 < kNumExpertsPerRank)
                    ? ((smem_expert_cache->m_blocks[i1] << 16) | (63u - i1))
                    : 0u;
                auto cmp_swap_pair = [](uint32_t& a, uint32_t& b, const bool desc) {
                    if ((a < b) == desc) { const uint32_t t = a; a = b; b = t; }
                };
                for (uint32_t k = 2u; k <= 64u; k <<= 1) {
                    for (uint32_t j = (k >> 1); j > 0u; j >>= 1) {
                        if (j == 1u) {
                            const bool desc = ((i0 & k) == 0u);
                            cmp_swap_pair(kv0, kv1, desc);
                        } else {
                            const uint32_t lane_xor = j >> 1;
                            const uint32_t peer0 = __shfl_xor_sync(0xffffffff, kv0, lane_xor);
                            const uint32_t peer1 = __shfl_xor_sync(0xffffffff, kv1, lane_xor);
                            const bool low_lane = ((lane_idx & lane_xor) == 0u);
                            const bool desc0 = ((i0 & k) == 0u);
                            const bool desc1 = ((i1 & k) == 0u);
                            if ((kv0 < peer0) == (desc0 == low_lane)) kv0 = peer0;
                            if ((kv1 < peer1) == (desc1 == low_lane)) kv1 = peer1;
                        }
                    }
                }
                // Write sorted descending order into l1_tile_prefix scratch first, then pair hot+cold if mode==2
                if (i0 < kNumExpertsPerRank) smem_expert_cache->exec_expert[i0] = 63u - (kv0 & 63u);
                if (i1 < kNumExpertsPerRank) smem_expert_cache->exec_expert[i1] = 63u - (kv1 & 63u);
                __syncwarp();
                if constexpr (kAgyHotColdMode == 2) {
                    uint32_t tmp0 = (i0 < kNumExpertsPerRank)
                        ? smem_expert_cache->exec_expert[(i0 & 1u) == 0u ? (i0 >> 1) : (kNumExpertsPerRank - 1u - (i0 >> 1))]
                        : 0u;
                    uint32_t tmp1 = (i1 < kNumExpertsPerRank)
                        ? smem_expert_cache->exec_expert[(i1 & 1u) == 0u ? (i1 >> 1) : (kNumExpertsPerRank - 1u - (i1 >> 1))]
                        : 0u;
                    __syncwarp();
                    if (i0 < kNumExpertsPerRank) smem_expert_cache->exec_expert[i0] = tmp0;
                    if (i1 < kNumExpertsPerRank) smem_expert_cache->exec_expert[i1] = tmp1;
                    __syncwarp();
                }
                uint32_t ord_l1 = 0u, ord_l2 = 0u;
                for (uint32_t base_o = 0u; base_o < kNumExpertsPerRank; base_o += 32u) {
                    const uint32_t o = base_o + lane_idx;
                    const uint32_t mb_o = (o < kNumExpertsPerRank)
                        ? smem_expert_cache->m_blocks[smem_expert_cache->exec_expert[o]]
                        : 0u;
                    uint32_t s_l1 = mb_o * (L1_SHAPE_N / BLOCK_N);
                    uint32_t s_l2 = mb_o * (L2_SHAPE_N / BLOCK_N);
                    #pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        const uint32_t n_l1 = __shfl_up_sync(0xffffffff, s_l1, offset);
                        const uint32_t n_l2 = __shfl_up_sync(0xffffffff, s_l2, offset);
                        if (static_cast<int>(lane_idx) >= offset) { s_l1 += n_l1; s_l2 += n_l2; }
                    }
                    if (o < kNumExpertsPerRank) {
                        smem_expert_cache->l1_tile_prefix[o + 1] = ord_l1 + s_l1;
                        smem_expert_cache->l2_tile_prefix[o + 1] = ord_l2 + s_l2;
                    }
                    ord_l1 += __shfl_sync(0xffffffff, s_l1, 31);
                    ord_l2 += __shfl_sync(0xffffffff, s_l2, 31);
                }
                __syncwarp();
            }
        } else if constexpr (kAgyCacheRankCounts) {
            DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Rank count cache expects >= 2 dispatch warps");
            constexpr uint32_t kRankCacheLoadThreads = (kNumDispatchWarps - 1u) * 32u;
            constexpr uint32_t kTotalRankExpertEntries = kNumRanks * kNumExpertsPerRank;
            const uint32_t rc_tid = (warp_idx - 1u) * 32u + lane_idx;
            #pragma unroll 4
            for (uint32_t idx = rc_tid; idx < kTotalRankExpertEntries; idx += kRankCacheLoadThreads) {
                const uint32_t r = idx / kNumExpertsPerRank;
                const uint32_t e = idx - r * kNumExpertsPerRank;
                const uint32_t cnt = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_ptr(r, e));
                smem_expert_cache->set_rank_count(e, r, static_cast<uint16_t>(cnt));
            }
            __syncwarp();
        }
    };

    const auto run_single_phase_blocks = [&](const bool is_l1, const int role_id, auto&& func) {
        const uint32_t phase_block_ks = is_l1 ? (L1_SHAPE_K / BLOCK_K) : (L2_SHAPE_K / BLOCK_K);
        const uint32_t phase_block_ns = is_l1 ? (L1_SHAPE_N / BLOCK_N) : (L2_SHAPE_N / BLOCK_N);
        const bool phase_nmajor = is_l1 ? MegaMoEPhase::l1_nmajor_schedule : MegaMoEPhase::l2_nmajor_schedule;

        if constexpr (kAgyDynSched) {
            const uint32_t* tile_prefix = is_l1 ? smem_expert_cache->l1_tile_prefix
                                                : smem_expert_cache->l2_tile_prefix;
            const uint32_t total_tiles = tile_prefix[kNumExpertsPerRank];
            if (sm_idx >= total_tiles)
                return;
            uint32_t* task_ptr = is_l1 ? workspace.get_l1_task_count_ptr()
                                       : workspace.get_l2_task_count_ptr();

            auto dispatch_tile_id = [&](const uint32_t& tile_id) {
                uint32_t lo = 0u, hi = kNumExpertsPerRank;
                while (hi - lo > 1u) {
                    const uint32_t mid = (lo + hi) >> 1;
                    if (tile_prefix[mid] <= tile_id) lo = mid;
                    else hi = mid;
                }
                const uint32_t local_tile = tile_id - tile_prefix[lo];
                const uint32_t local_expert_idx = smem_expert_cache->exec_expert[lo];
                const uint32_t num_m_blocks = smem_expert_cache->m_blocks[local_expert_idx];
                const uint32_t recv_tokens = smem_expert_cache->recv_tokens[local_expert_idx];
                const uint32_t pool_block_offset = smem_expert_cache->pool_block_offset[local_expert_idx];
                uint32_t m_block_idx, n_block_idx;
                if constexpr (kAgyMcast == 2u) {
                    const uint32_t pair_m = num_m_blocks & ~1u;
                    const uint32_t pair_tiles = pair_m * phase_block_ns;
                    if (local_tile < pair_tiles) {
                        const uint32_t raw_gm = is_l1 ? (kAgyL1GroupM > 1u ? kAgyL1GroupM : 2u)
                                                      : (kAgyL2GroupM > 1u ? kAgyL2GroupM : 2u);
                        const uint32_t even_gm = (raw_gm >= 2u) ? (raw_gm & ~1u) : 2u;
                        const uint32_t tiles_per_group = even_gm * phase_block_ns;
                        const uint32_t group_id = local_tile / tiles_per_group;
                        const uint32_t in_group = local_tile - group_id * tiles_per_group;
                        const uint32_t group_base_m = group_id * even_gm;
                        const uint32_t rem_m = pair_m - group_base_m;
                        const uint32_t eff_gm = rem_m < even_gm ? rem_m : even_gm;
                        n_block_idx = in_group / eff_gm;
                        m_block_idx = group_base_m + (in_group - n_block_idx * eff_gm);
                    } else {
                        m_block_idx = pair_m;
                        n_block_idx = local_tile - pair_tiles;
                    }
                } else if constexpr (kAgySwizzleTiles) {
                    const uint32_t phase_group_m = is_l1 ? kAgyL1GroupM : kAgyL2GroupM;
                    if (phase_group_m > 1u) {
                        if (is_l1) {
                            constexpr uint32_t kL1GroupM = kAgyL1GroupM > 1u ? kAgyL1GroupM : 4u;
                            constexpr uint32_t kL1TilesPerGroup = kL1GroupM * (L1_SHAPE_N / BLOCK_N);
                            const uint32_t group_id = local_tile / kL1TilesPerGroup;
                            const uint32_t in_group = local_tile - group_id * kL1TilesPerGroup;
                            const uint32_t group_base_m = group_id * kL1GroupM;
                            const uint32_t rem_m = num_m_blocks - group_base_m;
                            if (rem_m >= kL1GroupM) {
                                n_block_idx = in_group / kL1GroupM;
                                m_block_idx = group_base_m + (in_group - n_block_idx * kL1GroupM);
                            } else {
                                n_block_idx = in_group / rem_m;
                                m_block_idx = group_base_m + (in_group - n_block_idx * rem_m);
                            }
                        } else {
                            constexpr uint32_t kL2GroupM = kAgyL2GroupM > 1u ? kAgyL2GroupM : 4u;
                            constexpr uint32_t kL2TilesPerGroup = kL2GroupM * (L2_SHAPE_N / BLOCK_N);
                            const uint32_t group_id = local_tile / kL2TilesPerGroup;
                            const uint32_t in_group = local_tile - group_id * kL2TilesPerGroup;
                            const uint32_t group_base_m = group_id * kL2GroupM;
                            const uint32_t rem_m = num_m_blocks - group_base_m;
                            if (rem_m >= kL2GroupM) {
                                n_block_idx = in_group / kL2GroupM;
                                m_block_idx = group_base_m + (in_group - n_block_idx * kL2GroupM);
                            } else {
                                n_block_idx = in_group / rem_m;
                                m_block_idx = group_base_m + (in_group - n_block_idx * rem_m);
                            }
                        }
                    } else if (phase_nmajor) {
                        n_block_idx = local_tile / num_m_blocks;
                        m_block_idx = local_tile - n_block_idx * num_m_blocks;
                    } else {
                        m_block_idx = local_tile / phase_block_ns;
                        n_block_idx = local_tile - m_block_idx * phase_block_ns;
                    }
                } else if (phase_nmajor) {
                    n_block_idx = local_tile / num_m_blocks;
                    m_block_idx = local_tile - n_block_idx * num_m_blocks;
                } else {
                    m_block_idx = local_tile / phase_block_ns;
                    n_block_idx = local_tile - m_block_idx * phase_block_ns;
                }
                const uint32_t ebm = kAgyAdaptBm ? smem_expert_cache->expert_bm[local_expert_idx] : BLOCK_M;
                const uint32_t m_start = m_block_idx * ebm;
                const uint32_t valid_m = m_start < recv_tokens ? cute::min(recv_tokens - m_start, ebm) : 0u;
                func(is_l1, local_expert_idx, phase_block_ks, m_block_idx, n_block_idx,
                     pool_block_offset + m_block_idx, valid_m);
            };

            if constexpr (kAgyStaticStride) {
                for (uint32_t next_tile = sm_idx; next_tile < total_tiles; next_tile += kNumSMs) {
                    dispatch_tile_id(next_tile);
                }
            } else {
                // Wave 0 starts with deterministic sm_idx (zero atomics), then dynamic ring
                uint32_t next_tile = sm_idx;
                while (true) {
                    dispatch_tile_id(next_tile);
                    if (total_tiles <= kNumSMs)
                        break;
                    if constexpr (kAgyMcast == 2u) {
                        const uint32_t cta_rank = cute::block_rank_in_cluster();
                        if (role_id == 1 and cta_rank == 0u) {
                            dyn_free_barriers[dyn_slot]->wait(dyn_phase ^ 1u);
                            if (lane_idx == 0) {
                                next_tile = kNumSMs + ptx::atomic_add(task_ptr, 2u);
                                ptx::st_shared(smem_expert_cache->dyn_tile_ring + dyn_slot, next_tile);
                                dyn_ready_barriers[dyn_slot]->arrive();
                                dsmem_publish_peer_ring(dyn_slot, next_tile + 1u);
                            }
                            next_tile = __shfl_sync(0xffffffff, next_tile, 0);
                        } else {
                            dyn_ready_barriers[dyn_slot]->wait(dyn_phase);
                            next_tile = ptx::ld_shared(smem_expert_cache->dyn_tile_ring + dyn_slot);
                            if (lane_idx == 0)
                                dyn_free_barriers[dyn_slot]->arrive(0u);
                        }
                    } else {
                        if (role_id == 1) {
                            // TMA B producer fetches next global tile index
                            dyn_free_barriers[dyn_slot]->wait(dyn_phase ^ 1u);
                            if (lane_idx == 0) {
                                next_tile = kNumSMs + ptx::atomic_add(task_ptr, 1u);
                                ptx::st_shared(smem_expert_cache->dyn_tile_ring + dyn_slot, next_tile);
                                dyn_ready_barriers[dyn_slot]->arrive();
                            }
                            next_tile = __shfl_sync(0xffffffff, next_tile, 0);
                        } else {
                            // TMA A (role_id == 0) and Role 3 warps (role_id == 2) consume next tile index
                            dyn_ready_barriers[dyn_slot]->wait(dyn_phase);
                            next_tile = ptx::ld_shared(smem_expert_cache->dyn_tile_ring + dyn_slot);
                            if (lane_idx == 0)
                                dyn_free_barriers[dyn_slot]->arrive();
                        }
                    }
                    advance_dyn_ring();
                    if (next_tile >= total_tiles)
                        break;
                }
            }
        } else {
            if constexpr (kAgyUseExpertCache) {
                load_scheduler_from_expert_cache();
            } else {
                scheduler.fetch_expert_recv_count();
            }
            scheduler.block_idx = sm_idx;
            scheduler.set_expert_idx(0);
            if (is_l1) {
                while (scheduler.current_local_expert_idx < kNumExpertsPerRank) {
                    if (scheduler.template fetch_next_phase_block<sched::SM90BlockPhase::Linear1>()) {
                        scheduler.block_idx += kNumSMs;
                        func(true, scheduler.current_local_expert_idx, L1_SHAPE_K / BLOCK_K,
                             scheduler.m_block_idx, scheduler.n_block_idx,
                             scheduler.get_current_pool_block_offset() + scheduler.m_block_idx,
                             scheduler.template get_valid_m<false>());
                    } else if (scheduler.current_local_expert_idx >= kNumExpertsPerRank) {
                        break;
                    }
                }
            } else {
                while (scheduler.current_local_expert_idx < kNumExpertsPerRank) {
                    if (scheduler.template fetch_next_phase_block<sched::SM90BlockPhase::Linear2>()) {
                        scheduler.block_idx += kNumSMs;
                        func(false, scheduler.current_local_expert_idx, L2_SHAPE_K / BLOCK_K,
                             scheduler.m_block_idx, scheduler.n_block_idx,
                             scheduler.get_current_pool_block_offset() + scheduler.m_block_idx,
                             scheduler.template get_valid_m<false>());
                    } else if (scheduler.current_local_expert_idx >= kNumExpertsPerRank) {
                        break;
                    }
                }
            }
        }
    };

    uint32_t* const interleave_l1_done_base =
        workspace.get_l1_empty_count_ptr(0) + (kAgyDedupDisp ? (kNumRanks * kNumMaxTokensPerRank) : 0u);
    uint32_t* const interleave_wave_cursor_base =
        interleave_l1_done_base + kNumExpertsPerRank;
    uint32_t* const interleave_ready_mb_queue =
        interleave_l1_done_base + kInterleaveDoneCapWords;

    const auto run_interleaved_l1_l2_blocks = [&](const int role_id, const uint32_t& fused_tid, auto&& func) {
        constexpr uint32_t kL1BlockKs = L1_SHAPE_K / BLOCK_K;
        constexpr uint32_t kL2BlockKs = L2_SHAPE_K / BLOCK_K;
        constexpr uint32_t kL1BlockNs = L1_SHAPE_N / BLOCK_N;
        constexpr uint32_t kL2BlockNs = L2_SHAPE_N / BLOCK_N;

        const uint32_t total_l1_tiles = smem_expert_cache->l1_tile_prefix[kNumExpertsPerRank];
        const uint32_t total_l2_tiles = smem_expert_cache->l2_tile_prefix[kNumExpertsPerRank];
        if (total_l1_tiles == 0u)
            return;

        uint32_t* const l1_cursor_ptr      = workspace.get_l1_task_count_ptr();
        uint32_t* const l2_cursor_ptr      = workspace.get_l2_task_count_ptr();
        uint32_t* const ready_l2_avail_ptr = workspace.get_shared_l2_task_count_ptr();

        auto dispatch_interleaved_l1 = [&](const uint32_t& tile_id) {
            uint32_t lo = 0u, hi = kNumExpertsPerRank;
            while (hi - lo > 1u) {
                const uint32_t mid = (lo + hi) >> 1;
                if (smem_expert_cache->l1_tile_prefix[mid] <= tile_id) lo = mid;
                else hi = mid;
            }
            const uint32_t local_tile = tile_id - smem_expert_cache->l1_tile_prefix[lo];
            const uint32_t local_expert_idx = smem_expert_cache->exec_expert[lo];
            const uint32_t num_m_blocks = smem_expert_cache->m_blocks[local_expert_idx];
            const uint32_t recv_tokens = smem_expert_cache->recv_tokens[local_expert_idx];
            const uint32_t pool_block_offset = smem_expert_cache->pool_block_offset[local_expert_idx];
            uint32_t m_block_idx, n_block_idx;
            if constexpr (kAgyMcast == 2u) {
                const uint32_t pair_m = num_m_blocks & ~1u;
                const uint32_t pair_tiles = pair_m * kL1BlockNs;
                if (local_tile < pair_tiles) {
                    constexpr uint32_t kRawGm = kAgyL1GroupM > 1u ? kAgyL1GroupM : 2u;
                    constexpr uint32_t kEvenGm = (kRawGm >= 2u) ? (kRawGm & ~1u) : 2u;
                    const uint32_t tiles_per_group = kEvenGm * kL1BlockNs;
                    const uint32_t group_id = local_tile / tiles_per_group;
                    const uint32_t in_group = local_tile - group_id * tiles_per_group;
                    const uint32_t group_base_m = group_id * kEvenGm;
                    const uint32_t rem_m = pair_m - group_base_m;
                    const uint32_t eff_gm = rem_m < kEvenGm ? rem_m : kEvenGm;
                    n_block_idx = in_group / eff_gm;
                    m_block_idx = group_base_m + (in_group - n_block_idx * eff_gm);
                } else {
                    m_block_idx = pair_m;
                    n_block_idx = local_tile - pair_tiles;
                }
            } else if constexpr (kAgySwizzleTiles and (kAgyL1GroupM > 1u)) {
                constexpr uint32_t kL1GroupM = kAgyL1GroupM > 1u ? kAgyL1GroupM : 4u;
                constexpr uint32_t kL1TilesPerGroup = kL1GroupM * kL1BlockNs;
                const uint32_t group_id = local_tile / kL1TilesPerGroup;
                const uint32_t in_group = local_tile - group_id * kL1TilesPerGroup;
                const uint32_t group_base_m = group_id * kL1GroupM;
                const uint32_t rem_m = num_m_blocks - group_base_m;
                if (rem_m >= kL1GroupM) {
                    n_block_idx = in_group / kL1GroupM;
                    m_block_idx = group_base_m + (in_group - n_block_idx * kL1GroupM);
                } else {
                    n_block_idx = in_group / rem_m;
                    m_block_idx = group_base_m + (in_group - n_block_idx * rem_m);
                }
            } else if constexpr (MegaMoEPhase::l1_nmajor_schedule) {
                n_block_idx = local_tile / num_m_blocks;
                m_block_idx = local_tile - n_block_idx * num_m_blocks;
            } else {
                m_block_idx = local_tile / kL1BlockNs;
                n_block_idx = local_tile - m_block_idx * kL1BlockNs;
            }
            const uint32_t ebm = kAgyAdaptBm ? smem_expert_cache->expert_bm[local_expert_idx] : BLOCK_M;
            const uint32_t m_start = m_block_idx * ebm;
            const uint32_t valid_m = m_start < recv_tokens ? cute::min(recv_tokens - m_start, ebm) : 0u;
            AGY_PROF_INTERLEAVE_TILE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, true);
            func(true, local_expert_idx, kL1BlockKs, m_block_idx, n_block_idx,
                 pool_block_offset + m_block_idx, valid_m);
            AGY_PROF_INTERLEAVE_TILE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, true);
        };

        auto dispatch_interleaved_l2_direct_tile = [&](const uint32_t& l2_tile_id) {
            uint32_t lo = 0u, hi = kNumExpertsPerRank;
            while (hi - lo > 1u) {
                const uint32_t mid = (lo + hi) >> 1;
                if (smem_expert_cache->l2_tile_prefix[mid] <= l2_tile_id) lo = mid;
                else hi = mid;
            }
            const uint32_t local_tile = l2_tile_id - smem_expert_cache->l2_tile_prefix[lo];
            const uint32_t local_expert_idx = smem_expert_cache->exec_expert[lo];
            const uint32_t num_m_blocks = smem_expert_cache->m_blocks[local_expert_idx];
            const uint32_t recv_tokens = smem_expert_cache->recv_tokens[local_expert_idx];
            const uint32_t pool_block_offset = smem_expert_cache->pool_block_offset[local_expert_idx];
            uint32_t m_block_idx, n_block_idx;
            if constexpr (kAgyMcast == 2u) {
                const uint32_t pair_m = num_m_blocks & ~1u;
                const uint32_t pair_tiles = pair_m * kL2BlockNs;
                if (local_tile < pair_tiles) {
                    constexpr uint32_t kRawGm = kAgyL2GroupM > 1u ? kAgyL2GroupM : 2u;
                    constexpr uint32_t kEvenGm = (kRawGm >= 2u) ? (kRawGm & ~1u) : 2u;
                    const uint32_t tiles_per_group = kEvenGm * kL2BlockNs;
                    const uint32_t group_id = local_tile / tiles_per_group;
                    const uint32_t in_group = local_tile - group_id * tiles_per_group;
                    const uint32_t group_base_m = group_id * kEvenGm;
                    const uint32_t rem_m = pair_m - group_base_m;
                    const uint32_t eff_gm = rem_m < kEvenGm ? rem_m : kEvenGm;
                    n_block_idx = in_group / eff_gm;
                    m_block_idx = group_base_m + (in_group - n_block_idx * eff_gm);
                } else {
                    m_block_idx = pair_m;
                    n_block_idx = local_tile - pair_tiles;
                }
            } else if constexpr (kAgySwizzleTiles and (kAgyL2GroupM > 1u)) {
                constexpr uint32_t kL2GroupM = kAgyL2GroupM > 1u ? kAgyL2GroupM : 4u;
                constexpr uint32_t kL2TilesPerGroup = kL2GroupM * kL2BlockNs;
                const uint32_t group_id = local_tile / kL2TilesPerGroup;
                const uint32_t in_group = local_tile - group_id * kL2TilesPerGroup;
                const uint32_t group_base_m = group_id * kL2GroupM;
                const uint32_t rem_m = num_m_blocks - group_base_m;
                if (rem_m >= kL2GroupM) {
                    n_block_idx = in_group / kL2GroupM;
                    m_block_idx = group_base_m + (in_group - n_block_idx * kL2GroupM);
                } else {
                    n_block_idx = in_group / rem_m;
                    m_block_idx = group_base_m + (in_group - n_block_idx * rem_m);
                }
            } else if constexpr (MegaMoEPhase::l2_nmajor_schedule) {
                n_block_idx = local_tile / num_m_blocks;
                m_block_idx = local_tile - n_block_idx * num_m_blocks;
            } else {
                m_block_idx = local_tile / kL2BlockNs;
                n_block_idx = local_tile - m_block_idx * kL2BlockNs;
            }
            const uint32_t ebm = kAgyAdaptBm ? smem_expert_cache->expert_bm[local_expert_idx] : BLOCK_M;
            const uint32_t m_start = m_block_idx * ebm;
            const uint32_t valid_m = m_start < recv_tokens ? cute::min(recv_tokens - m_start, ebm) : 0u;
            AGY_PROF_INTERLEAVE_TILE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
            func(false, local_expert_idx, kL2BlockKs, m_block_idx, n_block_idx,
                 pool_block_offset + m_block_idx, valid_m);
            AGY_PROF_INTERLEAVE_TILE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
        };

        auto dispatch_interleaved_l2 = [&](const uint32_t& l2_seq_tile, const uint32_t& packed_desc) {
            const uint32_t local_expert_idx = (packed_desc >> 20) - 1u;
            const uint32_t ord_idx = packed_desc & 0xfffffu;
            const uint32_t in_mb_n = l2_seq_tile % kL2BlockNs;
            const uint32_t num_m_blocks = smem_expert_cache->m_blocks[local_expert_idx];
            const uint32_t recv_tokens = smem_expert_cache->recv_tokens[local_expert_idx];
            const uint32_t pool_block_offset = smem_expert_cache->pool_block_offset[local_expert_idx];
            uint32_t m_block_idx = ord_idx;
            uint32_t n_block_idx = in_mb_n;
            if constexpr (kAgyInterleaveMode != 3) {
                const uint32_t local_tile = ord_idx * kL2BlockNs + in_mb_n;
                if constexpr (kAgySwizzleTiles and (kAgyL2GroupM > 1u)) {
                    constexpr uint32_t kL2GroupM = kAgyL2GroupM > 1u ? kAgyL2GroupM : 4u;
                    constexpr uint32_t kL2TilesPerGroup = kL2GroupM * kL2BlockNs;
                    const uint32_t group_id = local_tile / kL2TilesPerGroup;
                    const uint32_t in_group = local_tile - group_id * kL2TilesPerGroup;
                    const uint32_t group_base_m = group_id * kL2GroupM;
                    const uint32_t rem_m = num_m_blocks - group_base_m;
                    if (rem_m >= kL2GroupM) {
                        n_block_idx = in_group / kL2GroupM;
                        m_block_idx = group_base_m + (in_group - n_block_idx * kL2GroupM);
                    } else {
                        n_block_idx = in_group / rem_m;
                        m_block_idx = group_base_m + (in_group - n_block_idx * rem_m);
                    }
                } else if constexpr (MegaMoEPhase::l2_nmajor_schedule) {
                    n_block_idx = local_tile / num_m_blocks;
                    m_block_idx = local_tile - n_block_idx * num_m_blocks;
                }
            }
            const uint32_t ebm = kAgyAdaptBm ? smem_expert_cache->expert_bm[local_expert_idx] : BLOCK_M;
            const uint32_t m_start = m_block_idx * ebm;
            const uint32_t valid_m = m_start < recv_tokens ? cute::min(recv_tokens - m_start, ebm) : 0u;
            AGY_PROF_INTERLEAVE_TILE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
            func(false, local_expert_idx, kL2BlockKs, m_block_idx, n_block_idx,
                 pool_block_offset + m_block_idx, valid_m);
            AGY_PROF_INTERLEAVE_TILE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
        };

        if constexpr (kAgyInterleaveMode == 6) {
            // =========================================================================
            // Mode 6: Wave-Lagged EPW Pipelining (L1(w+1) before L2(w))!
            //   - W <= 1: exact Mode-5 [L1(0), L2(0)] virtual-task order.
            //   - W >= 2: 2*W sub-waves:
            //       s=0        -> L1(0)
            //       s=1        -> L1(1)
            //       even s>=2  -> L2((s-2)/2)   (except the final sub-wave)
            //       odd  s>=3  -> L1((s+1)/2)
            //       s=2W-1     -> L2(W-1)
            //     hence every L2(w) is lagged by at least one full sub-wave.
            //   - Same Mode-5 K-block bitmask release/wait and workspace cleanup.
            // =========================================================================
            const uint32_t total_vtasks = total_l1_tiles + total_l2_tiles;
            const uint32_t w0_e_hi = cute::min(kNumExpertsPerRank, kNumExpertsPerWave);
            const uint32_t wave0_direct_l1 = cute::min(kNumSMs, smem_expert_cache->l1_tile_prefix[w0_e_hi]);
            if (sm_idx < wave0_direct_l1) {
                dispatch_interleaved_l1(sm_idx);
            }
            if (total_vtasks <= wave0_direct_l1)
                return;

            while (true) {
                uint32_t tag = 0xffffffffu;
                const uint32_t cta_rank = (kAgyMcast == 2u) ? cute::block_rank_in_cluster() : 0u;
                if (role_id == 1 and (kAgyMcast != 2u or cta_rank == 0u)) {
                    dyn_free_barriers[dyn_slot]->wait(dyn_phase ^ 1u);
                    if (lane_idx == 0) {
                        const uint32_t vtask = wave0_direct_l1 + ptx::atomic_add(l1_cursor_ptr, kAgyMcast);
                        if (vtask < total_vtasks) {
                            if constexpr (kNumExpertWaves <= 1u) {
                                // Exact Mode-5 single-wave order: all L1(0), then all L2(0).
                                if (vtask < total_l1_tiles) {
                                    tag = vtask;
                                } else {
                                    tag = 0x80000000u | (vtask - total_l1_tiles);
                                }
                            } else {
                                uint32_t cum = 0u;
                                #pragma unroll 1
                                for (uint32_t s = 0u; s < 2u * kNumExpertWaves; ++ s) {
                                    bool is_l2 = false;
                                    uint32_t w = 0u;
                                    if (s == 0u) {
                                        is_l2 = false; w = 0u;
                                    } else if (s == 1u) {
                                        is_l2 = false; w = 1u;
                                    } else if (s == 2u * kNumExpertWaves - 1u) {
                                        is_l2 = true; w = kNumExpertWaves - 1u;
                                    } else if ((s & 1u) == 0u) {
                                        is_l2 = true; w = (s - 2u) >> 1;
                                    } else {
                                        is_l2 = false; w = (s + 1u) >> 1;
                                    }
                                    const uint32_t e_lo = w * kNumExpertsPerWave;
                                    const uint32_t e_hi = cute::min(kNumExpertsPerRank, e_lo + kNumExpertsPerWave);
                                    const uint32_t cnt_s = is_l2
                                        ? (smem_expert_cache->l2_tile_prefix[e_hi] - smem_expert_cache->l2_tile_prefix[e_lo])
                                        : (smem_expert_cache->l1_tile_prefix[e_hi] - smem_expert_cache->l1_tile_prefix[e_lo]);
                                    if (vtask < cum + cnt_s) {
                                        const uint32_t in_sub = vtask - cum;
                                        tag = is_l2
                                            ? (0x80000000u | (smem_expert_cache->l2_tile_prefix[e_lo] + in_sub))
                                            : (smem_expert_cache->l1_tile_prefix[e_lo] + in_sub);
                                        break;
                                    }
                                    cum += cnt_s;
                                }
                            }
                        } else {
                            tag = 0xffffffffu;
                        }
                        ptx::st_shared(smem_expert_cache->dyn_tile_ring + dyn_slot, tag);
                        dyn_ready_barriers[dyn_slot]->arrive();
                        if constexpr (kAgyMcast == 2u) {
                            const uint32_t peer_tag = (tag == 0xffffffffu) ? 0xffffffffu : (tag + 1u);
                            dsmem_publish_peer_ring(dyn_slot, peer_tag);
                        }
                    }
                    tag = __shfl_sync(0xffffffff, tag, 0);
                } else {
                    dyn_ready_barriers[dyn_slot]->wait(dyn_phase);
                    tag = ptx::ld_shared(smem_expert_cache->dyn_tile_ring + dyn_slot);
                    if (lane_idx == 0) {
                        if constexpr (kAgyMcast == 2u)
                            dyn_free_barriers[dyn_slot]->arrive(0u);
                        else
                            dyn_free_barriers[dyn_slot]->arrive();
                    }
                }
                advance_dyn_ring();
                if (tag == 0xffffffffu)
                    break;
                if ((tag & 0x80000000u) == 0u) {
                    dispatch_interleaved_l1(tag);
                } else {
                    dispatch_interleaved_l2_direct_tile(tag & 0x7fffffffu);
                }
            }
            return;
        }

        if constexpr (kAgyInterleaveMode == 5) {
            // =========================================================================
            // Mode 5: Official B300-Style `epw` Wave Alternation + `L2KBlockDependency`!
            //   - Tasks alternate per `epw` wave: [Wave w: L1(epw)] -> [Wave w: L2(epw)].
            //   - TMA Warp B claims `vtask` via `atomicAdd` with ZERO global-memory spin,
            //     immediately prefetching L2 weights from HBM while L1 tiles finish!
            //   - TMA Warp A waits ONLY on `1u << k_block_idx` in `interleave_ready_mb_queue`
            //     inside the L2 K-loop, streaming hot L2-cached activations K-block by K-block!
            // =========================================================================
            const uint32_t total_vtasks = total_l1_tiles + total_l2_tiles;
            const uint32_t w0_e_hi = cute::min(kNumExpertsPerRank, kNumExpertsPerWave);
            const uint32_t wave0_direct_l1 = cute::min(kNumSMs, smem_expert_cache->l1_tile_prefix[w0_e_hi]);
            if (sm_idx < wave0_direct_l1) {
                dispatch_interleaved_l1(sm_idx);
            }
            if (total_vtasks <= wave0_direct_l1)
                return;

            while (true) {
                uint32_t tag = 0xffffffffu;
                if (role_id == 1) {
                    dyn_free_barriers[dyn_slot]->wait(dyn_phase ^ 1u);
                    if (lane_idx == 0) {
                        const uint32_t vtask = wave0_direct_l1 + ptx::atomic_add(l1_cursor_ptr, 1u);
                        if (vtask < total_vtasks) {
                            uint32_t w_lo = 0u, w_hi = kNumExpertWaves;
                            while (w_hi - w_lo > 1u) {
                                const uint32_t w_mid = (w_lo + w_hi) >> 1;
                                const uint32_t e_mid = w_mid * kNumExpertsPerWave;
                                const uint32_t v_mid =
                                    smem_expert_cache->l1_tile_prefix[e_mid] +
                                    smem_expert_cache->l2_tile_prefix[e_mid];
                                if (v_mid <= vtask) w_lo = w_mid;
                                else w_hi = w_mid;
                            }
                            const uint32_t e_lo = w_lo * kNumExpertsPerWave;
                            const uint32_t e_hi = cute::min(kNumExpertsPerRank, e_lo + kNumExpertsPerWave);
                            const uint32_t v_base =
                                smem_expert_cache->l1_tile_prefix[e_lo] +
                                smem_expert_cache->l2_tile_prefix[e_lo];
                            const uint32_t in_wave = vtask - v_base;
                            const uint32_t w_l1 =
                                smem_expert_cache->l1_tile_prefix[e_hi] -
                                smem_expert_cache->l1_tile_prefix[e_lo];
                            if (in_wave < w_l1) {
                                tag = smem_expert_cache->l1_tile_prefix[e_lo] + in_wave;
                            } else {
                                tag = 0x80000000u | (smem_expert_cache->l2_tile_prefix[e_lo] + (in_wave - w_l1));
                            }
                        } else {
                            tag = 0xffffffffu;
                        }
                        ptx::st_shared(smem_expert_cache->dyn_tile_ring + dyn_slot, tag);
                        dyn_ready_barriers[dyn_slot]->arrive();
                    }
                    tag = __shfl_sync(0xffffffff, tag, 0);
                } else {
                    dyn_ready_barriers[dyn_slot]->wait(dyn_phase);
                    tag = ptx::ld_shared(smem_expert_cache->dyn_tile_ring + dyn_slot);
                    if (lane_idx == 0)
                        dyn_free_barriers[dyn_slot]->arrive();
                }
                advance_dyn_ring();
                if (tag == 0xffffffffu)
                    break;
                if ((tag & 0x80000000u) == 0u) {
                    dispatch_interleaved_l1(tag);
                } else {
                    dispatch_interleaved_l2_direct_tile(tag & 0x7fffffffu);
                }
            }
            return;
        }

        if constexpr (kAgyInterleaveMode == 4 and kAgyHotColdMode == 0) {
            // =========================================================================
            // Mode 4: Expert-Per-Wave (kNumExpertsPerWave = epw) Interleaved L1/L2 atomicAdd!
            // Virtual task stream:
            //   [Wave 0: L1(epw)] -> [Wave 0: Ready-L2(epw)] ->
            //   [Wave 1: L1(epw)] -> [Wave 1: Ready-L2(epw)] -> ...
            // Single lock-free atomicAdd cursor with ZERO grid_sync hard boundaries!
            // =========================================================================
            const uint32_t total_vtasks = total_l1_tiles + total_l2_tiles;
            uint32_t first_wave_l1 = 0u;
            #pragma unroll 1
            for (uint32_t w = 0u; w < kNumExpertWaves; ++ w) {
                const uint32_t e_lo = w * kNumExpertsPerWave;
                const uint32_t e_hi = cute::min(kNumExpertsPerRank, e_lo + kNumExpertsPerWave);
                const uint32_t wl1 = smem_expert_cache->l1_tile_prefix[e_hi] - smem_expert_cache->l1_tile_prefix[e_lo];
                if (wl1 > 0u) {
                    first_wave_l1 = wl1;
                    break;
                }
            }
            const uint32_t wave0_direct_l1 = cute::min(kNumSMs, first_wave_l1);
            if (sm_idx < wave0_direct_l1) {
                dispatch_interleaved_l1(sm_idx);
            }
            if (total_vtasks <= wave0_direct_l1)
                return;

            while (true) {
                uint32_t tag = 0xffffffffu;
                uint32_t packed_aux = 0u;
                if (role_id == 1) {
                    dyn_free_barriers[dyn_slot]->wait(dyn_phase ^ 1u);
                    if (lane_idx == 0) {
                        const uint32_t vtask = wave0_direct_l1 + ptx::atomic_add(l1_cursor_ptr, 1u);
                        if (vtask < total_vtasks) {
                            uint32_t w_lo = 0u, w_hi = kNumExpertWaves;
                            while (w_hi - w_lo > 1u) {
                                const uint32_t w_mid = (w_lo + w_hi) >> 1;
                                const uint32_t e_mid = w_mid * kNumExpertsPerWave;
                                const uint32_t v_mid =
                                    smem_expert_cache->l1_tile_prefix[e_mid] +
                                    smem_expert_cache->l2_tile_prefix[e_mid];
                                if (v_mid <= vtask) w_lo = w_mid;
                                else w_hi = w_mid;
                            }
                            const uint32_t e_lo = w_lo * kNumExpertsPerWave;
                            const uint32_t e_hi = cute::min(kNumExpertsPerRank, e_lo + kNumExpertsPerWave);
                            const uint32_t v_base =
                                smem_expert_cache->l1_tile_prefix[e_lo] +
                                smem_expert_cache->l2_tile_prefix[e_lo];
                            const uint32_t in_wave = vtask - v_base;
                            const uint32_t w_l1 =
                                smem_expert_cache->l1_tile_prefix[e_hi] -
                                smem_expert_cache->l1_tile_prefix[e_lo];
                            if (in_wave < w_l1) {
                                tag = smem_expert_cache->l1_tile_prefix[e_lo] + in_wave;
                            } else {
                                const uint32_t in_wave_l2 = in_wave - w_l1;
                                const uint32_t wave_mb_base = smem_expert_cache->pool_block_offset[e_lo];
                                const uint32_t r_mb = wave_mb_base + (in_wave_l2 / kL2BlockNs);
                                do {
                                    packed_aux = ptx::ld_acq(interleave_ready_mb_queue + r_mb);
                                } while (packed_aux == 0u);
                                tag = 0x80000000u | in_wave_l2;
                            }
                        } else {
                            tag = 0xffffffffu;
                        }
                        ptx::st_shared(smem_expert_cache->dyn_tile_ring + dyn_slot, tag);
                        ptx::st_shared(smem_expert_cache->dyn_aux_ring + dyn_slot, packed_aux);
                        dyn_ready_barriers[dyn_slot]->arrive();
                    }
                    tag = __shfl_sync(0xffffffff, tag, 0);
                    packed_aux = __shfl_sync(0xffffffff, packed_aux, 0);
                } else {
                    dyn_ready_barriers[dyn_slot]->wait(dyn_phase);
                    tag = ptx::ld_shared(smem_expert_cache->dyn_tile_ring + dyn_slot);
                    packed_aux = ptx::ld_shared(smem_expert_cache->dyn_aux_ring + dyn_slot);
                    if (lane_idx == 0)
                        dyn_free_barriers[dyn_slot]->arrive();
                }
                advance_dyn_ring();
                if (tag == 0xffffffffu)
                    break;
                if ((tag & 0x80000000u) == 0u) {
                    dispatch_interleaved_l1(tag);
                } else {
                    dispatch_interleaved_l2(tag & 0x7fffffffu, packed_aux);
                }
            }
            return;
        }

        // Modes 1..3: Global Ready-L2 / Balanced / M-Block Interleaved atomicAdd
        if (sm_idx < total_l1_tiles) {
            dispatch_interleaved_l1(sm_idx);
        }

        while (true) {
            uint32_t tag = 0xffffffffu;
            uint32_t packed_aux = 0u;
            if (role_id == 1) {
                dyn_free_barriers[dyn_slot]->wait(dyn_phase ^ 1u);
                if (lane_idx == 0) {
                    auto try_claim_ready_l2 = [&]() -> bool {
                        uint32_t cur_l2 = ptx::ld_volatile(l2_cursor_ptr);
                        const uint32_t avail_l2 = ptx::ld_acq(ready_l2_avail_ptr);
                        while (cur_l2 < avail_l2) {
                            const uint32_t old = atomicCAS(l2_cursor_ptr, cur_l2, cur_l2 + 1u);
                            if (old == cur_l2) {
                                tag = 0x80000000u | cur_l2;
                                return true;
                            }
                            cur_l2 = old;
                        }
                        return false;
                    };
                    auto try_claim_l1 = [&]() -> bool {
                        if (total_l1_tiles <= kNumSMs)
                            return false;
                        if (kNumSMs + ptx::ld_volatile(l1_cursor_ptr) >= total_l1_tiles)
                            return false;
                        const uint32_t my_l1 = kNumSMs + ptx::atomic_add(l1_cursor_ptr, 1u);
                        if (my_l1 < total_l1_tiles) {
                            tag = my_l1;
                            return true;
                        }
                        return false;
                    };

                    constexpr bool kBalancedSplit = (kAgyInterleaveMode == 2 or kAgyInterleaveMode == 3);
                    const bool prefer_l2 = kBalancedSplit ? ((sm_idx & 1u) == 0u) : true;
                    bool claimed = false;
                    if (prefer_l2) {
                        claimed = try_claim_ready_l2();
                        if (not claimed) claimed = try_claim_l1();
                    } else {
                        claimed = try_claim_l1();
                        if (not claimed) claimed = try_claim_ready_l2();
                    }

                    if (not claimed) {
                        // All L1 tiles have now been claimed across the grid; claim remaining L2 tile via atomicAdd!
                        const uint32_t my_l2 = ptx::atomic_add(l2_cursor_ptr, 1u);
                        if (my_l2 < total_l2_tiles) {
                            tag = 0x80000000u | my_l2;
                        } else {
                            tag = 0xffffffffu;
                        }
                    }

                    if (tag != 0xffffffffu and (tag & 0x80000000u) != 0u) {
                        const uint32_t r_mb = (tag & 0x7fffffffu) / kL2BlockNs;
                        do {
                            packed_aux = ptx::ld_acq(interleave_ready_mb_queue + r_mb);
                        } while (packed_aux == 0u);
                    }
                    ptx::st_shared(smem_expert_cache->dyn_tile_ring + dyn_slot, tag);
                    ptx::st_shared(smem_expert_cache->dyn_aux_ring + dyn_slot, packed_aux);
                    dyn_ready_barriers[dyn_slot]->arrive();
                }
                tag = __shfl_sync(0xffffffff, tag, 0);
                packed_aux = __shfl_sync(0xffffffff, packed_aux, 0);
            } else {
                dyn_ready_barriers[dyn_slot]->wait(dyn_phase);
                tag = ptx::ld_shared(smem_expert_cache->dyn_tile_ring + dyn_slot);
                packed_aux = ptx::ld_shared(smem_expert_cache->dyn_aux_ring + dyn_slot);
                if (lane_idx == 0)
                    dyn_free_barriers[dyn_slot]->arrive();
            }
            advance_dyn_ring();
            if (tag == 0xffffffffu)
                break;
            if ((tag & 0x80000000u) == 0u) {
                dispatch_interleaved_l1(tag);
            } else {
                dispatch_interleaved_l2(tag & 0x7fffffffu, packed_aux);
            }
        }
    };

    const auto for_each_selected_block_role = [&](const int role_id, const uint32_t& fused_tid, auto&& func) {
        if constexpr (MegaMoEPhase::is_fused) {
            if constexpr (kAgyInterleaveActive) {
                run_interleaved_l1_l2_blocks(role_id, fused_tid, func);
            } else {
                AGY_PROF_ROLE_PHASE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, true);
                run_single_phase_blocks(true, role_id, func);
                AGY_PROF_ROLE_PHASE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, true);
                AGY_PROF_ROLE_SYNC_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid);
                comm::grid_sync<
                    kNumSMs, kFusedGridSyncIndex,
                    comm::BarrierTimeoutPolicy::TrapOnly>(
                    workspace, sm_idx, fused_tid,
                    [=]() { ptx::sync_unaligned(64u + kNumEpilogueThreads, kFusedPhaseBarrierIdx); }
                );
                if (role_id == 0) {
                    if (cute::elect_one_sync()) {
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    __syncwarp();
                }
                AGY_PROF_ROLE_SYNC_END(MegaMoEPhase, sm_idx, role_id, fused_tid);
                AGY_PROF_ROLE_PHASE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
                run_single_phase_blocks(false, role_id, func);
                AGY_PROF_ROLE_PHASE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
            }
        } else if constexpr (MegaMoEPhase::runs_linear1) {
            AGY_PROF_ROLE_PHASE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, true);
            run_single_phase_blocks(true, role_id, func);
            AGY_PROF_ROLE_PHASE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, true);
        } else {
            AGY_PROF_ROLE_PHASE_BEGIN(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
            run_single_phase_blocks(false, role_id, func);
            AGY_PROF_ROLE_PHASE_END(MegaMoEPhase, sm_idx, role_id, fused_tid, false);
        }
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if constexpr (kAgyDedupDisp) {
            constexpr uint32_t kDedupWords = kNumRanks * kNumMaxTokensPerRank;
            auto* dedup_base = workspace.get_l1_empty_count_ptr(0);
            auto* dedup_vec = reinterpret_cast<uint4*>(dedup_base);
            const uint32_t gtid = sm_idx * kNumDispatchThreads + thread_idx;
            const uint32_t gstride = kNumSMs * kNumDispatchThreads;
            for (uint32_t q = gtid; q < kDedupWords / 4u; q += gstride) {
                uint4 cur;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(cur.x), "=r"(cur.y), "=r"(cur.z), "=r"(cur.w)
                             : "l"(dedup_vec + q));
                if ((cur.x | cur.y | cur.z | cur.w) != 0u) {
                    dedup_vec[q] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            if constexpr ((kDedupWords % 4u) != 0u) {
                for (uint32_t idx = (kDedupWords & ~3u) + gtid; idx < kDedupWords; idx += gstride) {
                    uint32_t w = 0u;
                    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(w) : "l"(dedup_base + idx));
                    if (w != 0u) {
                        dedup_base[idx] = 0u;
                    }
                }
            }
        }
        if (sm_idx == 0) {
            if constexpr (kAgyDynSched) {
                if (thread_idx == 0) {
                    if constexpr (kAgyFastClean) {
                        uint32_t* l1_tc = workspace.get_l1_task_count_ptr();
                        uint32_t* l2_tc = workspace.get_l2_task_count_ptr();
                        if (*l1_tc != 0u) *l1_tc = 0u;
                        if (*l2_tc != 0u) *l2_tc = 0u;
                    } else {
                        *workspace.get_l1_task_count_ptr() = 0u;
                        *workspace.get_l2_task_count_ptr() = 0u;
                    }
                    if constexpr (kAgyInterleaveActive) {
                        *workspace.get_shared_l1_task_count_ptr() = 0u;
                        *workspace.get_shared_l2_task_count_ptr() = 0u;
                    }
                }
                if constexpr (kAgyInterleaveActive and kAgyInterleaveMode == 4) {
                    for (uint32_t w = thread_idx; w < kNumExpertWaves; w += kNumDispatchThreads) {
                        interleave_wave_cursor_base[w] = 0u;
                    }
                }
            }
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else if constexpr (kAgyFastClean) {
            DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                uint32_t num_recv_tokens = 0u;
                if constexpr (kAgyUseExpertCache) {
                    num_recv_tokens = smem_expert_cache->recv_tokens[i];
                } else {
                    num_recv_tokens = scheduler.get_num_tokens(i);
                }

                // Always clear expert_recv_count_sum_ptr(i) because upper 32 bits hold (kNumSMs * kNumRanks).
                if (warp_idx == 0 and lane_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                }

                // Skip pool offset reduction, stats atomic, and l1_full_count loop when expert received 0 tokens.
                if (num_recv_tokens > 0u) {
                    if constexpr (MegaMoEPhase::one_warp_cleanup) {
                        if (warp_idx == 0) {
                            uint32_t num_recv_m_blocks = 0u;
                            uint32_t cleanup_pool_block_offset = 0u;
                            if constexpr (kAgyUseExpertCache) {
                                num_recv_m_blocks = smem_expert_cache->m_blocks[i];
                                cleanup_pool_block_offset = smem_expert_cache->pool_block_offset[i];
                            } else {
                                num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                                cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);
                            }
                            if (lane_idx == 0 and cumulative_local_expert_recv_stats != nullptr) {
                                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                            }
                            if constexpr (kAgyInterleaveActive and kAgyInterleaveMode >= 1 and kAgyInterleaveMode <= 4 and kAgyInterleaveMode != 3) {
                                if (lane_idx == 0) interleave_l1_done_base[i] = 0u;
                            }
                            for (uint32_t j = lane_idx; j < num_recv_m_blocks; j += 32u) {
                                *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                                if constexpr (kAgyInterleaveActive) {
                                    interleave_ready_mb_queue[cleanup_pool_block_offset + j] = 0u;
                                    if constexpr (kAgyInterleaveMode == 3 or ((kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) and (2u * L1_OUT_BLOCK_N == BLOCK_K))) {
                                        interleave_l1_done_base[cleanup_pool_block_offset + j] = 0u;
                                    }
                                }
                            }
                            __syncwarp();
                        }
                    } else {
                        uint32_t num_recv_m_blocks = 0u;
                        uint32_t cleanup_pool_block_offset = 0u;
                        if constexpr (kAgyUseExpertCache) {
                            num_recv_m_blocks = smem_expert_cache->m_blocks[i];
                            cleanup_pool_block_offset = smem_expert_cache->pool_block_offset[i];
                        } else {
                            num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                            cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);
                        }
                        if (warp_idx == 1) {
                            if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr) {
                                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                            }
                            __syncwarp();
                        }
                        if constexpr (kAgyInterleaveActive and kAgyInterleaveMode >= 1 and kAgyInterleaveMode <= 4 and kAgyInterleaveMode != 3) {
                            if (thread_idx == 0) interleave_l1_done_base[i] = 0u;
                        }
                        for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                            *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                            if constexpr (kAgyInterleaveActive) {
                                interleave_ready_mb_queue[cleanup_pool_block_offset + j] = 0u;
                                if constexpr (kAgyInterleaveMode == 3 or ((kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) and (2u * L1_OUT_BLOCK_N == BLOCK_K))) {
                                    interleave_l1_done_base[cleanup_pool_block_offset + j] = 0u;
                                }
                            }
                        }
                        __syncwarp();
                    }
                }
            }
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                if constexpr (!MegaMoEPhase::one_warp_cleanup)
                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if constexpr (MegaMoEPhase::one_warp_cleanup) {
                    if (warp_idx == 0) {
                        if (lane_idx == 0) {
                            *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                            if (cumulative_local_expert_recv_stats != nullptr)
                                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        }
                    }
                } else {
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }
                }

                if constexpr (MegaMoEPhase::one_warp_cleanup) {
                    if (warp_idx == 0) {
                        if constexpr (kAgyInterleaveActive and kAgyInterleaveMode >= 1 and kAgyInterleaveMode <= 4 and kAgyInterleaveMode != 3) {
                            if (lane_idx == 0 and num_recv_tokens > 0u) interleave_l1_done_base[i] = 0u;
                        }
                        for (uint32_t j = lane_idx; j < num_recv_m_blocks; j += 32) {
                            *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                            if constexpr (kAgyInterleaveActive) {
                                interleave_ready_mb_queue[cleanup_pool_block_offset + j] = 0u;
                                if constexpr (kAgyInterleaveMode == 3 or ((kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) and (2u * L1_OUT_BLOCK_N == BLOCK_K))) {
                                    interleave_l1_done_base[cleanup_pool_block_offset + j] = 0u;
                                }
                            }
                        }
                        __syncwarp();
                    }
                } else {
                    if constexpr (kAgyInterleaveActive and kAgyInterleaveMode >= 1 and kAgyInterleaveMode <= 4 and kAgyInterleaveMode != 3) {
                        if (thread_idx == 0 and num_recv_tokens > 0u) interleave_l1_done_base[i] = 0u;
                    }
                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                        if constexpr (kAgyInterleaveActive) {
                            interleave_ready_mb_queue[cleanup_pool_block_offset + j] = 0u;
                            if constexpr (kAgyInterleaveMode == 3 or ((kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) and (2u * L1_OUT_BLOCK_N == BLOCK_K))) {
                                interleave_l1_done_base[cleanup_pool_block_offset + j] = 0u;
                            }
                        }
                    }
                    __syncwarp();
                }
            }
        }
    };

    constexpr bool kSwizzleScatL2 = (not kSwapABActive) and ((WG_BLOCK_N % 64u) == 0u);
    const auto get_scat_l2_elem_idx = [](const uint32_t& wg_idx,
                                         const uint32_t& row_in_wg,
                                         const uint32_t& col_in_wg) -> uint32_t {
        if constexpr (kSwizzleScatL2) {
            const uint32_t chunk = col_in_wg >> 3;
            const uint32_t rem   = col_in_wg & 7u;
            return wg_idx * WG_BLOCK_M * WG_BLOCK_N +
                   row_in_wg * WG_BLOCK_N +
                   ((chunk ^ (row_in_wg & 7u)) << 3) + rem;
        } else {
            return wg_idx * WG_BLOCK_M * WG_BLOCK_N +
                   row_in_wg * WG_BLOCK_N + col_in_wg;
        }
    };

    const auto run_dispatch_l2_scatter_loop = [&]() {
        if constexpr (kScatDispActive) {
            constexpr uint32_t kRowBytesPerLane = (BLOCK_N * kCombineElementBytes) / 32u;
            constexpr uint32_t kRowElemsPerLane = BLOCK_N / 32u;
            DG_STATIC_ASSERT(kRowBytesPerLane == 4u or kRowBytesPerLane == 8u or
                             kRowBytesPerLane == 16u or kRowBytesPerLane == 32u,
                             "Unexpected BLOCK_N scatter width per lane");
            constexpr uint32_t kRegBatch =
                (kRowBytesPerLane == 32u) ? (kNumDispatchRegisters <= 32u ? 1u : 2u) :
                ((kRowBytesPerLane == 16u) ? (kNumDispatchRegisters <= 32u ? 2u : 4u) : 4u);
            DG_STATIC_ASSERT(kRegBatch <= 32u, "kRegBatch must fit in warp lanes");

            uint32_t disp_scat_phase = 0u;
            if (lane_idx == 0) {
                sm90_fp8_mega_moe_scat_bar_arrive(scat_free_bar);
            }
            __syncwarp();

            while (true) {
                sm90_fp8_mega_moe_scat_bar_wait(scat_ready_bar, disp_scat_phase);
                const uint4 meta = ptx::ld_shared(scat_meta_ptr);
                if (meta.w != 0u) {
                    break;
                }
                const uint32_t tile_m_idx     = meta.x;
                const uint32_t tile_n_blk_idx = meta.y;
                const uint32_t tile_valid_m   = meta.z;
                const uint32_t base_col_bytes = tile_n_blk_idx * BLOCK_N * kCombineElementBytes;
                AGY_PROF_DISP_SCAT_BEGIN(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps,
                                         tile_m_idx / BLOCK_M, tile_n_blk_idx);

                bool arrived_free = false;
                for (uint32_t r_base = warp_idx;
                     r_base < tile_valid_m;
                     r_base += kRegBatch * kNumDispatchWarps) {
                    const bool is_last_batch =
                        (r_base + kRegBatch * kNumDispatchWarps) >= tile_valid_m;

                    uint64_t my_row_dst_addr = 0ull;
                    if (lane_idx < kRegBatch) {
                        const uint32_t r_meta = r_base + lane_idx * kNumDispatchWarps;
                        if (r_meta < tile_valid_m) {
                            const auto src_meta = *workspace.get_token_src_metadata_ptr(tile_m_idx + r_meta);
                            const uint32_t dst_topk = kAgyDedupScatReq ? (src_meta.topk_idx & 0xffu) : src_meta.topk_idx;
                            const auto dst_tok_buf = combine_token_buffer.get_rank_buffer(dst_topk)
                                                                         .get_data_buffer(src_meta.token_idx);
                            auto* dst_ptr = math::advance_ptr<uint8_t>(dst_tok_buf.get_base_ptr(), base_col_bytes);
                            my_row_dst_addr = reinterpret_cast<uint64_t>(sym_buffer.map(dst_ptr, src_meta.rank_idx));
                        }
                    }

                    if constexpr (kRowBytesPerLane == 16u) {
                        uint4 reg_vec[kRegBatch];
                        const uint32_t col = lane_idx * kRowElemsPerLane;
                        const uint32_t wg_n = col / WG_BLOCK_N;
                        const uint32_t col_in_wg = col % WG_BLOCK_N;
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r < tile_valid_m) {
                                const uint32_t wg_m = r / WG_BLOCK_M;
                                const uint32_t row_in_wg = r % WG_BLOCK_M;
                                const uint32_t wg = wg_m * kWarpgroupSplitN + wg_n;
                                const uint32_t idx = get_scat_l2_elem_idx(wg, row_in_wg, col_in_wg);
                                reg_vec[b] = ptx::ld_shared(reinterpret_cast<const uint4*>(smem_scat_l2 + idx));
                            }
                        }
                        if (is_last_batch) {
                            __syncwarp();
                            if (lane_idx == 0) {
                                sm90_fp8_mega_moe_scat_bar_arrive(scat_free_bar);
                            }
                            arrived_free = true;
                        }
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r >= tile_valid_m) break;
                            const uint64_t row_addr = __shfl_sync(0xffffffff, my_row_dst_addr, b);
                            *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(row_addr) + lane_idx * 16u) = reg_vec[b];
                        }
                    } else if constexpr (kRowBytesPerLane == 8u) {
                        uint2 reg_vec[kRegBatch];
                        const uint32_t col = lane_idx * kRowElemsPerLane;
                        const uint32_t wg_n = col / WG_BLOCK_N;
                        const uint32_t col_in_wg = col % WG_BLOCK_N;
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r < tile_valid_m) {
                                const uint32_t wg_m = r / WG_BLOCK_M;
                                const uint32_t row_in_wg = r % WG_BLOCK_M;
                                const uint32_t wg = wg_m * kWarpgroupSplitN + wg_n;
                                const uint32_t idx = get_scat_l2_elem_idx(wg, row_in_wg, col_in_wg);
                                uint2 v;
                                asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                                             : "=r"(v.x), "=r"(v.y)
                                             : "l"(__cvta_generic_to_shared(smem_scat_l2 + idx)));
                                reg_vec[b] = v;
                            }
                        }
                        if (is_last_batch) {
                            __syncwarp();
                            if (lane_idx == 0) {
                                sm90_fp8_mega_moe_scat_bar_arrive(scat_free_bar);
                            }
                            arrived_free = true;
                        }
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r >= tile_valid_m) break;
                            const uint64_t row_addr = __shfl_sync(0xffffffff, my_row_dst_addr, b);
                            *reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(row_addr) + lane_idx * 8u) = reg_vec[b];
                        }
                    } else if constexpr (kRowBytesPerLane == 32u) {
                        uint4 reg_vec0[kRegBatch];
                        uint4 reg_vec1[kRegBatch];
                        const uint32_t col0 = lane_idx * 8u;
                        const uint32_t col1 = 256u + lane_idx * 8u;
                        const uint32_t wg_n0 = col0 / WG_BLOCK_N, col_in_wg0 = col0 % WG_BLOCK_N;
                        const uint32_t wg_n1 = col1 / WG_BLOCK_N, col_in_wg1 = col1 % WG_BLOCK_N;
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r < tile_valid_m) {
                                const uint32_t wg_m = r / WG_BLOCK_M;
                                const uint32_t row_in_wg = r % WG_BLOCK_M;
                                const uint32_t idx0 = get_scat_l2_elem_idx(wg_m * kWarpgroupSplitN + wg_n0, row_in_wg, col_in_wg0);
                                const uint32_t idx1 = get_scat_l2_elem_idx(wg_m * kWarpgroupSplitN + wg_n1, row_in_wg, col_in_wg1);
                                reg_vec0[b] = ptx::ld_shared(reinterpret_cast<const uint4*>(smem_scat_l2 + idx0));
                                reg_vec1[b] = ptx::ld_shared(reinterpret_cast<const uint4*>(smem_scat_l2 + idx1));
                            }
                        }
                        if (is_last_batch) {
                            __syncwarp();
                            if (lane_idx == 0) {
                                sm90_fp8_mega_moe_scat_bar_arrive(scat_free_bar);
                            }
                            arrived_free = true;
                        }
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r >= tile_valid_m) break;
                            const uint64_t row_addr = __shfl_sync(0xffffffff, my_row_dst_addr, b);
                            auto* mapped = reinterpret_cast<uint8_t*>(row_addr) + lane_idx * 16u;
                            *reinterpret_cast<uint4*>(mapped) = reg_vec0[b];
                            *reinterpret_cast<uint4*>(mapped + 512u) = reg_vec1[b];
                        }
                    } else {
                        uint32_t reg_vec[kRegBatch];
                        const uint32_t col = lane_idx * 2u;
                        const uint32_t wg_n = col / WG_BLOCK_N;
                        const uint32_t col_in_wg = col % WG_BLOCK_N;
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r < tile_valid_m) {
                                const uint32_t wg_m = r / WG_BLOCK_M;
                                const uint32_t row_in_wg = r % WG_BLOCK_M;
                                const uint32_t wg = wg_m * kWarpgroupSplitN + wg_n;
                                const uint32_t idx = get_scat_l2_elem_idx(wg, row_in_wg, col_in_wg);
                                reg_vec[b] = ptx::ld_shared(reinterpret_cast<const uint32_t*>(smem_scat_l2 + idx));
                            }
                        }
                        if (is_last_batch) {
                            __syncwarp();
                            if (lane_idx == 0) {
                                sm90_fp8_mega_moe_scat_bar_arrive(scat_free_bar);
                            }
                            arrived_free = true;
                        }
                        #pragma unroll
                        for (uint32_t b = 0; b < kRegBatch; ++ b) {
                            const uint32_t r = r_base + b * kNumDispatchWarps;
                            if (r >= tile_valid_m) break;
                            const uint64_t row_addr = __shfl_sync(0xffffffff, my_row_dst_addr, b);
                            *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(row_addr) + lane_idx * 4u) = reg_vec[b];
                        }
                    }
                }
                if (not arrived_free) {
                    if (lane_idx == 0) {
                        sm90_fp8_mega_moe_scat_bar_arrive(scat_free_bar);
                    }
                }
                AGY_PROF_DISP_SCAT_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps,
                                       tile_m_idx / BLOCK_M, tile_n_blk_idx);
            }
        }
    };

    const auto run_all_idle_reg_combine = [&](const uint32_t& comb_warp_global_id,
                                              const uint32_t& comb_warp_global_stride) {
        if constexpr (kAgyAllIdleCombActive) {
            auto run_chunked = [&]<uint32_t kRunChunks>() {
                constexpr uint32_t kElemsPerVector = 8u;
                constexpr uint32_t kNumBF16PairsPerVector = 4u;
                constexpr uint32_t kRunChunkElems = kHidden / kRunChunks;
                constexpr uint32_t kRunInputBytes = kCombineInputHiddenBytes / kRunChunks;
                constexpr uint32_t kRunOutputBytes = kCombineOutputHiddenBytes / kRunChunks;
                constexpr uint32_t kRunVecsPerLane = kRunChunkElems / (32u * kElemsPerVector);
                static_assert(kRunVecsPerLane <= 2u, "All-idle register combine expects <= 2 uint4 per lane");
                const uint32_t total_items = num_tokens * kRunChunks;
                const bool y_aligned_16 = (reinterpret_cast<uintptr_t>(y) & 15u) == 0u;
                uint32_t cached_token = 0xffffffffu;
                uint32_t cached_mask = 0u;
                for (uint32_t work_id = comb_warp_global_id;
                     work_id < total_items;
                     work_id += comb_warp_global_stride) {
                    const uint32_t token_idx = work_id / kRunChunks;
                    const uint32_t chunk = work_id - token_idx * kRunChunks;
                    if (token_idx != cached_token) {
                        cached_token = token_idx;
                        const int stored_slot = lane_idx < kNumTopk
                            ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                                     token_idx * kNumTopk + lane_idx))
                            : -1;
                        cached_mask = __ballot_sync(0xffffffff, stored_slot >= 0);
                    }
                    const uint32_t input_off = chunk * kRunInputBytes;
                    const uint32_t output_off = chunk * kRunOutputBytes;
                    uint32_t mask = cached_mask;
                    float2 reduced[kRunVecsPerLane * kNumBF16PairsPerVector] = {};
                    while (mask & (mask - 1u)) {
                        const uint32_t s0 = static_cast<uint32_t>(__ffs(mask) - 1);
                        mask &= (mask - 1u);
                        const uint32_t s1 = static_cast<uint32_t>(__ffs(mask) - 1);
                        mask &= (mask - 1u);
                        const auto* src0 = reinterpret_cast<const uint4*>(
                            math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(s0).get_data_buffer(token_idx).get_base_ptr(),
                                input_off));
                        const auto* src1 = reinterpret_cast<const uint4*>(
                            math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(s1).get_data_buffer(token_idx).get_base_ptr(),
                                input_off));
                        uint4 v0[kRunVecsPerLane], v1[kRunVecsPerLane];
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) v0[j] = ptx::ld_evict_first(src0 + j * 32u + lane_idx);
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) v1[j] = ptx::ld_evict_first(src1 + j * 32u + lane_idx);
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            const auto* bf0 = reinterpret_cast<const nv_bfloat162*>(&v0[j]);
                            const auto* bf1 = reinterpret_cast<const nv_bfloat162*>(&v1[j]);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l) {
                                ptx::accumulate(reduced[j * kNumBF16PairsPerVector + l], bf0[l]);
                                ptx::accumulate(reduced[j * kNumBF16PairsPerVector + l], bf1[l]);
                            }
                        }
                    }
                    if (mask) {
                        const uint32_t s0 = static_cast<uint32_t>(__ffs(mask) - 1);
                        const auto* src0 = reinterpret_cast<const uint4*>(
                            math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(s0).get_data_buffer(token_idx).get_base_ptr(),
                                input_off));
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            const uint4 v0 = ptx::ld_evict_first(src0 + j * 32u + lane_idx);
                            const auto* bf0 = reinterpret_cast<const nv_bfloat162*>(&v0);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                ptx::accumulate(reduced[j * kNumBF16PairsPerVector + l], bf0[l]);
                        }
                    }
                    auto* dst_y_ptr = math::advance_ptr<uint8_t>(
                        y, static_cast<uint64_t>(token_idx) * kCombineOutputHiddenBytes + output_off);
                    if (y_aligned_16) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            uint4 casted;
                            auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumBF16PairsPerVector + l]);
                            *reinterpret_cast<uint4*>(dst_y_ptr + (j * 32u + lane_idx) * 16u) = casted;
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            auto* dst_bf16 = reinterpret_cast<nv_bfloat16*>(dst_y_ptr + (j * 32u + lane_idx) * 16u);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l) {
                                const float2 r_pair = reduced[j * kNumBF16PairsPerVector + l];
                                dst_bf16[2u * l + 0u] = __float2bfloat16_rn(r_pair.x);
                                dst_bf16[2u * l + 1u] = __float2bfloat16_rn(r_pair.y);
                            }
                        }
                    }
                }
            };
            constexpr uint32_t kFineChunks = kHidden / 256u;
            constexpr uint32_t kDualChunks = kHidden / 512u;
            if (num_tokens * kFineChunks <= comb_warp_global_stride) {
                run_chunked.template operator()<kFineChunks>();
            } else {
                run_chunked.template operator()<kDualChunks>();
            }
        }
    };

    // ROLE 1: DISPATCH WARPS
    if constexpr (kRoleMode <= 1) {
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        AGY_PROF_DISP_OPEN(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);

        if constexpr (MegaMoEPhase::runs_linear2 and not MegaMoEPhase::runs_linear1) {
            AGY_PROF_SPLIT_L2_PROLOGUE_BEGIN(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
            if constexpr (kScatDispActive) {
                if constexpr (kAgyUseExpertCache) {
                    populate_smem_expert_cache();
                    ptx::sync_unaligned(kPreGemmSyncThreads, kDispatchWithEpilogueBarrierIdx);
                } else {
                    ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
                }
                AGY_PROF_SPLIT_L2_PROLOGUE_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
                run_dispatch_l2_scatter_loop();
                if constexpr (kAgyUseExpertCache) {
                    load_scheduler_from_expert_cache();
                } else {
                    scheduler.fetch_expert_recv_count();
                }
                ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            } else {
                if constexpr (kAgyUseExpertCache) {
                    populate_smem_expert_cache();
                    ptx::sync_unaligned(kPreGemmSyncThreads, kDispatchWithEpilogueBarrierIdx);
                    if constexpr (not kAgyFastClean) {
                        load_scheduler_from_expert_cache();
                    }
                } else {
                    scheduler.fetch_expert_recv_count();
                    ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
                }
                AGY_PROF_SPLIT_L2_PROLOGUE_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
            }
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            AGY_PROF_DISP_CLEANUP_BEGIN(MegaMoEPhase, sm_idx, warp_idx, lane_idx, true);
            cleanup_workspace();
            AGY_PROF_DISP_CLEANUP_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag,
                comm::BarrierTimeoutPolicy::TrapOnly>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);
            AGY_PROF_DISP_BAR3_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
            AGY_PROF_DISP_CLOSE(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);
            return;
        }
        AGY_PROF_NOTIFY_BEGIN(MegaMoEPhase, sm_idx, warp_idx, lane_idx);

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const uint32_t active_disp_sms = kAgyCompactNotify
            ? cute::min(kNumSMs, cute::max(1u, math::ceil_div(num_tokens, kNumDispatchWarps * kNumTokensPerWarp)))
            : kNumSMs;

        if (not kAgyCompactNotify or sm_idx < active_disp_sms) {
            // Count tokens per expert
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    const int expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        atomicAdd_block(smem_expert_count + expert_idx, 1);
                }
                __syncwarp();
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Stake out per-expert SM offsets via global atomic
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
                smem_expert_count[i] = static_cast<uint32_t>(
                    ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Write source token-topk indices (and optional pushed weights / dedup_scat metadata) to remote ranks
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                const bool in_range = (i + (lane_idx / kNumTopk) < num_tokens) and (lane_idx < kNumActivateLanes);
                const int expert_idx = in_range ? static_cast<int>(
                    __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx)) : -1;
                const bool valid_route = expert_idx >= 0;
                uint32_t active_mask = 0u;
                if constexpr (kAgyDedupScatReq) {
                    active_mask = __ballot_sync(0xffffffff, valid_route);
                }
                if (valid_route) {
                    const uint32_t token_topk_idx = i * kNumTopk + lane_idx;
                    const uint32_t dst_rank_idx = static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
                    const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                        static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                    uint32_t packed_send = token_topk_idx;
                    if constexpr (kAgyDedupScatReq) {
                        const uint32_t group_id = lane_idx / kNumTopk;
                        const uint32_t peer_mask = __match_any_sync(active_mask, (group_id << 8) | dst_rank_idx);
                        const uint32_t dup_count = static_cast<uint32_t>(__popc(peer_mask));
                        const uint32_t leader_topk = static_cast<uint32_t>(__ffs(peer_mask) - 1) - group_id * kNumTopk;
                        packed_send = (dup_count << 26) | (leader_topk << 21) | token_topk_idx;
                    }
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed_send;
                    if constexpr (kAgyFastPull) {
                        const uint32_t weight_bits = __float_as_uint(
                            __ldg(input_topk_weights_buffer.get_base_ptr<float>() + token_topk_idx));
                        *sym_buffer.map(dst_ptr + kNumMaxTokensPerRank, dst_rank_idx) = weight_bits;
                    }
                }
                __syncwarp();
            }
        }

        uint32_t active_d_warps = kNumDispatchWarps;
        uint32_t hetero_pull_start = sm_idx * kNumDispatchWarps + warp_idx;
        uint32_t hetero_pull_stride = kNumSMs * kNumDispatchWarps;
        uint32_t* hetero_cnt_ptr = nullptr;
        if constexpr (kAgyHeteroD8D4) {
            uint32_t hw_smid = sm_idx;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(hw_smid));
            const uint32_t norm_smid = (hw_smid < kNumSMs) ? hw_smid : (hw_smid % kNumSMs);
            if constexpr (kNumDispatchWarps == 6u) {
                if constexpr (kAgyDMaskMode == 1) {
                    active_d_warps = ((hw_smid & 1u) == 0u) ? 6u : 2u;
                } else if constexpr (kAgyDMaskMode == 2) {
                    active_d_warps = (norm_smid < (kNumSMs / 2u)) ? 6u : 2u;
                } else if constexpr (kAgyDMaskMode == 3) {
                    active_d_warps = (norm_smid < (kNumSMs / 2u)) ? 6u : 0u;
                } else if constexpr (kAgyDMaskMode == 4) {
                    active_d_warps = ((hw_smid & 1u) == 0u) ? 6u : 0u;
                } else if constexpr (kAgyDMaskMode == 5) {
                    active_d_warps = ((hw_smid & 3u) != 3u) ? 6u : 2u;
                }
            } else if constexpr (kNumDispatchWarps == 2u) {
                if constexpr (kAgyDMaskMode == 1) {
                    active_d_warps = ((hw_smid & 1u) == 0u) ? 2u : 0u;
                } else if constexpr (kAgyDMaskMode == 2) {
                    active_d_warps = (norm_smid < (kNumSMs / 2u)) ? 2u : 0u;
                } else if constexpr (kAgyDMaskMode == 3) {
                    active_d_warps = ((hw_smid & 3u) == 0u) ? 2u : 0u;
                } else if constexpr (kAgyDMaskMode == 4) {
                    active_d_warps = ((hw_smid & 3u) != 3u) ? 2u : 0u;
                } else if constexpr (kAgyDMaskMode == 5) {
                    active_d_warps = (norm_smid >= (kNumSMs / 2u)) ? 2u : 0u;
                }
            } else {
                const bool is_d8_sm = (kAgyDMaskMode == 1) ? ((hw_smid & 1u) == 0u) : (norm_smid < (kNumSMs / 2u));
                active_d_warps = is_d8_sm ? 8u : 4u;
            }
            // Isolated slot in Workspace 128B header padding [48..51] --zero overlap with ring buffers!
            hetero_cnt_ptr = math::advance_ptr<uint32_t>(workspace.base, 48u);
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            if (warp_idx == 0 and lane_idx == 0) {
                const uint32_t w_base = (active_d_warps > 0u)
                    ? ptx::atomic_add_rel(hetero_cnt_ptr, active_d_warps)
                    : 0u;
                ptx::st_shared(smem_expert_count + 0, w_base);
            }
        }

        comm::grid_sync<
            kNumSMs, kDispatchGridSyncIndex,
            comm::BarrierTimeoutPolicy::TrapOnly>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if constexpr (kAgyHeteroD8D4) {
            if (warp_idx == 0 and lane_idx == 0) {
                ptx::st_shared(smem_expert_count + 1, ptx::ld_acq(hetero_cnt_ptr));
            }
        }

        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                const uint64_t local_cnt = expert_status & 0xffffffffull;
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = local_cnt;
                if constexpr (kAgyCompactNotify) {
                    const uint64_t send_packet = (static_cast<uint64_t>(kNumSMs) << 32) | local_cnt;
                    auto* dst_sum_ptr = sym_buffer.map(
                        workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx);
                    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(dst_sum_ptr), "l"(send_packet) : "memory");
                } else {
                    ptx::atomic_add_sys(
                        sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                        expert_status);
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        if constexpr (kAgyHeteroD8D4) {
            hetero_pull_start = ptx::ld_shared(smem_expert_count + 0) + warp_idx;
            hetero_pull_stride = ptx::ld_shared(smem_expert_count + 1);
        }

        AGY_PROF_BEFORE_BAR1(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumDispatchThreads,
            kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag,
            comm::BarrierTimeoutPolicy::TrapOnly>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);
        AGY_PROF_AFTER_BAR1(MegaMoEPhase, sm_idx, warp_idx, lane_idx);

        // CTA 0 evaluates Eq.(1)-(3) on the live post-dispatch histogram and
        // publishes one decision.  The decision lives in global workspace:
        // the pre-barrier SMEM region below is recycled by scatter/combine, so
        // aliasing smem_expert_count would silently destroy the policy.
        if (thread_idx == 0u) {
            const uint32_t weave_epoch = weave_epoch_base + 1u;
            if (sm_idx == 0u) {
                const weave::WorkloadSpec wl{
                    kNumExpertsPerRank,
                    static_cast<uint32_t>(kHidden),
                    static_cast<uint32_t>(kIntermediateHidden)};
                const weave::WeaveKnobs knobs =
                    weave::default_weave_knobs(weave::HardwareArch::H20_SM90);
                const uint32_t m_out = weave::count_remote_expert_sends(
                    workspace, kNumExperts, kNumExpertsPerRank,
                    static_cast<uint32_t>(sym_buffer.rank_idx));
                (void)weave::cta0_select_from_packed_hist(
                    weave_control, weave::HardwareArch::H20_SM90, wl, knobs,
                    workspace.get_expert_recv_count_sum_ptr(),
                    0u, m_out, weave_epoch);
            } else {
                weave::PolicyDecision discarded{};
                (void)weave::wait_and_copy_policy(
                    weave_control, weave_epoch, &discarded);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        if constexpr (kAgyHeteroD8D4) {
            if (sm_idx == 0 and warp_idx == 0 and lane_idx == 0) {
                *hetero_cnt_ptr = 0u;
            }
        }

        if constexpr (kAgyUseExpertCache) {
            populate_smem_expert_cache();
#if defined(AGY_DEDUP_STATS) && AGY_DEDUP_STATS
            if (sm_idx == 0 and warp_idx == 0) {
                uint32_t m_sum = 0u, active_e = 0u;
                uint32_t pad16 = 0u, pad32 = 0u, pad64 = 0u;
                for (uint32_t base_e = 0u; base_e < kNumExpertsPerRank; base_e += 32u) {
                    const uint32_t e = base_e + lane_idx;
                    const uint32_t c = (e < kNumExpertsPerRank) ? smem_expert_cache->recv_tokens[e] : 0u;
                    if (c > 0u) {
                        m_sum += c;
                        active_e += 1u;
                        pad16 += ((c + 15u) & ~15u);
                        pad32 += ((c + 31u) & ~31u);
                        pad64 += ((c + 63u) & ~63u);
                    }
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    m_sum    += __shfl_xor_sync(0xffffffff, m_sum, off);
                    active_e += __shfl_xor_sync(0xffffffff, active_e, off);
                    pad16    += __shfl_xor_sync(0xffffffff, pad16, off);
                    pad32    += __shfl_xor_sync(0xffffffff, pad32, off);
                    pad64    += __shfl_xor_sync(0xffffffff, pad64, off);
                }
                float cand_pred = 1e30f;
                if (lane_idx < 8u) {
                    constexpr uint32_t kCandBm[8]     = {16u, 16u, 16u, 32u, 32u, 64u, 64u, 64u};
                    constexpr float kCandNc[8]        = {156.f, 156.f, 78.f, 156.f, 156.f, 312.f, 312.f, 468.f};
                    constexpr float kCandNe[8]        = {1248.f, 1248.f, 1248.f, 1248.f, 1248.f, 624.f, 624.f, 624.f};
                    constexpr float kCandNeSm[8]      = {16.f, 16.f, 16.f, 16.f, 16.f, 8.f, 8.f, 8.f};
                    constexpr float kCandBmem[8]      = {3350.f, 3350.f, 3350.f, 3350.f, 3350.f, 2050.0f, 2050.0f, 2050.0f};
                    constexpr float kCandPmma[8]      = {136.3f, 136.3f, 143.1f, 203.4f, 203.4f, 242.0f, 242.0f, 240.0f};
                    constexpr int   kCandComm[8]      = {0, 1, 0, 0, 1, 0, 1, 1};
                    constexpr float kCandOvh[8]       = {44.6f, 42.8f, 43.8f, 52.0f, 50.0f, 52.6f, 48.6f, 47.1f};
                    constexpr float kCandAlpha[8]     = {0.728f, 0.728f, 0.728f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f};

                    const uint32_t bm = kCandBm[lane_idx];
                    const uint32_t pm = (bm == 16u) ? pad16 : ((bm == 32u) ? pad32 : pad64);
                    const float extra_mb = static_cast<float>(pm / bm > active_e ? (pm / bm - active_e) : 0u);
                    const float b_in = static_cast<float>(kHidden + (kHidden >> 7) * 4u);
                    const float b_out = static_cast<float>(kHidden * 2u);
                    const float w_bytes = 3.0f * static_cast<float>(kHidden) * static_cast<float>(L1_SHAPE_N / 2u);
                    const float v_in = static_cast<float>(m_sum) * b_in;
                    const float v_out = static_cast<float>(m_sum) * b_out;
                    const float bw_pull = fminf(370.0f, fmaxf(20.0f, kCandNc[lane_idx] * 1.20f)) * 1e9f;
                    const float bw_scat_e = fminf(370.0f, kCandNe[lane_idx] * 0.14f) * 1e9f;
                    const float bw_scat_c = fminf(370.0f, fmaxf(20.0f, kCandNc[lane_idx] * 1.00f)) * 1e9f;
                    const float t_pull = (v_in / bw_pull) * 1e6f;
                    const float f_gemm = 6.0f * static_cast<float>(kHidden) * static_cast<float>(L1_SHAPE_N / 2u) * static_cast<float>(pm);
                    const float v_mem = (static_cast<float>(active_e) + kCandAlpha[lane_idx] * extra_mb) * w_bytes
                                      + static_cast<float>(pm) * static_cast<float>(kHidden + L1_SHAPE_N) * 2.0f;
                    const float t_mem = (v_mem / (kCandBmem[lane_idx] * 1e9f)) * 1e6f;
                    const float t_mma = (f_gemm / (kCandPmma[lane_idx] * 1e12f)) * 1e6f;
                    const float t_epi = kCandOvh[lane_idx] + (v_out / (kCandNeSm[lane_idx] * 420e9f)) * 1e6f;
                    const float t_comp = fmaxf(t_mem, t_mma) + t_epi;
                    const float t_first = fmaxf(1.5f, t_comp / fmaxf(1.0f, static_cast<float>(active_e) * 2.0f));
                    cand_pred = (kCandComm[lane_idx] == 0)
                        ? fmaxf(t_pull, t_comp + (v_out / bw_scat_e) * 1e6f)
                        : fmaxf(fmaxf(t_pull, t_first) + (v_out / bw_scat_c) * 1e6f, t_comp + 0.8f);
                }
                uint32_t best_pid = lane_idx;
                float best_val = cand_pred;
                #pragma unroll
                for (int off = 4; off > 0; off >>= 1) {
                    const float other_val = __shfl_xor_sync(0xffffffff, best_val, off);
                    const uint32_t other_pid = __shfl_xor_sync(0xffffffff, best_pid, off);
                    if (other_val < best_val) { best_val = other_val; best_pid = other_pid; }
                }
                asm volatile("" :: "r"(best_pid), "f"(best_val) : "memory");
            }
#endif
        }
        // Sync with TMA & epilogue warps before pulling tokens
        ptx::sync_unaligned(kPreGemmSyncThreads, kDispatchWithEpilogueBarrierIdx);

        if constexpr (kAgyHeteroD8D4 and not MegaMoEPhase::is_fused) {
            if (warp_idx >= active_d_warps) {
                AGY_PROF_DISP_CLOSE(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);
                return;
            }
        }

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        if constexpr (kAgyUseExpertCache) {
            load_scheduler_from_expert_cache();
        } else {
            scheduler.fetch_expert_recv_count();
        }

        AgyDedupDeferredEntry* my_dedup_fifo = smem_dedup_fifo_base + warp_idx * kDedupFifoCap;
        uint32_t fifo_head = 0u, fifo_tail = 0u, fifo_count = 0u;
        const auto copy_local_pool_token = [&](
            const uint32_t& dst_pool_idx, const uint32_t& src_pool_idx,
            const uint32_t& src_rank, const uint32_t& src_tok,
            const uint32_t& topk_meta, const uint32_t& w_bits,
            const uint32_t& m_blk_pool_idx) {
            const auto* src_vec = reinterpret_cast<const uint4*>(
                l1_token_buffer.get_data_buffer(src_pool_idx).get_base_ptr());
            auto* dst_vec = reinterpret_cast<uint4*>(
                l1_token_buffer.get_data_buffer(dst_pool_idx).get_base_ptr());
            #pragma unroll
            for (uint32_t v = lane_idx; v < kHidden / 16u; v += 32u) {
                dst_vec[v] = src_vec[v];
            }
            constexpr uint32_t kNumSFFloatsLocal = kHidden / 128u;
            auto* local_sf_ptr = l1_sf_buffer.get_base_ptr<float>();
            #pragma unroll
            for (uint32_t s = lane_idx; s < kNumSFFloatsLocal; s += 32u) {
                local_sf_ptr[s * kSFPoolStrideTokens + dst_pool_idx] =
                    local_sf_ptr[s * kSFPoolStrideTokens + src_pool_idx];
            }
            if (lane_idx == 0) {
                *l1_topk_weights_buffer.get_data_buffer(dst_pool_idx).get_base_ptr<float>() =
                    __uint_as_float(w_bits);
                *workspace.get_token_src_metadata_ptr(dst_pool_idx) =
                    {src_rank, src_tok, topk_meta};
            }
            __syncwarp();
            if (lane_idx == 0) {
                ptx::red_add_rel(workspace.get_l1_full_count_ptr(m_blk_pool_idx), 1u);
            }
            __syncwarp();
        };
        const auto drain_one_dedup_fifo = [&](const bool& block_until_ready) -> bool {
            if (fifo_count == 0u) return false;
            const AgyDedupDeferredEntry ent = my_dedup_fifo[fifo_head];
            uint32_t* dptr = workspace.get_l1_empty_count_ptr(0) +
                             ent.src_rank_idx * kNumMaxTokensPerRank + ent.src_token_idx;
            uint32_t ready = 0u;
            if (lane_idx == 0) {
                if (block_until_ready) {
                    while ((ptx::ld_acq(dptr) & 0x80000000u) == 0u) {}
                    ready = 1u;
                } else {
                    ready = (ptx::ld_acq(dptr) & 0x80000000u) ? 1u : 0u;
                }
            }
            ready = __shfl_sync(0xffffffff, ready, 0);
            if (not ready) return false;
            copy_local_pool_token(
                ent.pool_token_idx, ent.leader_pool_idx,
                ent.src_rank_idx, ent.src_token_idx,
                ent.packed_topk_meta, ent.weight_bits, ent.m_block_pool_idx);
            fifo_head = (fifo_head + 1u) & (kDedupFifoCap - 1u);
            -- fifo_count;
            return true;
        };

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Consume the CTA-0 decision: all warps pull the first V_head tasks,
        // then only the selected N_c communication warps per SM remain active.
        const weave::PolicyDecision weave_policy =
            weave::load_published_policy(weave_control);
        const bool weave_policy_ok = weave::policy_is_valid(weave_policy);
        uint32_t total_recv_tokens = 0u;
        #pragma unroll
        for (uint32_t e = 0u; e < kNumExpertsPerRank; ++e) {
            total_recv_tokens += scheduler.get_num_tokens(e);
        }
        const uint32_t policy_comm_warps = weave_policy_ok
            ? weave::r_comm_units(weave_policy.selected_r_mask)
            : kNumDispatchWarps;
        const uint32_t active_pull_warps = policy_comm_warps == 0u ? 1u :
            (policy_comm_warps < active_d_warps ? policy_comm_warps : active_d_warps);
        const uint32_t selected_v_head = weave_policy_ok
            ? weave_policy.selected_v_head : 0u;
        const uint32_t head_pull_end = selected_v_head == 0u
            ? total_recv_tokens
            : (selected_v_head < total_recv_tokens ? selected_v_head : total_recv_tokens);

        const uint32_t pull_token_start = kAgyHeteroD8D4
            ? hetero_pull_start
            : (sm_idx * kNumDispatchWarps + warp_idx);
        const uint32_t pull_token_stride = kAgyHeteroD8D4
            ? hetero_pull_stride
            : (kNumSMs * kNumDispatchWarps);
        if (warp_idx == 0u or ((not (kAgyHeteroD8D4 and MegaMoEPhase::is_fused) or warp_idx < active_d_warps) and pull_token_stride > 0u)) {
            AGY_PROF_BEFORE_PULL(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);
        }
        if ((not (kAgyHeteroD8D4 and MegaMoEPhase::is_fused) or warp_idx < active_d_warps) and pull_token_stride > 0u) {
        for (uint32_t token_idx = pull_token_start; ; token_idx += pull_token_stride) {
            if (token_idx >= head_pull_end and warp_idx >= active_pull_warps) break;
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    const uint32_t phys_r =
#if defined(AGY_DEDUP_TIER) && AGY_DEDUP_TIER
                        (j < kNumRanks) ? ((sym_buffer.rank_idx + j) % kNumRanks) : j;
#else
                        j;
#endif
                    if constexpr (kAgyCacheRankCounts) {
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(smem_expert_cache->get_rank_count(
                                static_cast<uint32_t>(current_expert_idx), phys_r)) : 0u;
                    } else {
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(phys_r, current_expert_idx)) : 0;
                    }
                }
            }

            // Round-robin rank selection (identical to SM100)
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
#if defined(AGY_DEDUP_TIER) && AGY_DEDUP_TIER
                    current_rank_in_expert_idx = (sym_buffer.rank_idx + current_rank_in_expert_idx) % kNumRanks;
#endif
                    break;
                }
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            const auto* slot_meta_ptr = workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t raw_token_topk_idx = *slot_meta_ptr;
            const uint32_t src_token_topk_idx = kAgyDedupScatReq ? (raw_token_topk_idx & 0x1fffffu) : raw_token_topk_idx;
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;
            const uint32_t packed_topk_meta = kAgyDedupScatReq
                ? (((raw_token_topk_idx >> 26) << 16) | (((raw_token_topk_idx >> 21) & 0x1fu) << 8) | src_topk_idx)
                : src_topk_idx;
            uint32_t pool_token_idx, m_block_pool_idx;
            if constexpr (kAgyAdaptBm) {
                const uint32_t cur_ebm = smem_expert_cache->expert_bm[current_expert_idx];
                const uint32_t cur_pool_base = smem_expert_cache->pool_block_offset[current_expert_idx];
                const uint32_t mb_in_exp = token_idx_in_expert / cur_ebm;
                const uint32_t in_mb = token_idx_in_expert - mb_in_exp * cur_ebm;
                m_block_pool_idx = cur_pool_base + mb_in_exp;
                pool_token_idx = m_block_pool_idx * BLOCK_M + in_mb;
            } else {
                pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                m_block_pool_idx = expert_pool_block_offset + token_idx_in_expert / BLOCK_M;
            }
            constexpr uint32_t kNumSFFloats = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");

            if constexpr (kAgyFastPull) {
                const uint32_t pushed_weight_bits = *(slot_meta_ptr + kNumMaxTokensPerRank);
                auto* local_sf_ptr = l1_sf_buffer.get_base_ptr<float>();

                if (current_rank_in_expert_idx == sym_buffer.rank_idx) {
                    // Direct local-rank HBM/L2 vectorized copy (bypasses SMEM + 2x TMA + mbarrier)
                    const auto* src_vec = reinterpret_cast<const uint4*>(
                        input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr());
                    auto* dst_vec = reinterpret_cast<uint4*>(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr());
                    #pragma unroll
                    for (uint32_t v = lane_idx; v < kHidden / 16u; v += 32u) {
                        dst_vec[v] = src_vec[v];
                    }
                    const auto* local_in_sf = input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>();
                    #pragma unroll
                    for (uint32_t s = lane_idx; s < kNumSFFloats; s += 32u) {
                        local_sf_ptr[s * kSFPoolStrideTokens + pool_token_idx] = local_in_sf[s];
                    }
                    if (lane_idx == 0) {
                        *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() =
                            __uint_as_float(pushed_weight_bits);
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, packed_topk_meta};
                    }
                    __syncwarp();
                    if (lane_idx == 0) {
                        ptx::red_add_rel(workspace.get_l1_full_count_ptr(m_block_pool_idx), 1u);
                    }
                    __syncwarp();
                } else {
                    uint32_t dedup_prev = 0u;
                    uint32_t* dedup_ptr = nullptr;
                    if constexpr (kAgyDedupDisp) {
                        dedup_ptr = workspace.get_l1_empty_count_ptr(0) +
                                    current_rank_in_expert_idx * kNumMaxTokensPerRank + src_token_idx;
                        if (lane_idx == 0) {
                            dedup_prev = atomicCAS(dedup_ptr, 0u, 0x40000000u | pool_token_idx);
                        }
                        dedup_prev = __shfl_sync(0xffffffff, dedup_prev, 0);
                    }

                    if (not kAgyDedupDisp or dedup_prev == 0u) {
                        // Unique remote token (or dedup leader): overlapped NVLink TMA + vectorized float4 SF
                        if (cute::elect_one_sync()) {
                            ptx::tma_load_1d(
                                pull_buffer.get_base_ptr(),
                                sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                               current_rank_in_expert_idx),
                                pull_mbarrier, kHidden);
                            ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                        }
                        const auto* remote_sf_ptr = sym_buffer.map(
                            input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                            current_rank_in_expert_idx);
                        if constexpr (kNumSFFloats % 4u == 0u and kNumSFFloats / 4u <= 32u) {
                            if (lane_idx < kNumSFFloats / 4u) {
                                const float4 sf4 = ptx::ld_global(reinterpret_cast<const float4*>(remote_sf_ptr) + lane_idx);
                                const uint32_t base_s = lane_idx * 4u;
                                local_sf_ptr[(base_s + 0u) * kSFPoolStrideTokens + pool_token_idx] = sf4.x;
                                local_sf_ptr[(base_s + 1u) * kSFPoolStrideTokens + pool_token_idx] = sf4.y;
                                local_sf_ptr[(base_s + 2u) * kSFPoolStrideTokens + pool_token_idx] = sf4.z;
                                local_sf_ptr[(base_s + 3u) * kSFPoolStrideTokens + pool_token_idx] = sf4.w;
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                                const uint32_t j = i * 32 + lane_idx;
                                if (j < kNumSFFloats)
                                    local_sf_ptr[j * kSFPoolStrideTokens + pool_token_idx] = remote_sf_ptr[j];
                            }
                        }
                        if (lane_idx == 0) {
                            *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() =
                                __uint_as_float(pushed_weight_bits);
                            *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                                {current_rank_in_expert_idx, src_token_idx, packed_topk_meta};
                        }
                        __syncwarp();
                        if (cute::elect_one_sync()) {
                            ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                            ptx::tma_store_1d(
                                l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                                pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
                            cute::tma_store_arrive();
                            ptx::tma_store_wait<0>();
                            if constexpr (kAgyDedupDisp) {
                                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                             :: "l"(dedup_ptr), "r"(0x80000000u | pool_token_idx) : "memory");
                            }
                            ptx::red_add_rel(workspace.get_l1_full_count_ptr(m_block_pool_idx), 1u);
                        }
                        __syncwarp();
                    } else if ((dedup_prev & 0x80000000u) != 0u) {
                        // Leader already published token to local L1 pool -> copy from local L2!
                        if (lane_idx == 0) {
                            (void)ptx::ld_acq(dedup_ptr);
                        }
                        __syncwarp();
                        copy_local_pool_token(
                            pool_token_idx, dedup_prev & 0x3fffffffu,
                            current_rank_in_expert_idx, src_token_idx,
                            packed_topk_meta, pushed_weight_bits, m_block_pool_idx);
                    } else {
                        // Leader still in flight -> defer into circular SMEM FIFO so unique pulls continue!
                        if (fifo_count == kDedupFifoCap) {
                            drain_one_dedup_fifo(true);
                        }
                        if (lane_idx == 0) {
                            my_dedup_fifo[fifo_tail] = {
                                pool_token_idx,
                                dedup_prev & 0x3fffffffu,
                                current_rank_in_expert_idx,
                                src_token_idx,
                                packed_topk_meta,
                                pushed_weight_bits,
                                m_block_pool_idx,
                                0u
                            };
                        }
                        __syncwarp();
                        fifo_tail = (fifo_tail + 1u) & (kDedupFifoCap - 1u);
                        ++ fifo_count;
                    }
                    if constexpr (kAgyDedupDisp) {
                        drain_one_dedup_fifo(false);
                    }
                }
            } else {
                // TMA pull token data into SMEM (Official path)
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kSFPoolStrideTokens + pool_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, packed_topk_meta};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(workspace.get_l1_full_count_ptr(m_block_pool_idx), 1);
                }
                __syncwarp();
            }
        }

        if constexpr (kAgyDedupDisp) {
            while (fifo_count > 0u) {
                drain_one_dedup_fifo(true);
            }
        }
        AGY_PROF_AFTER_PULL(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps,
                            (MegaMoEPhase::is_fused and not kScatDispActive and not kAgyAllIdleCombActive));
        } else {
            if (warp_idx == 0u) {
                AGY_PROF_AFTER_PULL(MegaMoEPhase, sm_idx, 0u, lane_idx, 1u,
                                    (MegaMoEPhase::is_fused and not kScatDispActive and not kAgyAllIdleCombActive));
            }
        }

        if constexpr (MegaMoEPhase::is_fused) {
            if constexpr (kScatDispActive) {
                // Temporal policy: s=comm uses the dispatch-warp handoff;
                // s=epi leaves the write to the inline epilogue path.
                const weave::PolicyDecision scatter_policy =
                    weave::load_published_policy(weave_control);
                const bool policy_valid = weave::policy_is_valid(scatter_policy);
                if (!policy_valid ||
                    scatter_policy.selected_scatter_mode ==
                        static_cast<uint32_t>(weave::ScatterPath::CommQueueAsync)) {
                    run_dispatch_l2_scatter_loop();
                }
                if constexpr (kAgyUseExpertCache) {
                    load_scheduler_from_expert_cache();
                } else {
                    scheduler.fetch_expert_recv_count();
                }
                ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            }
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            if constexpr (kAgyAllIdleCombActive) {
                AGY_PROF_DISP_COMB_BEGIN(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);
                constexpr uint32_t kTotalCombWarpsPerSM = kNumDispatchWarps + kNumEpilogueWarps;
                run_all_idle_reg_combine(sm_idx * kTotalCombWarpsPerSM + warp_idx,
                                         kNumSMs * kTotalCombWarpsPerSM);
                AGY_PROF_DISP_COMB_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);
            }
            AGY_PROF_DISP_CLEANUP_BEGIN(MegaMoEPhase, sm_idx, warp_idx, lane_idx,
                                        (not kScatDispActive and not kAgyAllIdleCombActive));
            cleanup_workspace();
            AGY_PROF_DISP_CLEANUP_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag,
                comm::BarrierTimeoutPolicy::TrapOnly>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);
            AGY_PROF_DISP_BAR3_END(MegaMoEPhase, sm_idx, warp_idx, lane_idx);
        }
        AGY_PROF_DISP_CLOSE(MegaMoEPhase, sm_idx, warp_idx, lane_idx, kNumDispatchWarps);
        return;

    // ROLE 2: GEMM TMA load warps (A+SFA and B)
    // The first two non-epilogue warps issue the TMA loads. Four-warp
    // configurations keep the remaining two warps idle for register handoff.
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        AGY_PROF_TMA_OPEN(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaA);
        if constexpr (kAgyUseExpertCache) {
            ptx::sync_unaligned(kPreGemmSyncThreads, kDispatchWithEpilogueBarrierIdx);
        }

        for_each_selected_block_role(
            0, lane_idx,
            [&](const bool& is_linear1_phase,
                const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                const uint32_t& pool_block_idx, const uint32_t& valid_m) {
            const auto tensor_map_a_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const bool has_valid_m = valid_m > 0;
            AGY_PROF_TMA_A_COUNT_TILE(MegaMoEPhase, sm_idx, lane_idx, is_linear1_phase);

            // Wait for the pool to be ready (L1) or establish TMA async proxy visibility (Interleaved L2).
            if (has_valid_m) {
                if (is_linear1_phase) {
#if defined(AGY_PROF) && (AGY_PROF > 0)
                    uint64_t _agy_tw0 = 0ull;
                    if (lane_idx == 0u && agy_prof_raw_ptr != nullptr && sm_idx < kAgyProfMaxSMs) {
                        _agy_tw0 = agy_prof_globaltimer_ns();
#if AGY_PROF == 1
                        agy_prof_emit_rec(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhase), sm_idx,
                                          kAgyProfTrackTmaA, _agy_tw0, kAgyEvWait, pool_block_idx);
#endif
                    }
#endif
                    const auto ptr = workspace.get_l1_full_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                    if constexpr (kAgyFastPull) {
                        if (cute::elect_one_sync()) {
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                        }
                        __syncwarp();
                    }
#if defined(AGY_PROF) && (AGY_PROF > 0)
                    if (lane_idx == 0u && agy_prof_raw_ptr != nullptr && sm_idx < kAgyProfMaxSMs) {
                        const uint64_t _agy_tw1 = agy_prof_globaltimer_ns();
                        auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhase);
                        _sl->sm[sm_idx].l1_pool_wait_ns += (_agy_tw1 - _agy_tw0);
#if AGY_PROF == 1
                        agy_prof_emit_rec(_sl, sm_idx, kAgyProfTrackTmaA, _agy_tw1,
                                          kAgyEvWait | kAgyFiTagEnd, pool_block_idx);
#endif
                    }
#endif
                } else if constexpr (kAgyInterleaveActive and kAgyInterleaveMode != 5 and kAgyInterleaveMode != 6) {
                    if (cute::elect_one_sync()) {
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    __syncwarp();
                }
            }
            AGY_PROF_TMA_TILE_LOAD_BEGIN(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaA,
                                         is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    if constexpr (kAgyInterleaveActive and (kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6)) {
                        if (not is_linear1_phase) {
                            const uint32_t req_bit = 1u << k_block_idx;
                            if constexpr (2u * L1_OUT_BLOCK_N == BLOCK_K) {
                                while (((ptx::ld_acq(interleave_ready_mb_queue + pool_block_idx) &
                                         ptx::ld_acq(interleave_l1_done_base + pool_block_idx)) & req_bit) == 0u);
                            } else {
                                while ((ptx::ld_acq(interleave_ready_mb_queue + pool_block_idx) & req_bit) == 0u);
                            }
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                        }
                    }
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t n_m_boxes = kAgyAdaptBm ? cute::max(1u, (valid_m + 15u) >> 4) : 1u;
                    const uint32_t act_tx_bytes = kAgyAdaptBm
                        ? kNumTMATilesPerStage * n_m_boxes * 16u * kTMATileK * sizeof(a_dtype_t)
                        : SMEM_A_SIZE_PER_STAGE;

                    // TMA load A. BK256 is two independently-swizzled BK128
                    // planes so each tensor-map box stays within one 128B atom.
                    // Adaptive BM_e loads only ceil(valid_m/16) 16-row boxes.
                    #pragma unroll
                    for (uint32_t k_tile = 0; k_tile < kNumTMATilesPerStage; ++ k_tile) {
                        if constexpr (kAgyAdaptBm) {
                            for (uint32_t mbx = 0; mbx < n_m_boxes; ++ mbx) {
                                tma::copy<kTMATileK, 16u, kSwizzleAMode, a_dtype_t>(
                                    tensor_map_a_ptr, full_barriers[stage_idx],
                                    smem_a[stage_idx] + k_tile * LOAD_BLOCK_M * kTMATileK + mbx * 16u * kTMATileK,
                                    k_idx + k_tile * kTMATileK, m_idx + mbx * 16u, 1);
                            }
                        } else {
                            tma::copy<kTMATileK, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                tensor_map_a_ptr, full_barriers[stage_idx],
                                smem_a[stage_idx] + k_tile * LOAD_BLOCK_M * kTMATileK,
                                k_idx + k_tile * kTMATileK, m_idx, 1);
                        }
                    }

                    // TMA load SFA with A on the same producer warp.
                    if (is_linear1_phase) {
                        // L1 SFA per-128: one vector per BK128 plane.
                        #pragma unroll
                        for (uint32_t sf_group = 0;
                             sf_group < BLOCK_K / kGranK; ++ sf_group) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] +
                                    sf_group * kL2SFAHalfStride,
                                m_idx,
                                k_block_idx * (BLOCK_K / kGranK) + sf_group,
                                1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            act_tx_bytes +
                                (BLOCK_K / kGranK) * BLOCK_M * sizeof(float));
                    } else {
                        // L2 SFA per-64: one TMA per scale group.
                        #pragma unroll
                        for (uint32_t sf_group = 0;
                             sf_group < kNumL2SFAKGroups; ++ sf_group) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] +
                                    sf_group * kL2SFAHalfStride,
                                m_idx,
                                k_block_idx * kNumL2SFAKGroups + sf_group,
                                1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            act_tx_bytes +
                                kNumL2SFAKGroups * BLOCK_M * sizeof(float));
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
            AGY_PROF_TMA_TILE_LOAD_END(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaA,
                                       is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
        });
        AGY_PROF_TMA_CLOSE(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaA);

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        AGY_PROF_TMA_OPEN(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaB);
        if constexpr (kAgyUseExpertCache) {
            ptx::sync_unaligned(kPreGemmSyncThreads, kDispatchWithEpilogueBarrierIdx);
        }

        for_each_selected_block_role(
            1, 32u + lane_idx,
            [&](const bool& is_linear1_phase,
                const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                const uint32_t& pool_block_idx, const uint32_t& valid_m) {
            const auto tensor_map_b_ptr =
                !is_linear1_phase ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = !is_linear1_phase ? L2_SHAPE_N : L1_SHAPE_N;
            const uint32_t mcast_b =
                (kAgyMcast == 2u and m_block_idx < (smem_expert_cache->m_blocks[local_expert_idx] & ~1u))
                    ? 2u : 1u;

            AGY_PROF_TMA_TILE_LOAD_BEGIN(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaB,
                                         is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load B in independent BK128 x BN<=256 planes. The
                    // tensor-map box limit requires two N planes for BN512.
                    #pragma unroll
                    for (uint32_t k_tile = 0; k_tile < kNumTMATilesPerStage; ++ k_tile) {
                        #pragma unroll
                        for (uint32_t n_tile = 0;
                             n_tile < kNumTMANTilesPerStage; ++ n_tile) {
                            tma::copy<kTMATileK, kTMATileN, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] +
                                    k_tile * LOAD_BLOCK_N * kTMATileK +
                                    n_tile * kTMATileN * kTMATileK,
                                k_idx + k_tile * kTMATileK,
                                n_idx + n_tile * kTMATileN,
                                mcast_b);
                        }
                    }

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
            AGY_PROF_TMA_TILE_LOAD_END(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaB,
                                       is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
        });
        if constexpr (kAgyMcast == 2u) {
            for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                empty_barriers[stage_idx]->wait(phase ^ 1);
        }
        AGY_PROF_TMA_CLOSE(MegaMoEPhase, sm_idx, lane_idx, kAgyProfTrackTmaB);

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        if constexpr (kRoleMode == 1) {
            role3_noinline<2>(DG_SM90_FP8_MOE_KERNEL_ARGS);
            if constexpr (MegaMoEPhase::runs_linear2) {
                role3_noinline<3>(DG_SM90_FP8_MOE_KERNEL_ARGS);
            }
            return;
        }
    }
    } // end kRoleMode <= 1
    if constexpr (kRoleMode == 0 or kRoleMode == 2 or kRoleMode == 3) {
    if (kRoleMode >= 2 or warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if constexpr (kAgyMcast == 2u) {
                if (lane_idx < 2u)
                    empty_barriers[s]->arrive(lane_idx);
            } else {
                if (lane_idx == 0)
                    empty_barriers[s]->arrive();
            }
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;
        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N = kSplitNWarpgroups ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;

        DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups, "Invalid warpgroup split");
        if constexpr (kSplitNWarpgroups and not kSwapABActive) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Split WGs must each run one WGMMA tile per K-block");
        } else if constexpr (not kSwapABActive) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");
        }

        if constexpr (kRoleMode != 3) {
        AGY_PROF_MATH_OPEN(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx);
        // Sync with dispatch
        ptx::sync_unaligned(kPreGemmSyncThreads, kDispatchWithEpilogueBarrierIdx);
        uint32_t math_scat_phase = 0u;
        bool scat_ever_used = false;
        bool scat_slot_free = false;

        for_each_selected_block_role(
            2, 64u + epilogue_thread_idx,
            [&](const bool& is_linear1_phase,
                const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                const uint32_t& pool_block_idx, const uint32_t& valid_m) {
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
            const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
            const uint32_t wg_n_idx = epilogue_wg_n_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_idx = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset = epilogue_wg_m_idx * WG_BLOCK_M;
            const uint32_t smem_cd_l1_wg_offset = kSplitNWarpgroups ? 0 :
                epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;
            // GEMM
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread =
                (kSwapABActive and BLOCK_M == 16) ? 8u :
                ((kSwapABActive and BLOCK_M == 32) ? 16u : WGMMA::kNumAccum);
            float final_accum[kAccumPerThread];
            if constexpr (not kBF16ScaledAccum) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                    final_accum[i] = 0.0f;
            }
            float accum[kAccumPerThread];

            const auto run_default_gemm_loop = [&]() {
                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert =
                    (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert =
                    (kHidden / 128) * kL2SFKBlocks;
                nv_bfloat162 swap_final_bf16[kAccumPerThread / 2];
                if constexpr (kBF16ScaledAccum and kSwapABActive) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                        swap_final_bf16[i] = __float2bfloat162_rn(0.0f);
                }

                if constexpr (kAgySwapPipe and kSwapABActive and BLOCK_K == 128 and kNumStages >= 3) {
                    auto run_swap_piped = [&]<uint32_t N_SWAP>() {
                        using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                        constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                        float swap_acc_0[kSwapAccum];
                        float swap_acc_1[kSwapAccum];

                        auto issue_swap = [&]<uint32_t kOff, uint32_t kCnt>(
                                              const uint32_t& s_idx, float* acc) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(acc[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < kCnt; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[s_idx] + wg_n_idx * BLOCK_K + kOff + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[s_idx] + kOff + k * SwapWGMMA::K, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, acc, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(acc[i]);
                        };

                        if (is_linear1_phase) {
                            auto promote_swap_l1 = [&](const uint32_t& s_idx, const float& g_sf, const float& u_sf,
                                                       float* acc) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[s_idx] + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[s_idx] + token_1) : 0.0f;
                                    if constexpr (kBF16ScaledAccum) {
                                        swap_final_bf16[2 * i] = __hfma2(
                                            __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]),
                                            __floats2bfloat162_rn(scale_0 * g_sf, scale_1 * g_sf),
                                            swap_final_bf16[2 * i]);
                                        swap_final_bf16[2 * i + 1] = __hfma2(
                                            __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]),
                                            __floats2bfloat162_rn(scale_0 * u_sf, scale_1 * u_sf),
                                            swap_final_bf16[2 * i + 1]);
                                    } else {
                                        final_accum[i * 4 + 0] += scale_0 * g_sf * acc[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * u_sf * acc[i * 4 + 2];
                                        final_accum[i * 4 + 1] += scale_1 * g_sf * acc[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * u_sf * acc[i * 4 + 3];
                                    }
                                }
                            };

                            const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* wsf_base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;

                            uint32_t k_block_idx = 0;
                            full_barriers[stage_idx]->wait(phase);
                            uint32_t s0 = stage_idx;
                            float g0 = __ldg(wsf_base + gate_n * kL1SFKBlocks + 0);
                            float u0 = __ldg(wsf_base + up_n * kL1SFKBlocks + 0);
                            issue_swap.template operator()<0, BLOCK_K / SwapWGMMA::K>(s0, swap_acc_0);
                            advance_pipeline(k_block_idx);

                            const uint32_t num_pairs = num_k_blocks / 2u;
                            uint32_t s1 = 0;
                            float g1 = 0.0f, u1 = 0.0f;
                            for (uint32_t pair = 0; ; ++ pair) {
                                full_barriers[stage_idx]->wait(phase);
                                s1 = stage_idx;
                                g1 = __ldg(wsf_base + gate_n * kL1SFKBlocks + k_block_idx);
                                u1 = __ldg(wsf_base + up_n * kL1SFKBlocks + k_block_idx);
                                issue_swap.template operator()<0, BLOCK_K / SwapWGMMA::K>(s1, swap_acc_1);
                                advance_pipeline(k_block_idx);

                                ptx::warpgroup_wait<1>();
                                promote_swap_l1(s0, g0, u0, swap_acc_0);
                                arrive_empty_barrier(s0);

                                if (pair + 1u == num_pairs) break;

                                full_barriers[stage_idx]->wait(phase);
                                s0 = stage_idx;
                                g0 = __ldg(wsf_base + gate_n * kL1SFKBlocks + k_block_idx);
                                u0 = __ldg(wsf_base + up_n * kL1SFKBlocks + k_block_idx);
                                issue_swap.template operator()<0, BLOCK_K / SwapWGMMA::K>(s0, swap_acc_0);
                                advance_pipeline(k_block_idx);

                                ptx::warpgroup_wait<1>();
                                promote_swap_l1(s1, g1, u1, swap_acc_1);
                                arrive_empty_barrier(s1);
                            }
                            ptx::warpgroup_wait<0>();
                            promote_swap_l1(s1, g1, u1, swap_acc_1);
                            arrive_empty_barrier(s1);
                        } else {
                            auto promote_swap_l2 = [&](const uint32_t& s_idx, const uint32_t& sf_grp,
                                                       const float& w_sf, float* acc) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[s_idx] + sf_grp * kL2SFAHalfStride + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[s_idx] + sf_grp * kL2SFAHalfStride + token_1) : 0.0f;
                                    if constexpr (kBF16ScaledAccum) {
                                        swap_final_bf16[2 * i] = __hfma2(
                                            __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]),
                                            __floats2bfloat162_rn(scale_0 * w_sf, scale_1 * w_sf),
                                            swap_final_bf16[2 * i]);
                                        swap_final_bf16[2 * i + 1] = __hfma2(
                                            __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]),
                                            __floats2bfloat162_rn(scale_0 * w_sf, scale_1 * w_sf),
                                            swap_final_bf16[2 * i + 1]);
                                    } else {
                                        final_accum[i * 4 + 0] += scale_0 * w_sf * acc[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * w_sf * acc[i * 4 + 2];
                                        final_accum[i * 4 + 1] += scale_1 * w_sf * acc[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * w_sf * acc[i * 4 + 3];
                                    }
                                }
                            };

                            const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                            const float* wsf_base = l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_n * kL2SFKBlocks;

                            uint32_t k_block_idx = 0;
                            full_barriers[stage_idx]->wait(phase);
                            uint32_t prev_s = stage_idx;
                            float prev_wsf = __ldg(wsf_base + 0);
                            issue_swap.template operator()<0, (BLOCK_K / 2) / SwapWGMMA::K>(prev_s, swap_acc_0);
                            issue_swap.template operator()<BLOCK_K / 2, (BLOCK_K / 2) / SwapWGMMA::K>(prev_s, swap_acc_1);
                            ptx::warpgroup_wait<1>();
                            promote_swap_l2(prev_s, 0u, prev_wsf, swap_acc_0);
                            advance_pipeline(k_block_idx);

                            for (; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                                full_barriers[stage_idx]->wait(phase);
                                const uint32_t curr_s = stage_idx;
                                const float curr_wsf = __ldg(wsf_base + k_block_idx);
                                issue_swap.template operator()<0, (BLOCK_K / 2) / SwapWGMMA::K>(curr_s, swap_acc_0);
                                ptx::warpgroup_wait<1>();
                                promote_swap_l2(prev_s, 1u, prev_wsf, swap_acc_1);
                                arrive_empty_barrier(prev_s);

                                issue_swap.template operator()<BLOCK_K / 2, (BLOCK_K / 2) / SwapWGMMA::K>(curr_s, swap_acc_1);
                                ptx::warpgroup_wait<1>();
                                promote_swap_l2(curr_s, 0u, curr_wsf, swap_acc_0);
                                prev_s = curr_s;
                                prev_wsf = curr_wsf;
                            }
                            ptx::warpgroup_wait<0>();
                            promote_swap_l2(prev_s, 1u, prev_wsf, swap_acc_1);
                            arrive_empty_barrier(prev_s);
                        }
                    };

                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if constexpr (BLOCK_M == 16) {
                        if (n_swap <= 8) run_swap_piped.template operator()<8>();
                        else             run_swap_piped.template operator()<16>();
                    } else if constexpr (BLOCK_M == 32) {
                        if (n_swap <= 8)       run_swap_piped.template operator()<8>();
                        else if (n_swap <= 16) run_swap_piped.template operator()<16>();
                        else if (n_swap <= 24) run_swap_piped.template operator()<24>();
                        else                   run_swap_piped.template operator()<32>();
                    } else {
                        if (n_swap <= 8)       run_swap_piped.template operator()<8>();
                        else if (n_swap <= 16) run_swap_piped.template operator()<16>();
                        else if (n_swap <= 32) run_swap_piped.template operator()<32>();
                        else                   run_swap_piped.template operator()<64>();
                    }
                } else {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                float gate_sf = 0.0f, up_sf = 0.0f;
                float l2_sf_lo = 0.0f, l2_sf_hi = 0.0f;
                full_barriers[stage_idx]->wait(phase);

                // Read SF (must precede warpgroup_arrive; skipped under swapAB which loads per-token SF)
                float scale_a_0_lo = 0.0f, scale_a_1_lo = 0.0f;
                float scale_a_0_hi = 0.0f, scale_a_1_hi = 0.0f;  // Only used in L2 (per-64 K)
                if constexpr (not kSwapABActive) {
                    if (is_linear1_phase) {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    } else {
                        // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                    }
                }

                // ----- Block (128, 128) weight SF (loaded directly from global) -----
                // L1 weight SF shape: (E, 2*IH/128, H/128) MN-major. The N axis is
                // [gate(IH/128), up(IH/128)]; with the gate/up gran-8 interleave on
                // the FP8 weight, each BLOCK_N=128 tile covers 64 rows of gate plus
                // 64 rows of up taken from the same original 128-row block, so:
                //     gate_sf_n = n_block_idx / 2
                //     up_sf_n   = (IH/128) + n_block_idx / 2
                //
                // L2 weight SF shape: (E, H/128, IH/128) MN-major. One scalar per
                // (BLOCK_N, BLOCK_K) tile, broadcast across all WGMMA accumulators.
                //
                if (is_linear1_phase) {
                    const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                    const uint32_t up_n   = kL1SFGateBlks + gate_n;
                    const float* base = l1_weights_sf +
                        local_expert_idx * kL1SFPerExpert + k_block_idx;
                    gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                    up_sf   = __ldg(base + up_n * kL1SFKBlocks);
                } else {
                    const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                    const float* base = l2_weights_sf +
                        local_expert_idx * kL2SFPerExpert + k_block_idx;
                    l2_sf_lo = __ldg(base + sf_n * kL2SFKBlocks);
                    if constexpr (WG_BLOCK_N > 128)
                        l2_sf_hi = __ldg(base + (sf_n + 1u) * kL2SFKBlocks);
                    else
                        l2_sf_hi = l2_sf_lo;
                }

                if (is_linear1_phase) {
                    if constexpr (kSwapABActive) {
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                const float scale_0 = token_0 < valid_m ?
                                    ptx::ld_shared(smem_sfa[stage_idx] + token_0) : 0.0f;
                                const float scale_1 = token_1 < valid_m ?
                                    ptx::ld_shared(smem_sfa[stage_idx] + token_1) : 0.0f;
                                if constexpr (kBF16ScaledAccum) {
                                    swap_final_bf16[2 * i] = __hfma2(
                                        __floats2bfloat162_rn(
                                            swap_accum[4 * i], swap_accum[4 * i + 1]),
                                        __floats2bfloat162_rn(
                                            scale_0 * gate_sf, scale_1 * gate_sf),
                                        swap_final_bf16[2 * i]);
                                    swap_final_bf16[2 * i + 1] = __hfma2(
                                        __floats2bfloat162_rn(
                                            swap_accum[4 * i + 2], swap_accum[4 * i + 3]),
                                        __floats2bfloat162_rn(
                                            scale_0 * up_sf, scale_1 * up_sf),
                                        swap_final_bf16[2 * i + 1]);
                                } else {
                                    final_accum[i * 4 + 0] +=
                                        scale_0 * gate_sf * swap_accum[i * 4 + 0];
                                    final_accum[i * 4 + 2] +=
                                        scale_0 * up_sf * swap_accum[i * 4 + 2];
                                    final_accum[i * 4 + 1] +=
                                        scale_1 * gate_sf * swap_accum[i * 4 + 1];
                                    final_accum[i * 4 + 3] +=
                                        scale_1 * up_sf * swap_accum[i * 4 + 3];
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (BLOCK_M == 16) {
                            if (n_swap <= 8)       run_swap_ab_l1.template operator()<8>();
                            else                   run_swap_ab_l1.template operator()<16>();
                        } else if constexpr (BLOCK_M == 32) {
                            if (n_swap <= 8)       run_swap_ab_l1.template operator()<8>();
                            else if (n_swap <= 16) run_swap_ab_l1.template operator()<16>();
                            else if (n_swap <= 24) run_swap_ab_l1.template operator()<24>();
                            else                   run_swap_ab_l1.template operator()<32>();
#if !(defined(AGY_FULL_NSWAP) && AGY_FULL_NSWAP)
                        } else if constexpr (kIntermediateHidden <= 2048) {
                            // SWAPAB_FINE_LADDER: 8-token granularity, matching the
                            // large-IH path and eliminating the 0.366 -> 5.94e-04
                            // numerical drift source at BLOCK_N=256.
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else if (n_swap <= 24) {
                                run_swap_ab_l1.template operator()<24>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l1.template operator()<32>();
                            } else if (n_swap <= 40) {
                                run_swap_ab_l1.template operator()<40>();
                            } else if (n_swap <= 48) {
                                run_swap_ab_l1.template operator()<48>();
                            } else if (n_swap <= 56) {
                                run_swap_ab_l1.template operator()<56>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
#endif
                        } else {
                            switch (n_swap) {
                                case 8:  run_swap_ab_l1.template operator()<8>();  break;
                                case 16: run_swap_ab_l1.template operator()<16>(); break;
                                case 24: run_swap_ab_l1.template operator()<24>(); break;
                                case 32: run_swap_ab_l1.template operator()<32>(); break;
                                case 40: run_swap_ab_l1.template operator()<40>(); break;
                                case 48: run_swap_ab_l1.template operator()<48>(); break;
                                case 56: run_swap_ab_l1.template operator()<56>(); break;
                                default: run_swap_ab_l1.template operator()<64>(); break;
                            }
                        }
                    } else {
                        // Single per-128 K-block WGMMA group
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        // L1: gate/up alternate at gran=8 along N; each `i` block of 8
                        // cols belongs entirely to one of {gate, up}, so .x and .y
                        // share the same scalar.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float sb = (i & 1u) ? up_sf : gate_sf;
                            final_accum[i*4+0] += scale_a_0_lo * sb * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * sb * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * sb * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * sb * accum[i*4+3];
                        }
                    }
                } else {
                    if constexpr (kSwapABActive) {
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            auto promote_swap_accum = [&](const uint32_t& sf_group) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_1) : 0.0f;
                                    if constexpr (kBF16ScaledAccum) {
                                        swap_final_bf16[2 * i] = __hfma2(
                                            __floats2bfloat162_rn(
                                                swap_accum[4 * i],
                                                swap_accum[4 * i + 1]),
                                            __floats2bfloat162_rn(
                                                scale_0 * l2_sf_lo,
                                                scale_1 * l2_sf_lo),
                                            swap_final_bf16[2 * i]);
                                        swap_final_bf16[2 * i + 1] = __hfma2(
                                            __floats2bfloat162_rn(
                                                swap_accum[4 * i + 2],
                                                swap_accum[4 * i + 3]),
                                            __floats2bfloat162_rn(
                                                scale_0 * l2_sf_lo,
                                                scale_1 * l2_sf_lo),
                                            swap_final_bf16[2 * i + 1]);
                                    } else {
                                        final_accum[i * 4 + 0] +=
                                            scale_0 * l2_sf_lo * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] +=
                                            scale_0 * l2_sf_lo * swap_accum[i * 4 + 2];
                                        final_accum[i * 4 + 1] +=
                                            scale_1 * l2_sf_lo * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] +=
                                            scale_1 * l2_sf_lo * swap_accum[i * 4 + 3];
                                    }
                                }
                            };

                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_wait<0>();
                            promote_swap_accum(0);

                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * SwapWGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k_off, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_wait<0>();
                            promote_swap_accum(1);

                            arrive_empty_barrier(stage_idx);
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (BLOCK_M == 16) {
                            if (n_swap <= 8)       run_swap_ab_l2.template operator()<8>();
                            else                   run_swap_ab_l2.template operator()<16>();
                        } else if constexpr (BLOCK_M == 32) {
                            if (n_swap <= 8)       run_swap_ab_l2.template operator()<8>();
                            else if (n_swap <= 16) run_swap_ab_l2.template operator()<16>();
                            else if (n_swap <= 24) run_swap_ab_l2.template operator()<24>();
                            else                   run_swap_ab_l2.template operator()<32>();
#if !(defined(AGY_FULL_NSWAP) && AGY_FULL_NSWAP)
                        } else if constexpr (kIntermediateHidden <= 2048) {
                            // SWAPAB_FINE_LADDER: 8-token granularity, matching the
                            // large-IH path and eliminating the 0.366 -> 5.94e-04
                            // numerical drift source at BLOCK_N=256.
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 24) {
                                run_swap_ab_l2.template operator()<24>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else if (n_swap <= 40) {
                                run_swap_ab_l2.template operator()<40>();
                            } else if (n_swap <= 48) {
                                run_swap_ab_l2.template operator()<48>();
                            } else if (n_swap <= 56) {
                                run_swap_ab_l2.template operator()<56>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
#endif
                        } else {
                            switch (n_swap) {
                                case 8:  run_swap_ab_l2.template operator()<8>();  break;
                                case 16: run_swap_ab_l2.template operator()<16>(); break;
                                case 24: run_swap_ab_l2.template operator()<24>(); break;
                                case 32: run_swap_ab_l2.template operator()<32>(); break;
                                case 40: run_swap_ab_l2.template operator()<40>(); break;
                                case 48: run_swap_ab_l2.template operator()<48>(); break;
                                case 56: run_swap_ab_l2.template operator()<56>(); break;
                                default: run_swap_ab_l2.template operator()<64>(); break;
                            }
                        }
                    } else {
                    // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                    // First half: K=0..63, SFA = scale_a_*_lo
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    // L2 weight SF is per 128 output columns; M64N256 spans two SF groups.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                        final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                    }

                    // Second half: K=64..127, SFA = scale_a_*_hi
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    arrive_empty_barrier(stage_idx);

                    // L2 second half: same SFA half, still choose weight SF by N chunk.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                        final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                    }
                    }
                }
                }
                }

                if constexpr (kBF16ScaledAccum and kSwapABActive) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                        const float2 pair =
                            __bfloat1622float2(swap_final_bf16[i]);
                        final_accum[2 * i] = pair.x;
                        final_accum[2 * i + 1] = pair.y;
                    }
                }
            };

            const auto run_bk256_gemm_loop = [&]() {
                if constexpr (BLOCK_K == 256 and kSwapABActive) {
                    constexpr uint32_t kL1SFKBlocks = kHidden / kGranK;
                    constexpr uint32_t kL2SFKBlocks = kIntermediateHidden / kGranK;
                    constexpr uint32_t kL1SFGateBlks = kIntermediateHidden / kGranK;
                    constexpr uint32_t kL1SFPerExpert =
                        (kIntermediateHidden * 2 / kGranK) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert =
                        (kHidden / kGranK) * kL2SFKBlocks;
                    nv_bfloat162 swap_final_bf16[kAccumPerThread / 2];
                    if constexpr (kBF16ScaledAccum) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                            swap_final_bf16[i] = __float2bfloat162_rn(0.0f);
                    }

                    if constexpr (kAgySwapPipe and kNumStages >= 2 and
                                  (kBF16ScaledAccum or kNumEpilogueRegisters >= 208 or BLOCK_M <= 32)) {
                        auto run_swap_bk256_piped = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_acc_0[kSwapAccum];
                            float swap_acc_1[kSwapAccum];

                            auto issue_swap_bk256 = [&]<uint32_t kPlane, uint32_t kOff, uint32_t kCnt>(
                                                        const uint32_t& s_idx, float* acc) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < kCnt; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[s_idx] + kPlane * LOAD_BLOCK_N * kTMATileK +
                                        wg_n_idx * kTMATileK + kOff + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[s_idx] + kPlane * LOAD_BLOCK_M * kTMATileK +
                                        kOff + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, acc, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                            };

                            if (is_linear1_phase) {
                                auto promote_swap_l1 = [&](const uint32_t& s_idx, const uint32_t& k_plane,
                                                           const float& g_sf, const float& u_sf,
                                                           float* acc) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(acc[i]);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const uint32_t token_0 = i * 8 + col_idx * 2;
                                        const uint32_t token_1 = token_0 + 1;
                                        const float scale_0 = token_0 < valid_m ?
                                            ptx::ld_shared(smem_sfa[s_idx] + k_plane * kL2SFAHalfStride + token_0) : 0.0f;
                                        const float scale_1 = token_1 < valid_m ?
                                            ptx::ld_shared(smem_sfa[s_idx] + k_plane * kL2SFAHalfStride + token_1) : 0.0f;
                                        if constexpr (kBF16ScaledAccum) {
                                            swap_final_bf16[2 * i] = __hfma2(
                                                __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]),
                                                __floats2bfloat162_rn(scale_0 * g_sf, scale_1 * g_sf),
                                                swap_final_bf16[2 * i]);
                                            swap_final_bf16[2 * i + 1] = __hfma2(
                                                __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]),
                                                __floats2bfloat162_rn(scale_0 * u_sf, scale_1 * u_sf),
                                                swap_final_bf16[2 * i + 1]);
                                        } else {
                                            final_accum[i * 4 + 0] += scale_0 * g_sf * acc[i * 4 + 0];
                                            final_accum[i * 4 + 2] += scale_0 * u_sf * acc[i * 4 + 2];
                                            final_accum[i * 4 + 1] += scale_1 * g_sf * acc[i * 4 + 1];
                                            final_accum[i * 4 + 3] += scale_1 * u_sf * acc[i * 4 + 3];
                                        }
                                    }
                                };

                                const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                                const uint32_t up_n   = kL1SFGateBlks + gate_n;
                                const float* wsf_base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;
                                const float* wsf_gate = wsf_base + gate_n * kL1SFKBlocks;
                                const float* wsf_up   = wsf_base + up_n * kL1SFKBlocks;

                                uint32_t k_block_idx = 0;
                                full_barriers[stage_idx]->wait(phase);
                                uint32_t prev_s = stage_idx;
                                const float g_p0 = __ldg(wsf_gate + 0u);
                                const float u_p0 = __ldg(wsf_up + 0u);
                                float prev_g_p1 = __ldg(wsf_gate + 1u);
                                float prev_u_p1 = __ldg(wsf_up + 1u);

                                issue_swap_bk256.template operator()<0u, 0u, kTMATileK / SwapWGMMA::K>(prev_s, swap_acc_0);
                                issue_swap_bk256.template operator()<1u, 0u, kTMATileK / SwapWGMMA::K>(prev_s, swap_acc_1);
                                ptx::warpgroup_wait<1>();
                                promote_swap_l1(prev_s, 0u, g_p0, u_p0, swap_acc_0);
                                advance_pipeline(k_block_idx);

                                for (; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                                    full_barriers[stage_idx]->wait(phase);
                                    const uint32_t curr_s = stage_idx;
                                    const uint32_t sf_k = k_block_idx * kNumTMATilesPerStage;
                                    const float curr_g_p0 = __ldg(wsf_gate + sf_k + 0u);
                                    const float curr_u_p0 = __ldg(wsf_up + sf_k + 0u);
                                    const float curr_g_p1 = __ldg(wsf_gate + sf_k + 1u);
                                    const float curr_u_p1 = __ldg(wsf_up + sf_k + 1u);

                                    issue_swap_bk256.template operator()<0u, 0u, kTMATileK / SwapWGMMA::K>(curr_s, swap_acc_0);
                                    ptx::warpgroup_wait<1>();
                                    promote_swap_l1(prev_s, 1u, prev_g_p1, prev_u_p1, swap_acc_1);
                                    arrive_empty_barrier(prev_s);

                                    issue_swap_bk256.template operator()<1u, 0u, kTMATileK / SwapWGMMA::K>(curr_s, swap_acc_1);
                                    ptx::warpgroup_wait<1>();
                                    promote_swap_l1(curr_s, 0u, curr_g_p0, curr_u_p0, swap_acc_0);

                                    prev_s = curr_s;
                                    prev_g_p1 = curr_g_p1;
                                    prev_u_p1 = curr_u_p1;
                                }
                                ptx::warpgroup_wait<0>();
                                promote_swap_l1(prev_s, 1u, prev_g_p1, prev_u_p1, swap_acc_1);
                                arrive_empty_barrier(prev_s);
                            } else {
                                auto promote_swap_l2 = [&](const uint32_t& s_idx, const uint32_t& sf_grp,
                                                           const float& w_sf, float* acc) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(acc[i]);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const uint32_t token_0 = i * 8 + col_idx * 2;
                                        const uint32_t token_1 = token_0 + 1;
                                        const float scale_0 = token_0 < valid_m ?
                                            ptx::ld_shared(smem_sfa[s_idx] + sf_grp * kL2SFAHalfStride + token_0) : 0.0f;
                                        const float scale_1 = token_1 < valid_m ?
                                            ptx::ld_shared(smem_sfa[s_idx] + sf_grp * kL2SFAHalfStride + token_1) : 0.0f;
                                        if constexpr (kBF16ScaledAccum) {
                                            swap_final_bf16[2 * i] = __hfma2(
                                                __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]),
                                                __floats2bfloat162_rn(scale_0 * w_sf, scale_1 * w_sf),
                                                swap_final_bf16[2 * i]);
                                            swap_final_bf16[2 * i + 1] = __hfma2(
                                                __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]),
                                                __floats2bfloat162_rn(scale_0 * w_sf, scale_1 * w_sf),
                                                swap_final_bf16[2 * i + 1]);
                                        } else {
                                            final_accum[i * 4 + 0] += scale_0 * w_sf * acc[i * 4 + 0];
                                            final_accum[i * 4 + 2] += scale_0 * w_sf * acc[i * 4 + 2];
                                            final_accum[i * 4 + 1] += scale_1 * w_sf * acc[i * 4 + 1];
                                            final_accum[i * 4 + 3] += scale_1 * w_sf * acc[i * 4 + 3];
                                        }
                                    }
                                };

                                const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                                const float* wsf_base = l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_n * kL2SFKBlocks;

                                uint32_t k_block_idx = 0;
                                full_barriers[stage_idx]->wait(phase);
                                uint32_t prev_s = stage_idx;
                                const float wsf_p0 = __ldg(wsf_base + 0u);
                                float prev_wsf_p1 = __ldg(wsf_base + 1u);

                                issue_swap_bk256.template operator()<0u, 0u, kL2ActsSFGranK / SwapWGMMA::K>(prev_s, swap_acc_0);
                                issue_swap_bk256.template operator()<0u, kL2ActsSFGranK, kL2ActsSFGranK / SwapWGMMA::K>(prev_s, swap_acc_1);
                                ptx::warpgroup_wait<1>();
                                promote_swap_l2(prev_s, 0u, wsf_p0, swap_acc_0);

                                issue_swap_bk256.template operator()<1u, 0u, kL2ActsSFGranK / SwapWGMMA::K>(prev_s, swap_acc_0);
                                ptx::warpgroup_wait<1>();
                                promote_swap_l2(prev_s, 1u, wsf_p0, swap_acc_1);

                                issue_swap_bk256.template operator()<1u, kL2ActsSFGranK, kL2ActsSFGranK / SwapWGMMA::K>(prev_s, swap_acc_1);
                                ptx::warpgroup_wait<1>();
                                promote_swap_l2(prev_s, 2u, prev_wsf_p1, swap_acc_0);
                                advance_pipeline(k_block_idx);

                                for (; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                                    full_barriers[stage_idx]->wait(phase);
                                    const uint32_t curr_s = stage_idx;
                                    const uint32_t sf_k = k_block_idx * kNumTMATilesPerStage;
                                    const float curr_wsf_p0 = __ldg(wsf_base + sf_k + 0u);
                                    const float curr_wsf_p1 = __ldg(wsf_base + sf_k + 1u);

                                    issue_swap_bk256.template operator()<0u, 0u, kL2ActsSFGranK / SwapWGMMA::K>(curr_s, swap_acc_0);
                                    ptx::warpgroup_wait<1>();
                                    promote_swap_l2(prev_s, 3u, prev_wsf_p1, swap_acc_1);
                                    arrive_empty_barrier(prev_s);

                                    issue_swap_bk256.template operator()<0u, kL2ActsSFGranK, kL2ActsSFGranK / SwapWGMMA::K>(curr_s, swap_acc_1);
                                    ptx::warpgroup_wait<1>();
                                    promote_swap_l2(curr_s, 0u, curr_wsf_p0, swap_acc_0);

                                    issue_swap_bk256.template operator()<1u, 0u, kL2ActsSFGranK / SwapWGMMA::K>(curr_s, swap_acc_0);
                                    ptx::warpgroup_wait<1>();
                                    promote_swap_l2(curr_s, 1u, curr_wsf_p0, swap_acc_1);

                                    issue_swap_bk256.template operator()<1u, kL2ActsSFGranK, kL2ActsSFGranK / SwapWGMMA::K>(curr_s, swap_acc_1);
                                    ptx::warpgroup_wait<1>();
                                    promote_swap_l2(curr_s, 2u, curr_wsf_p1, swap_acc_0);

                                    prev_s = curr_s;
                                    prev_wsf_p1 = curr_wsf_p1;
                                }
                                ptx::warpgroup_wait<0>();
                                promote_swap_l2(prev_s, 3u, prev_wsf_p1, swap_acc_1);
                                arrive_empty_barrier(prev_s);
                            }
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (BLOCK_M == 16) {
                            if (n_swap <= 8) run_swap_bk256_piped.template operator()<8>();
                            else             run_swap_bk256_piped.template operator()<16>();
                        } else if constexpr (BLOCK_M == 32) {
                            if (n_swap <= 8)       run_swap_bk256_piped.template operator()<8>();
                            else if (n_swap <= 16) run_swap_bk256_piped.template operator()<16>();
                            else if (n_swap <= 24) run_swap_bk256_piped.template operator()<24>();
                            else                   run_swap_bk256_piped.template operator()<32>();
                        } else {
#if !(defined(AGY_FULL_NSWAP) && AGY_FULL_NSWAP)
                            if (n_swap <= 8)       run_swap_bk256_piped.template operator()<8>();
                            else if (n_swap <= 16) run_swap_bk256_piped.template operator()<16>();
                            else if (n_swap <= 32) run_swap_bk256_piped.template operator()<32>();
                            else                   run_swap_bk256_piped.template operator()<64>();
#else
                            switch (n_swap) {
                                case 0:
                                case 8:  run_swap_bk256_piped.template operator()<8>();  break;
                                case 16: run_swap_bk256_piped.template operator()<16>(); break;
                                case 24: run_swap_bk256_piped.template operator()<24>(); break;
                                case 32: run_swap_bk256_piped.template operator()<32>(); break;
                                case 40: run_swap_bk256_piped.template operator()<40>(); break;
                                case 48: run_swap_bk256_piped.template operator()<48>(); break;
                                case 56: run_swap_bk256_piped.template operator()<56>(); break;
                                default: run_swap_bk256_piped.template operator()<64>(); break;
                            }
#endif
                        }
                    } else {
                    auto run_swap_ab_bk256_step = [&]<uint32_t N_SWAP>(const uint32_t& k_block_idx) {
                        using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                        constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                        float swap_accum[kSwapAccum];

                        if (is_linear1_phase) {
                            const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;

                            #pragma unroll
                            for (uint32_t k_plane = 0; k_plane < kNumTMATilesPerStage; ++ k_plane) {
                                const uint32_t sf_k = k_block_idx * kNumTMATilesPerStage + k_plane;
                                const float gate_sf = __ldg(base + gate_n * kL1SFKBlocks + sf_k);
                                const float up_sf   = __ldg(base + up_n * kL1SFKBlocks + sf_k);

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < kTMATileK / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + k_plane * LOAD_BLOCK_N * kTMATileK +
                                        wg_n_idx * kTMATileK + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k_plane * LOAD_BLOCK_M * kTMATileK +
                                        k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + k_plane * kL2SFAHalfStride + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + k_plane * kL2SFAHalfStride + token_1) : 0.0f;
                                    if constexpr (kBF16ScaledAccum) {
                                        swap_final_bf16[2 * i] = __hfma2(
                                            __floats2bfloat162_rn(swap_accum[4 * i], swap_accum[4 * i + 1]),
                                            __floats2bfloat162_rn(scale_0 * gate_sf, scale_1 * gate_sf),
                                            swap_final_bf16[2 * i]);
                                        swap_final_bf16[2 * i + 1] = __hfma2(
                                            __floats2bfloat162_rn(swap_accum[4 * i + 2], swap_accum[4 * i + 3]),
                                            __floats2bfloat162_rn(scale_0 * up_sf, scale_1 * up_sf),
                                            swap_final_bf16[2 * i + 1]);
                                    } else {
                                        final_accum[i * 4 + 0] += scale_0 * gate_sf * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * up_sf * swap_accum[i * 4 + 2];
                                        final_accum[i * 4 + 1] += scale_1 * gate_sf * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * up_sf * swap_accum[i * 4 + 3];
                                    }
                                }
                            }
                            arrive_empty_barrier(stage_idx);
                        } else {
                            const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                            const float* base = l2_weights_sf + local_expert_idx * kL2SFPerExpert;

                            #pragma unroll
                            for (uint32_t sf_group = 0; sf_group < kNumL2SFAKGroups; ++ sf_group) {
                                const uint32_t k_plane = sf_group / 2u;
                                const uint32_t k_offset = (sf_group & 1u) * kL2ActsSFGranK;
                                const uint32_t sf_k = k_block_idx * kNumTMATilesPerStage + k_plane;
                                const float l2_sf = __ldg(base + sf_n * kL2SFKBlocks + sf_k);

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < kL2ActsSFGranK / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + k_plane * LOAD_BLOCK_N * kTMATileK +
                                        wg_n_idx * kTMATileK + k_offset + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k_plane * LOAD_BLOCK_M * kTMATileK +
                                        k_offset + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_1) : 0.0f;
                                    if constexpr (kBF16ScaledAccum) {
                                        swap_final_bf16[2 * i] = __hfma2(
                                            __floats2bfloat162_rn(swap_accum[4 * i], swap_accum[4 * i + 1]),
                                            __floats2bfloat162_rn(scale_0 * l2_sf, scale_1 * l2_sf),
                                            swap_final_bf16[2 * i]);
                                        swap_final_bf16[2 * i + 1] = __hfma2(
                                            __floats2bfloat162_rn(swap_accum[4 * i + 2], swap_accum[4 * i + 3]),
                                            __floats2bfloat162_rn(scale_0 * l2_sf, scale_1 * l2_sf),
                                            swap_final_bf16[2 * i + 1]);
                                    } else {
                                        final_accum[i * 4 + 0] += scale_0 * l2_sf * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * l2_sf * swap_accum[i * 4 + 2];
                                        final_accum[i * 4 + 1] += scale_1 * l2_sf * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * l2_sf * swap_accum[i * 4 + 3];
                                    }
                                }
                            }
                            arrive_empty_barrier(stage_idx);
                        }
                    };

                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (BLOCK_M == 16) {
                            if (n_swap <= 8) run_swap_ab_bk256_step.template operator()<8>(k_block_idx);
                            else             run_swap_ab_bk256_step.template operator()<16>(k_block_idx);
                        } else if constexpr (BLOCK_M == 32) {
                            if (n_swap <= 8)       run_swap_ab_bk256_step.template operator()<8>(k_block_idx);
                            else if (n_swap <= 16) run_swap_ab_bk256_step.template operator()<16>(k_block_idx);
                            else if (n_swap <= 24) run_swap_ab_bk256_step.template operator()<24>(k_block_idx);
                            else                   run_swap_ab_bk256_step.template operator()<32>(k_block_idx);
                        } else {
                            switch (n_swap) {
                                case 8:  run_swap_ab_bk256_step.template operator()<8>(k_block_idx);  break;
                                case 16: run_swap_ab_bk256_step.template operator()<16>(k_block_idx); break;
                                case 24: run_swap_ab_bk256_step.template operator()<24>(k_block_idx); break;
                                case 32: run_swap_ab_bk256_step.template operator()<32>(k_block_idx); break;
                                case 40: run_swap_ab_bk256_step.template operator()<40>(k_block_idx); break;
                                case 48: run_swap_ab_bk256_step.template operator()<48>(k_block_idx); break;
                                case 56: run_swap_ab_bk256_step.template operator()<56>(k_block_idx); break;
                                default: run_swap_ab_bk256_step.template operator()<64>(k_block_idx); break;
                            }
                        }
                    }
                    }

                    if constexpr (kBF16ScaledAccum) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                            const float2 pair = __bfloat1622float2(swap_final_bf16[i]);
                            final_accum[2 * i] = pair.x;
                            final_accum[2 * i + 1] = pair.y;
                        }
                    }
                } else if constexpr (BLOCK_K == 256) {
                    constexpr uint32_t kL1SFKBlocks = kHidden / kGranK;
                    constexpr uint32_t kL2SFKBlocks =
                        kIntermediateHidden / kGranK;
                    constexpr uint32_t kL1SFGateBlks =
                        kIntermediateHidden / kGranK;
                    constexpr uint32_t kL1SFPerExpert =
                        (kIntermediateHidden * 2 / kGranK) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert =
                        (kHidden / kGranK) * kL2SFKBlocks;
                    nv_bfloat162 final_bf16[kAccumPerThread / 2];
                    if constexpr (kBF16ScaledAccum) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                            final_bf16[i] = __float2bfloat162_rn(0.0f);
                    }

                    const auto issue_wgmma = [&]<uint32_t kNumWGMMAs>(
                                                     const uint32_t& k_plane,
                                                     const uint32_t& k_offset) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] +
                                    k_plane * LOAD_BLOCK_M * kTMATileK +
                                    row_block_offset * kTMATileK + k_offset +
                                    k * WGMMA::K,
                                1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] +
                                    k_plane * LOAD_BLOCK_N * kTMATileK +
                                    wg_n_idx * kTMATileK + k_offset +
                                    k * WGMMA::K,
                                1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k != 0);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                    };

                    const auto promote_l1 = [&](const float& scale_a_0,
                                                 const float& scale_a_1,
                                                 const float& gate_sf,
                                                 const float& up_sf) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float weight_sf = (i & 1u) ? up_sf : gate_sf;
                            if constexpr (kBF16ScaledAccum) {
                                const nv_bfloat162 scale0 =
                                    __float2bfloat162_rn(scale_a_0 * weight_sf);
                                const nv_bfloat162 scale1 =
                                    __float2bfloat162_rn(scale_a_1 * weight_sf);
                                final_bf16[2 * i] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i], accum[4 * i + 1]),
                                    scale0, final_bf16[2 * i]);
                                final_bf16[2 * i + 1] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i + 2], accum[4 * i + 3]),
                                    scale1, final_bf16[2 * i + 1]);
                            } else {
                                final_accum[i*4+0] +=
                                    scale_a_0 * weight_sf * accum[i*4+0];
                                final_accum[i*4+1] +=
                                    scale_a_0 * weight_sf * accum[i*4+1];
                                final_accum[i*4+2] +=
                                    scale_a_1 * weight_sf * accum[i*4+2];
                                final_accum[i*4+3] +=
                                    scale_a_1 * weight_sf * accum[i*4+3];
                            }
                        }
                    };
                    const auto promote_l2 = [&](const float& scale_a_0,
                                                 const float& scale_a_1,
                                                 const float& l2_sf_lo,
                                                 const float& l2_sf_hi) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float weight_sf =
                                (WG_BLOCK_N > 128 and i >= 16u) ?
                                    l2_sf_hi : l2_sf_lo;
                            if constexpr (kBF16ScaledAccum) {
                                const nv_bfloat162 scale0 =
                                    __float2bfloat162_rn(scale_a_0 * weight_sf);
                                const nv_bfloat162 scale1 =
                                    __float2bfloat162_rn(scale_a_1 * weight_sf);
                                final_bf16[2 * i] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i], accum[4 * i + 1]),
                                    scale0, final_bf16[2 * i]);
                                final_bf16[2 * i + 1] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i + 2], accum[4 * i + 3]),
                                    scale1, final_bf16[2 * i + 1]);
                            } else {
                                final_accum[i*4+0] +=
                                    scale_a_0 * weight_sf * accum[i*4+0];
                                final_accum[i*4+1] +=
                                    scale_a_0 * weight_sf * accum[i*4+1];
                                final_accum[i*4+2] +=
                                    scale_a_1 * weight_sf * accum[i*4+2];
                                final_accum[i*4+3] +=
                                    scale_a_1 * weight_sf * accum[i*4+3];
                            }
                        }
                    };

                    if constexpr (kAgyPipeWgmma and not MegaMoEPhase::is_fused and
                                  kBF16ScaledAccum and WG_BLOCK_N == 128 and
                                  kNumStages >= 2 and kNumEpilogueRegisters >= 208) {
                        float accum_1[kAccumPerThread];
                        const auto issue_async_bk256 =
                            [&]<uint32_t kPlane, uint32_t kOffset, uint32_t kNumWGMMAs>(
                                const uint32_t& s_idx, float* acc) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[s_idx] +
                                            kPlane * LOAD_BLOCK_M * kTMATileK +
                                            row_block_offset * kTMATileK + kOffset +
                                            k * WGMMA::K,
                                        1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[s_idx] +
                                            kPlane * LOAD_BLOCK_N * kTMATileK +
                                            wg_n_idx * kTMATileK + kOffset +
                                            k * WGMMA::K,
                                        1);
                                    WGMMA::wgmma(desc_a, desc_b, acc, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                            };

                        const auto promote_l1_buf = [&](float* acc,
                                                        const float& sa0, const float& sa1,
                                                        const float& g_sf, const float& u_sf) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(acc[i]);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float w_sf = (i & 1u) ? u_sf : g_sf;
                                const nv_bfloat162 s0 = __float2bfloat162_rn(sa0 * w_sf);
                                const nv_bfloat162 s1 = __float2bfloat162_rn(sa1 * w_sf);
                                final_bf16[2 * i] = __hfma2(
                                    __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]), s0, final_bf16[2 * i]);
                                final_bf16[2 * i + 1] = __hfma2(
                                    __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]), s1, final_bf16[2 * i + 1]);
                            }
                        };

                        const auto promote_l2_buf = [&](float* acc,
                                                        const float& sa0, const float& sa1,
                                                        const float& w_sf) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(acc[i]);
                            const nv_bfloat162 s0 = __float2bfloat162_rn(sa0 * w_sf);
                            const nv_bfloat162 s1 = __float2bfloat162_rn(sa1 * w_sf);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_bf16[2 * i] = __hfma2(
                                    __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]), s0, final_bf16[2 * i]);
                                final_bf16[2 * i + 1] = __hfma2(
                                    __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]), s1, final_bf16[2 * i + 1]);
                            }
                        };

                        if (is_linear1_phase) {
                            const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base     = l1_weights_sf + local_expert_idx * kL1SFPerExpert;
                            const float* wsf_gate = base + gate_n * kL1SFKBlocks;
                            const float* wsf_up   = base + up_n * kL1SFKBlocks;

                            uint32_t k_block_idx = 0;
                            full_barriers[stage_idx]->wait(phase);
                            uint32_t prev_s = stage_idx;
                            const float sa0_p0 = ptx::ld_shared(smem_sfa[prev_s] + row_offset_r0);
                            const float sa1_p0 = ptx::ld_shared(smem_sfa[prev_s] + row_offset_r1);
                            float prev_sa0_p1  = ptx::ld_shared(smem_sfa[prev_s] + kL2SFAHalfStride + row_offset_r0);
                            float prev_sa1_p1  = ptx::ld_shared(smem_sfa[prev_s] + kL2SFAHalfStride + row_offset_r1);
                            const float g_p0   = __ldg(wsf_gate + 0u);
                            const float u_p0   = __ldg(wsf_up + 0u);
                            float prev_g_p1    = __ldg(wsf_gate + 1u);
                            float prev_u_p1    = __ldg(wsf_up + 1u);

                            issue_async_bk256.template operator()<0u, 0u, kTMATileK / WGMMA::K>(prev_s, accum);
                            issue_async_bk256.template operator()<1u, 0u, kTMATileK / WGMMA::K>(prev_s, accum_1);
                            ptx::warpgroup_wait<1>();
                            promote_l1_buf(accum, sa0_p0, sa1_p0, g_p0, u_p0);
                            advance_pipeline(k_block_idx);

                            for (; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                                full_barriers[stage_idx]->wait(phase);
                                const uint32_t curr_s = stage_idx;
                                const uint32_t sf_k = k_block_idx * kNumTMATilesPerStage;
                                const float curr_sa0_p0 = ptx::ld_shared(smem_sfa[curr_s] + row_offset_r0);
                                const float curr_sa1_p0 = ptx::ld_shared(smem_sfa[curr_s] + row_offset_r1);
                                const float curr_sa0_p1 = ptx::ld_shared(smem_sfa[curr_s] + kL2SFAHalfStride + row_offset_r0);
                                const float curr_sa1_p1 = ptx::ld_shared(smem_sfa[curr_s] + kL2SFAHalfStride + row_offset_r1);
                                const float curr_g_p0   = __ldg(wsf_gate + sf_k + 0u);
                                const float curr_u_p0   = __ldg(wsf_up + sf_k + 0u);
                                const float curr_g_p1   = __ldg(wsf_gate + sf_k + 1u);
                                const float curr_u_p1   = __ldg(wsf_up + sf_k + 1u);

                                issue_async_bk256.template operator()<0u, 0u, kTMATileK / WGMMA::K>(curr_s, accum);
                                ptx::warpgroup_wait<1>();
                                arrive_empty_barrier(prev_s);
                                promote_l1_buf(accum_1, prev_sa0_p1, prev_sa1_p1, prev_g_p1, prev_u_p1);

                                issue_async_bk256.template operator()<1u, 0u, kTMATileK / WGMMA::K>(curr_s, accum_1);
                                ptx::warpgroup_wait<1>();
                                promote_l1_buf(accum, curr_sa0_p0, curr_sa1_p0, curr_g_p0, curr_u_p0);

                                prev_s      = curr_s;
                                prev_sa0_p1 = curr_sa0_p1;
                                prev_sa1_p1 = curr_sa1_p1;
                                prev_g_p1   = curr_g_p1;
                                prev_u_p1   = curr_u_p1;
                            }
                            ptx::warpgroup_wait<0>();
                            arrive_empty_barrier(prev_s);
                            promote_l1_buf(accum_1, prev_sa0_p1, prev_sa1_p1, prev_g_p1, prev_u_p1);
                        } else {
                            const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                            const float* wsf_col = l2_weights_sf +
                                local_expert_idx * kL2SFPerExpert + sf_n * kL2SFKBlocks;

                            uint32_t k_block_idx = 0;
                            full_barriers[stage_idx]->wait(phase);
                            uint32_t prev_s = stage_idx;
                            const float sa0_0 = ptx::ld_shared(smem_sfa[prev_s] + 0u * kL2SFAHalfStride + row_offset_r0);
                            const float sa1_0 = ptx::ld_shared(smem_sfa[prev_s] + 0u * kL2SFAHalfStride + row_offset_r1);
                            const float sa0_1 = ptx::ld_shared(smem_sfa[prev_s] + 1u * kL2SFAHalfStride + row_offset_r0);
                            const float sa1_1 = ptx::ld_shared(smem_sfa[prev_s] + 1u * kL2SFAHalfStride + row_offset_r1);
                            const float sa0_2 = ptx::ld_shared(smem_sfa[prev_s] + 2u * kL2SFAHalfStride + row_offset_r0);
                            const float sa1_2 = ptx::ld_shared(smem_sfa[prev_s] + 2u * kL2SFAHalfStride + row_offset_r1);
                            float prev_sa0_3  = ptx::ld_shared(smem_sfa[prev_s] + 3u * kL2SFAHalfStride + row_offset_r0);
                            float prev_sa1_3  = ptx::ld_shared(smem_sfa[prev_s] + 3u * kL2SFAHalfStride + row_offset_r1);
                            const float wsf_p0 = __ldg(wsf_col + 0u);
                            float prev_wsf_p1  = __ldg(wsf_col + 1u);

                            issue_async_bk256.template operator()<0u, 0u, kL2ActsSFGranK / WGMMA::K>(prev_s, accum);
                            issue_async_bk256.template operator()<0u, kL2ActsSFGranK, kL2ActsSFGranK / WGMMA::K>(prev_s, accum_1);
                            ptx::warpgroup_wait<1>();
                            promote_l2_buf(accum, sa0_0, sa1_0, wsf_p0);

                            issue_async_bk256.template operator()<1u, 0u, kL2ActsSFGranK / WGMMA::K>(prev_s, accum);
                            ptx::warpgroup_wait<1>();
                            promote_l2_buf(accum_1, sa0_1, sa1_1, wsf_p0);

                            issue_async_bk256.template operator()<1u, kL2ActsSFGranK, kL2ActsSFGranK / WGMMA::K>(prev_s, accum_1);
                            ptx::warpgroup_wait<1>();
                            promote_l2_buf(accum, sa0_2, sa1_2, prev_wsf_p1);
                            advance_pipeline(k_block_idx);

                            for (; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                                full_barriers[stage_idx]->wait(phase);
                                const uint32_t curr_s = stage_idx;
                                const uint32_t sf_k = k_block_idx * kNumTMATilesPerStage;
                                const float curr_sa0_0  = ptx::ld_shared(smem_sfa[curr_s] + 0u * kL2SFAHalfStride + row_offset_r0);
                                const float curr_sa1_0  = ptx::ld_shared(smem_sfa[curr_s] + 0u * kL2SFAHalfStride + row_offset_r1);
                                const float curr_sa0_1  = ptx::ld_shared(smem_sfa[curr_s] + 1u * kL2SFAHalfStride + row_offset_r0);
                                const float curr_sa1_1  = ptx::ld_shared(smem_sfa[curr_s] + 1u * kL2SFAHalfStride + row_offset_r1);
                                const float curr_sa0_2  = ptx::ld_shared(smem_sfa[curr_s] + 2u * kL2SFAHalfStride + row_offset_r0);
                                const float curr_sa1_2  = ptx::ld_shared(smem_sfa[curr_s] + 2u * kL2SFAHalfStride + row_offset_r1);
                                const float curr_sa0_3  = ptx::ld_shared(smem_sfa[curr_s] + 3u * kL2SFAHalfStride + row_offset_r0);
                                const float curr_sa1_3  = ptx::ld_shared(smem_sfa[curr_s] + 3u * kL2SFAHalfStride + row_offset_r1);
                                const float curr_wsf_p0 = __ldg(wsf_col + sf_k + 0u);
                                const float curr_wsf_p1 = __ldg(wsf_col + sf_k + 1u);

                                issue_async_bk256.template operator()<0u, 0u, kL2ActsSFGranK / WGMMA::K>(curr_s, accum);
                                ptx::warpgroup_wait<1>();
                                arrive_empty_barrier(prev_s);
                                promote_l2_buf(accum_1, prev_sa0_3, prev_sa1_3, prev_wsf_p1);

                                issue_async_bk256.template operator()<0u, kL2ActsSFGranK, kL2ActsSFGranK / WGMMA::K>(curr_s, accum_1);
                                ptx::warpgroup_wait<1>();
                                promote_l2_buf(accum, curr_sa0_0, curr_sa1_0, curr_wsf_p0);

                                issue_async_bk256.template operator()<1u, 0u, kL2ActsSFGranK / WGMMA::K>(curr_s, accum);
                                ptx::warpgroup_wait<1>();
                                promote_l2_buf(accum_1, curr_sa0_1, curr_sa1_1, curr_wsf_p0);

                                issue_async_bk256.template operator()<1u, kL2ActsSFGranK, kL2ActsSFGranK / WGMMA::K>(curr_s, accum_1);
                                ptx::warpgroup_wait<1>();
                                promote_l2_buf(accum, curr_sa0_2, curr_sa1_2, curr_wsf_p1);

                                prev_s      = curr_s;
                                prev_sa0_3  = curr_sa0_3;
                                prev_sa1_3  = curr_sa1_3;
                                prev_wsf_p1 = curr_wsf_p1;
                            }
                            ptx::warpgroup_wait<0>();
                            arrive_empty_barrier(prev_s);
                            promote_l2_buf(accum_1, prev_sa0_3, prev_sa1_3, prev_wsf_p1);
                        }
                    } else {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);

                        if (is_linear1_phase) {
                            const uint32_t gate_n =
                                (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                            const uint32_t up_n = kL1SFGateBlks + gate_n;
                            const float* base = l1_weights_sf +
                                local_expert_idx * kL1SFPerExpert;
                            #pragma unroll
                            for (uint32_t k_plane = 0;
                                 k_plane < kNumTMATilesPerStage; ++ k_plane) {
                                const float scale_a_0 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        k_plane * kL2SFAHalfStride + row_offset_r0);
                                const float scale_a_1 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        k_plane * kL2SFAHalfStride + row_offset_r1);
                                const uint32_t sf_k =
                                    k_block_idx * kNumTMATilesPerStage + k_plane;
                                const float gate_sf =
                                    __ldg(base + gate_n * kL1SFKBlocks + sf_k);
                                const float up_sf =
                                    __ldg(base + up_n * kL1SFKBlocks + sf_k);
                                issue_wgmma.template operator()<
                                    kTMATileK / WGMMA::K>(k_plane, 0);
                                promote_l1(
                                    scale_a_0, scale_a_1, gate_sf, up_sf);
                            }
                        } else {
                            const uint32_t sf_n =
                                (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                            const float* base = l2_weights_sf +
                                local_expert_idx * kL2SFPerExpert;
                            #pragma unroll
                            for (uint32_t sf_group = 0;
                                 sf_group < kNumL2SFAKGroups; ++ sf_group) {
                                const uint32_t k_plane = sf_group / 2u;
                                const uint32_t k_offset =
                                    (sf_group & 1u) * kL2ActsSFGranK;
                                const uint32_t sf_k =
                                    k_block_idx * kNumTMATilesPerStage + k_plane;
                                const float scale_a_0 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        sf_group * kL2SFAHalfStride + row_offset_r0);
                                const float scale_a_1 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        sf_group * kL2SFAHalfStride + row_offset_r1);
                                const float l2_sf_lo =
                                    __ldg(base + sf_n * kL2SFKBlocks + sf_k);
                                const float l2_sf_hi = WG_BLOCK_N > 128 ?
                                    __ldg(base +
                                          (sf_n + 1u) * kL2SFKBlocks + sf_k) :
                                    l2_sf_lo;
                                issue_wgmma.template operator()<
                                    kL2ActsSFGranK / WGMMA::K>(
                                        k_plane, k_offset);
                                promote_l2(scale_a_0, scale_a_1,
                                           l2_sf_lo, l2_sf_hi);
                            }
                        }
                        arrive_empty_barrier(stage_idx);
                    }
                    }

                    if constexpr (kBF16ScaledAccum) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                            const float2 pair = __bfloat1622float2(final_bf16[i]);
                            final_accum[2 * i] = pair.x;
                            final_accum[2 * i + 1] = pair.y;
                        }
                    }
                }
            };

            const auto run_bf16_scaled_accum_gemm_loop = [&]() {
                DG_STATIC_ASSERT(not kBF16ScaledAccum or
                                 kSwapABActive or
                                 (WG_BLOCK_M == 64 and
                                  (WG_BLOCK_N == 128 or WG_BLOCK_N == 256) and
                                  not kSwapABActive),
                                 "BF16 scaled accumulation expects swap-AB or non-swap M64N128/N256");
                nv_bfloat162 final_bf16[kAccumPerThread / 2];
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                    final_bf16[i] = __float2bfloat162_rn(0.0f);

                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert =
                    (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert =
                    (kHidden / 128) * kL2SFKBlocks;

                const auto issue_wgmma = [&]<uint32_t kOffset,
                                              uint32_t kNumWGMMAs>() {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                kOffset + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K +
                                kOffset + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();
                };

                const auto promote_l1 = [&](const float& scale_a_0,
                                             const float& scale_a_1,
                                             const float& gate_sf,
                                             const float& up_sf) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float weight_sf = (i & 1u) ? up_sf : gate_sf;
                        const nv_bfloat162 scale0 =
                            __float2bfloat162_rn(scale_a_0 * weight_sf);
                        const nv_bfloat162 scale1 =
                            __float2bfloat162_rn(scale_a_1 * weight_sf);
                        final_bf16[2 * i] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i], accum[4 * i + 1]),
                            scale0, final_bf16[2 * i]);
                        final_bf16[2 * i + 1] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i + 2], accum[4 * i + 3]),
                            scale1, final_bf16[2 * i + 1]);
                    }
                };
                const auto promote_l2 = [&](const float& scale_a_0,
                                             const float& scale_a_1,
                                             const float& weight_sf_lo,
                                             const float& weight_sf_hi) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const bool high_n = WG_BLOCK_N > 128 and i >= 16u;
                        const float weight_sf = high_n ? weight_sf_hi : weight_sf_lo;
                        const nv_bfloat162 scale0 =
                            __float2bfloat162_rn(scale_a_0 * weight_sf);
                        const nv_bfloat162 scale1 =
                            __float2bfloat162_rn(scale_a_1 * weight_sf);
                        final_bf16[2 * i] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i], accum[4 * i + 1]),
                            scale0, final_bf16[2 * i]);
                        final_bf16[2 * i + 1] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i + 2], accum[4 * i + 3]),
                            scale1, final_bf16[2 * i + 1]);
                    }
                };

                if constexpr (kAgyPipeWgmma and WG_BLOCK_N == 128 and BLOCK_K == 128 and kNumStages >= 3) {
                    float accum_1[kAccumPerThread];
                    const auto issue_async = [&]<uint32_t kOff, uint32_t kCnt>(
                                                 const uint32_t& s_idx, float* acc) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(acc[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < kCnt; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[s_idx] + row_block_offset * BLOCK_K + kOff + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[s_idx] + wg_n_idx * BLOCK_K + kOff + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, acc, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(acc[i]);
                    };
                    const auto promote_l1_buf = [&](float* acc,
                                                    const float& sa0, const float& sa1,
                                                    const float& g_sf, const float& u_sf) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(acc[i]);
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float w_sf = (i & 1u) ? u_sf : g_sf;
                            const nv_bfloat162 s0 = __float2bfloat162_rn(sa0 * w_sf);
                            const nv_bfloat162 s1 = __float2bfloat162_rn(sa1 * w_sf);
                            final_bf16[2 * i] = __hfma2(
                                __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]), s0, final_bf16[2 * i]);
                            final_bf16[2 * i + 1] = __hfma2(
                                __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]), s1, final_bf16[2 * i + 1]);
                        }
                    };
                    const auto promote_l2_buf = [&](float* acc,
                                                    const float& sa0, const float& sa1,
                                                    const float& w_sf) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(acc[i]);
                        const nv_bfloat162 s0 = __float2bfloat162_rn(sa0 * w_sf);
                        const nv_bfloat162 s1 = __float2bfloat162_rn(sa1 * w_sf);
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_bf16[2 * i] = __hfma2(
                                __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]), s0, final_bf16[2 * i]);
                            final_bf16[2 * i + 1] = __hfma2(
                                __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]), s1, final_bf16[2 * i + 1]);
                        }
                    };

                    if (is_linear1_phase) {
                        const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                        const uint32_t up_n = kL1SFGateBlks + gate_n;
                        const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;

                        uint32_t k_block_idx = 0;
                        full_barriers[stage_idx]->wait(phase);
                        uint32_t s0 = stage_idx;
                        float sa0_0 = ptx::ld_shared(smem_sfa[s0] + row_offset_r0);
                        float sa1_0 = ptx::ld_shared(smem_sfa[s0] + row_offset_r1);
                        float g0 = __ldg(base + gate_n * kL1SFKBlocks + 0);
                        float u0 = __ldg(base + up_n * kL1SFKBlocks + 0);
                        issue_async.template operator()<0, BLOCK_K / WGMMA::K>(s0, accum);
                        advance_pipeline(k_block_idx);

                        const uint32_t num_pairs = num_k_blocks / 2u;
                        uint32_t s1 = 0;
                        float sa0_1 = 0.0f, sa1_1 = 0.0f, g1 = 0.0f, u1 = 0.0f;
                        for (uint32_t pair = 0; ; ++ pair) {
                            full_barriers[stage_idx]->wait(phase);
                            s1 = stage_idx;
                            sa0_1 = ptx::ld_shared(smem_sfa[s1] + row_offset_r0);
                            sa1_1 = ptx::ld_shared(smem_sfa[s1] + row_offset_r1);
                            g1 = __ldg(base + gate_n * kL1SFKBlocks + k_block_idx);
                            u1 = __ldg(base + up_n * kL1SFKBlocks + k_block_idx);
                            issue_async.template operator()<0, BLOCK_K / WGMMA::K>(s1, accum_1);
                            advance_pipeline(k_block_idx);

                            ptx::warpgroup_wait<1>();
                            arrive_empty_barrier(s0);
                            promote_l1_buf(accum, sa0_0, sa1_0, g0, u0);

                            if (pair + 1u == num_pairs) break;

                            full_barriers[stage_idx]->wait(phase);
                            s0 = stage_idx;
                            sa0_0 = ptx::ld_shared(smem_sfa[s0] + row_offset_r0);
                            sa1_0 = ptx::ld_shared(smem_sfa[s0] + row_offset_r1);
                            g0 = __ldg(base + gate_n * kL1SFKBlocks + k_block_idx);
                            u0 = __ldg(base + up_n * kL1SFKBlocks + k_block_idx);
                            issue_async.template operator()<0, BLOCK_K / WGMMA::K>(s0, accum);
                            advance_pipeline(k_block_idx);

                            ptx::warpgroup_wait<1>();
                            arrive_empty_barrier(s1);
                            promote_l1_buf(accum_1, sa0_1, sa1_1, g1, u1);
                        }
                        ptx::warpgroup_wait<0>();
                        arrive_empty_barrier(s1);
                        promote_l1_buf(accum_1, sa0_1, sa1_1, g1, u1);
                    } else {
                        const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                        const float* base = l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_n * kL2SFKBlocks;

                        uint32_t k_block_idx = 0;
                        full_barriers[stage_idx]->wait(phase);
                        uint32_t prev_s = stage_idx;
                        const float sa0_lo_0 = ptx::ld_shared(smem_sfa[prev_s] + row_offset_r0);
                        const float sa1_lo_0 = ptx::ld_shared(smem_sfa[prev_s] + row_offset_r1);
                        float prev_sa0_hi = ptx::ld_shared(smem_sfa[prev_s] + kL2SFAHalfStride + row_offset_r0);
                        float prev_sa1_hi = ptx::ld_shared(smem_sfa[prev_s] + kL2SFAHalfStride + row_offset_r1);
                        float prev_wsf = __ldg(base + 0);

                        issue_async.template operator()<0, (BLOCK_K / 2) / WGMMA::K>(prev_s, accum);
                        issue_async.template operator()<BLOCK_K / 2, (BLOCK_K / 2) / WGMMA::K>(prev_s, accum_1);
                        ptx::warpgroup_wait<1>();
                        promote_l2_buf(accum, sa0_lo_0, sa1_lo_0, prev_wsf);
                        advance_pipeline(k_block_idx);

                        for (; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                            full_barriers[stage_idx]->wait(phase);
                            const uint32_t curr_s = stage_idx;
                            const float curr_sa0_lo = ptx::ld_shared(smem_sfa[curr_s] + row_offset_r0);
                            const float curr_sa1_lo = ptx::ld_shared(smem_sfa[curr_s] + row_offset_r1);
                            const float curr_sa0_hi = ptx::ld_shared(smem_sfa[curr_s] + kL2SFAHalfStride + row_offset_r0);
                            const float curr_sa1_hi = ptx::ld_shared(smem_sfa[curr_s] + kL2SFAHalfStride + row_offset_r1);
                            const float curr_wsf = __ldg(base + k_block_idx);

                            issue_async.template operator()<0, (BLOCK_K / 2) / WGMMA::K>(curr_s, accum);
                            ptx::warpgroup_wait<1>();
                            arrive_empty_barrier(prev_s);
                            promote_l2_buf(accum_1, prev_sa0_hi, prev_sa1_hi, prev_wsf);

                            issue_async.template operator()<BLOCK_K / 2, (BLOCK_K / 2) / WGMMA::K>(curr_s, accum_1);
                            ptx::warpgroup_wait<1>();
                            promote_l2_buf(accum, curr_sa0_lo, curr_sa1_lo, curr_wsf);

                            prev_s = curr_s;
                            prev_sa0_hi = curr_sa0_hi;
                            prev_sa1_hi = curr_sa1_hi;
                            prev_wsf = curr_wsf;
                        }
                        ptx::warpgroup_wait<0>();
                        arrive_empty_barrier(prev_s);
                        promote_l2_buf(accum_1, prev_sa0_hi, prev_sa1_hi, prev_wsf);
                    }
                } else {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    const float scale_a_0_lo =
                        ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    const float scale_a_1_lo =
                        ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);

                    if (is_linear1_phase) {
                        const uint32_t gate_n =
                            (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                        const uint32_t up_n = kL1SFGateBlks + gate_n;
                        const float* base = l1_weights_sf +
                            local_expert_idx * kL1SFPerExpert + k_block_idx;
                        const float gate_sf =
                            __ldg(base + gate_n * kL1SFKBlocks);
                        const float up_sf =
                            __ldg(base + up_n * kL1SFKBlocks);
                        issue_wgmma.template operator()<
                            0, BLOCK_K / WGMMA::K>();
                        arrive_empty_barrier(stage_idx);
                        promote_l1(
                            scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);
                    } else {
                        const float scale_a_0_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + kL2SFAHalfStride +
                                row_offset_r0);
                        const float scale_a_1_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + kL2SFAHalfStride +
                                row_offset_r1);
                        const uint32_t sf_n =
                            (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                        const float* base = l2_weights_sf +
                            local_expert_idx * kL2SFPerExpert + k_block_idx;
                        const float l2_sf_lo =
                            __ldg(base + sf_n * kL2SFKBlocks);
                        const float l2_sf_hi = WG_BLOCK_N > 128 ?
                            __ldg(base + (sf_n + 1u) * kL2SFKBlocks) :
                            l2_sf_lo;
                        issue_wgmma.template operator()<
                            0, (BLOCK_K / 2) / WGMMA::K>();
                        promote_l2(
                            scale_a_0_lo, scale_a_1_lo, l2_sf_lo, l2_sf_hi);
                        issue_wgmma.template operator()<
                            BLOCK_K / 2, (BLOCK_K / 2) / WGMMA::K>();
                        arrive_empty_barrier(stage_idx);
                        promote_l2(
                            scale_a_0_hi, scale_a_1_hi, l2_sf_lo, l2_sf_hi);
                    }
                }
                }

                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                    const float2 pair = __bfloat1622float2(final_bf16[i]);
                    final_accum[2 * i] = pair.x;
                    final_accum[2 * i + 1] = pair.y;
                }
            };

            AGY_PROF_TILE_MMA_BEGIN(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx,
                                    is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
            if constexpr (BLOCK_K == 256) {
                run_bk256_gemm_loop();
            } else if constexpr (kBF16ScaledAccum and not kSwapABActive) {
                run_bf16_scaled_accum_gemm_loop();
            } else {
                run_default_gemm_loop();
            }
            AGY_PROF_TILE_MMA_END(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx,
                                  is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);

            // Skip epilogue when block is past valid M (when kAgyInterleaveActive or kScatDispActive on L2, fall through cleanly!).
            if (row_block_offset >= valid_m and not kAgyInterleaveActive and not (not is_linear1_phase and kScatDispActive)) {
                if ((is_linear1_phase and kWarpgroupSplitM > 1 and not kSplitNWarpgroups) or
                    (not is_linear1_phase and (kDirectL2Scatter or kAgyFastSwapL2))) {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                return;
            }

            AGY_PROF_TILE_STORE_BEGIN(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx,
                                      is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (is_linear1_phase) {
                if constexpr (kScatDispActive and (kScatL2RawBytes > SMEM_DISPATCH_TOTAL_SIZE)) {
                    if (scat_ever_used and not scat_slot_free) {
                        sm90_fp8_mega_moe_scat_bar_wait(scat_free_bar, math_scat_phase);
                        scat_slot_free = true;
                    }
                }
                if (row_block_offset < valid_m) {
                if constexpr (kSwapABActive) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    const uint32_t out_col_base = wg_l1_out_n_idx + warp_idx_in_wg * 8 + row_idx;
#if defined(AGY_FAST_SWAP_EPI) && AGY_FAST_SWAP_EPI
                    const uint32_t store_col_swz = out_col_base ^ (col_idx * 8u);
#else
                    const uint32_t store_col_swz = out_col_base;
#endif
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
#if defined(AGY_FAST_SWAP_EPI) && AGY_FAST_SWAP_EPI
                            const float weight_0 = __ldg(l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_0)
                                .template get_base_ptr<float>());
#else
                            const float weight_0 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_0)
                                .template get_base_ptr<float>();
#endif
                            smem_cd_swap_l1_fp32[token_0 * L1_OUT_BLOCK_N + store_col_swz] =
                                silu(g0) * u0 * weight_0;
                        }
                        if (token_1 < valid_m) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
#if defined(AGY_FAST_SWAP_EPI) && AGY_FAST_SWAP_EPI
                            const float weight_1 = __ldg(l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_1)
                                .template get_base_ptr<float>());
#else
                            const float weight_1 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_1)
                                .template get_base_ptr<float>();
#endif
                            smem_cd_swap_l1_fp32[token_1 * L1_OUT_BLOCK_N + store_col_swz] =
                                silu(g1) * u1 * weight_1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

#if defined(AGY_FAST_SWAP_EPI) && AGY_FAST_SWAP_EPI
                    {
                        // kSwapSFSubBlocks: one activation scale per 64 output columns,
                        // matching kL2ActsSFGranK. BLOCK_N=128 -> 1, BLOCK_N=256 -> 2.
                        constexpr uint32_t kSwapSFSubBlocks = L1_OUT_BLOCK_N / kL2ActsSFGranK;
                        constexpr uint32_t kNumSubwarps = kNumEpilogueThreads / 16u;
                        DG_STATIC_ASSERT(L1_OUT_BLOCK_N % 64u == 0u,
                                         "swapAB L1_OUT_BLOCK_N must be a multiple of 64");
                        DG_STATIC_ASSERT(kNumEpilogueThreads % 16u == 0u,
                                         "Epilogue threads must form whole 16-lane subwarps");
                        DG_STATIC_ASSERT(SMEM_CD_SWAP_L1_FP32_SIZE % 16u == 0u,
                                         "smem_cd_swap_l1_fp8 base must be 16-byte aligned");

                        const uint32_t subwarp_id = epilogue_thread_idx / 16u;
                        const uint32_t lane16 = lane_idx & 15u;
                        const uint32_t half_mask = 0xffffu << (lane_idx & 16u);
                        const uint32_t total_groups = valid_m * kSwapSFSubBlocks;
                        auto* sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();

                        for (uint32_t work_idx = subwarp_id; work_idx < total_groups; work_idx += kNumSubwarps) {
                            const uint32_t token = (kSwapSFSubBlocks == 1u) ? work_idx : (work_idx / kSwapSFSubBlocks);
                            const uint32_t group = (kSwapSFSubBlocks == 1u) ? 0u : (work_idx % kSwapSFSubBlocks);
                            const uint32_t col_base = group * 64u + lane16 * 4u;
                            const uint32_t swz_col = col_base ^ (((token >> 1) & 3u) << 3);
                            const float4 vals = *reinterpret_cast<const float4*>(
                                smem_cd_swap_l1_fp32 + token * L1_OUT_BLOCK_N + swz_col);

                            float amax = cute::max(
                                cute::max(cute::abs(vals.x), cute::abs(vals.y)),
                                cute::max(cute::abs(vals.z), cute::abs(vals.w)));
                            #pragma unroll
                            for (int offset = 8; offset >= 1; offset >>= 1) {
                                amax = cute::max(amax, __shfl_xor_sync(half_mask, amax, offset));
                            }

                            float2 amax_pair = {amax, amax};
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
                                amax_pair, sf_pair, sf_inv_pair);
                            const float sf = sf_pair.x;
                            const float sf_inv = sf_inv_pair.x;

                            if (lane16 == 0u) {
                                const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                                const uint32_t sf_group = n_block_idx * kSwapSFSubBlocks + group;
                                sf_base_ptr[sf_group * kSFPoolStrideTokens + token_idx] = sf;
                            }

                            const __nv_fp8x2_e4m3 pair_lo(make_float2(vals.x * sf_inv, vals.y * sf_inv));
                            const __nv_fp8x2_e4m3 pair_hi(make_float2(vals.z * sf_inv, vals.w * sf_inv));
                            const uint32_t packed_fp8 =
                                static_cast<uint32_t>(pair_lo.__x) |
                                (static_cast<uint32_t>(pair_hi.__x) << 16);
                            *reinterpret_cast<uint32_t*>(
                                smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col_base) = packed_fp8;
                        }
                        cute::tma_store_fence();
                    }
#else
                    for (uint32_t token = epilogue_thread_idx; token < valid_m; token += kNumEpilogueThreads) {
                        #pragma unroll
                        for (uint32_t group = 0; group < L1_OUT_BLOCK_N / 64u; ++ group) {
                            float amax = 0.0f;
                            #pragma unroll
                            for (uint32_t c = 0; c < 64u; ++ c) {
                                const float v = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + group * 64u + c];
                                amax = cute::max(amax, cute::abs(v));
                            }
                            float2 amax_pair = {amax, amax};
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
                                amax_pair, sf_pair, sf_inv_pair);
                            const float sf = sf_pair.x;
                            const float sf_inv = sf_inv_pair.x;

                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                            const uint32_t sf_group = n_block_idx * (L1_OUT_BLOCK_N / 64u) + group;
                            sf_base_ptr[sf_group * kSFPoolStrideTokens + token_idx] = sf;

                            #pragma unroll
                            for (uint32_t c = 0; c < 64u; c += 2u) {
                                const uint32_t col = group * 64u + c;
                                const float v0 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 0] * sf_inv;
                                const float v1 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 1] * sf_inv;
                                const __nv_fp8x2_e4m3 pair(make_float2(v0, v1));
                                auto* ptr = reinterpret_cast<uint16_t*>(
                                    smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                                *ptr = pair.__x;
                            }
                        }
                    }
#endif

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                        cute::tma_store_fence();
                        if constexpr (kAgyAdaptBm) {
                            const uint32_t n_m_boxes = cute::max(1u, (valid_m + 15u) >> 4);
                            for (uint32_t mbx = 0; mbx < n_m_boxes; ++ mbx) {
                                cute::SM90_TMA_STORE_2D::copy(
                                    &tensor_map_l1_output,
                                    smem_cd_swap_l1_fp8 + mbx * 16u * L1_OUT_BLOCK_N,
                                    out_n_idx,
                                    m_idx + mbx * 16u);
                            }
                        } else {
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_swap_l1_fp8,
                                out_n_idx,
                                m_idx);
                        }
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                } else {
                    // L1 epilogue: SwiGLU, FP8 quantization, and TMA store
                    // Layout in `final_accum`:
                    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                    //   Pair `p` 鈭?[0, 8): gate chunk = 2p, up chunk = 2p+1.
                    //
                    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    DG_STATIC_ASSERT(WG_L1_OUT_BLOCK_N % 64 == 0,
                                     "Each L1 consumer must cover complete 64-column SF groups");
                    constexpr uint32_t kNumSFGroups = WG_L1_OUT_BLOCK_N / 64;
                    float swiglu_r0[kNumPairs][2];
                    float swiglu_r1[kNumPairs][2];

                    // Per-row amax, one scale for each 64-col L1 output group.
                    float amax_r0[kNumSFGroups] = {};
                    float amax_r1[kNumSFGroups] = {};

                    // Compute SwiGLU + per-group amax.
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        const uint32_t sf_group = p / 8;

                        auto clamp_gate = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(x, kActivationClamp);
                        };
                        auto clamp_up = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                        };
                        float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                        float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                        float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                        float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                        float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                        float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                        float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                        float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                        auto silu = [](float x) {
                            const float e = kFastMath ? __expf(-x) : expf(-x);
                            const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                            return x * sig;
                        };

                        if (valid_r0) {
                            swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                            swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                            amax_r0[sf_group] = cute::max(
                                amax_r0[sf_group],
                                cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                        } else {
                            swiglu_r0[p][0] = 0.0f;
                            swiglu_r0[p][1] = 0.0f;
                        }
                        if (valid_r1) {
                            swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                            swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                            amax_r1[sf_group] = cute::max(
                                amax_r1[sf_group],
                                cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                        } else {
                            swiglu_r1[p][0] = 0.0f;
                            swiglu_r1[p][1] = 0.0f;
                        }
                    }

                    float weight_r0 = 0.0f, weight_r1 = 0.0f;
                    // For large-capacity specializations, cached same-address
                    // loads avoid a shuffle dependency in this register-heavy path.
                    if constexpr (kNumMaxTokensPerRank <= 1024) {
                        const int topk_weight_src_lane = static_cast<int>(lane_idx - col_idx);
                        if (col_idx == 0) {
                            weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + row_offset_r0)
                                .template get_base_ptr<float>() : 0.0f;
                            weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + row_offset_r1)
                                .template get_base_ptr<float>() : 0.0f;
                        }
                        weight_r0 = __shfl_sync(0xffffffff, weight_r0, topk_weight_src_lane);
                        weight_r1 = __shfl_sync(0xffffffff, weight_r1, topk_weight_src_lane);
                    } else {
                        weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .template get_base_ptr<float>() : 0.0f;
                        weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .template get_base_ptr<float>() : 0.0f;
                    }
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        swiglu_r0[p][0] *= weight_r0;
                        swiglu_r0[p][1] *= weight_r0;
                        swiglu_r1[p][0] *= weight_r1;
                        swiglu_r1[p][1] *= weight_r1;
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] *= cute::abs(weight_r0);
                        amax_r1[g] *= cute::abs(weight_r1);
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], sm90_moe_math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], sm90_moe_math::ReduceMax<float>());
                    }

                    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
                            amax_pair, sf_pair, sf_inv_pair);
                        sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                        sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                    }

                    // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                    auto* smem_cd_l1_wg = smem_cd_l1
                        + smem_cd_l1_wg_offset;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t sf_group = p / 8;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                        const uint32_t col = p * 8 + col_idx * 2;
                        auto* p0 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1_wg + r_0 * WG_SMEM_CD_L1_STRIDE_N +
                            (kSplitNWarpgroups ? wg_l1_out_n_idx : 0) + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N +
                            (kSplitNWarpgroups ? wg_l1_out_n_idx : 0) + col);
                        if (valid_r0)
                            *p0 = r0_pair.__x;
                        if (valid_r1)
                            *p1 = r1_pair.__x;
                    }

                    // Write L2-activation SF as float, one value per 64 output columns.
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                        const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                        const uint32_t base_k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / 64u;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            if (valid_r0)
                                sf_base_ptr[(base_k_sf_idx + g) * kSFPoolStrideTokens + token_r0] = sf_r0[g];
                            if (valid_r1)
                                sf_base_ptr[(base_k_sf_idx + g) * kSFPoolStrideTokens + token_r1] = sf_r1[g];
                        }
                    }

                    // Issue TMA store of the entire tile. Padding rows beyond
                    // `valid_m` are written with stale/garbage FP8 to the L1-output
                    // pool buffer, but they are never consumed downstream: the L2
                    // GEMM tile loads them, but its NVLink-scatter epilogue is
                    // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                    // padding rows can produce NaN accumulators that simply stay
                    // in registers (only valid rows are converted to BF16 and
                    // STSM'd into smem). Using TMA for partial tiles is a large
                    // win for low-batch / decode where every tile is partial.
                    if constexpr (kSplitNWarpgroups) {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1,
                                out_n_idx,
                                m_idx);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                        ptx::tma_store_wait<0>();
                    } else {
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1_wg,
                                out_n_idx,
                                m_idx + row_block_offset);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                        ptx::tma_store_wait<0>();
                    }
                }
                } // end if (row_block_offset < valid_m) on L1

                if constexpr (kAgyInterleaveActive) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0u) {
                        asm volatile("fence.acq_rel.gpu;" ::: "memory");
                        constexpr uint32_t kL1BlockNs = L1_SHAPE_N / BLOCK_N;
                        constexpr uint32_t kL2BlockNs = L2_SHAPE_N / BLOCK_N;
                        if constexpr (kAgyInterleaveMode == 5 or kAgyInterleaveMode == 6) {
                            uint32_t _old_mask;
                            if constexpr (L1_OUT_BLOCK_N == BLOCK_K) {
                                const uint32_t bit = 1u << n_block_idx;
                                asm volatile("atom.global.release.gpu.or.b32 %0, [%1], %2;"
                                             : "=r"(_old_mask)
                                             : "l"(interleave_ready_mb_queue + pool_block_idx), "r"(bit)
                                             : "memory");
                            } else if constexpr (L1_OUT_BLOCK_N == 2u * BLOCK_K) {
                                const uint32_t bits = 3u << (2u * n_block_idx);
                                asm volatile("atom.global.release.gpu.or.b32 %0, [%1], %2;"
                                             : "=r"(_old_mask)
                                             : "l"(interleave_ready_mb_queue + pool_block_idx), "r"(bits)
                                             : "memory");
                            } else {
                                DG_STATIC_ASSERT(2u * L1_OUT_BLOCK_N == BLOCK_K, "Unsupported L1_OUT_BLOCK_N / BLOCK_K ratio");
                                uint32_t* mask_arr = ((n_block_idx & 1u) == 0u) ? interleave_ready_mb_queue : interleave_l1_done_base;
                                const uint32_t bit = 1u << (n_block_idx >> 1);
                                asm volatile("atom.global.release.gpu.or.b32 %0, [%1], %2;"
                                             : "=r"(_old_mask)
                                             : "l"(mask_arr + pool_block_idx), "r"(bit)
                                             : "memory");
                            }
                        } else {
                            uint32_t* const ready_mb_write_ptr = workspace.get_shared_l1_task_count_ptr();
                            uint32_t* const ready_l2_avail_ptr = workspace.get_shared_l2_task_count_ptr();
                            if constexpr (kAgyInterleaveMode == 3) {
                                const uint32_t prev = ptx::atomic_add_rel(
                                    interleave_l1_done_base + pool_block_idx, 1u);
                                if (prev + 1u == kL1BlockNs) {
                                    const uint32_t q_slot = ptx::atomic_add(ready_mb_write_ptr, 1u);
                                    const uint32_t packed = ((local_expert_idx + 1u) << 20) | (m_block_idx & 0xfffffu);
                                    asm volatile("st.global.release.gpu.u32 [%0], %1;"
                                                 :: "l"(interleave_ready_mb_queue + q_slot), "r"(packed)
                                                 : "memory");
                                    ptx::atomic_add_rel(ready_l2_avail_ptr, kL2BlockNs);
                                }
                            } else if constexpr (kAgyInterleaveMode == 4 and kAgyHotColdMode == 0) {
                                const uint32_t mb = smem_expert_cache->m_blocks[local_expert_idx];
                                const uint32_t prev = ptx::atomic_add_rel(
                                    interleave_l1_done_base + local_expert_idx, 1u);
                                if (prev + 1u == mb * kL1BlockNs) {
                                    const uint32_t wave_idx = local_expert_idx / kNumExpertsPerWave;
                                    const uint32_t wave_e_lo = wave_idx * kNumExpertsPerWave;
                                    const uint32_t wave_mb_base = smem_expert_cache->pool_block_offset[wave_e_lo];
                                    const uint32_t q_off = ptx::atomic_add(interleave_wave_cursor_base + wave_idx, mb);
                                    for (uint32_t m_b = 0u; m_b < mb; ++ m_b) {
                                        const uint32_t packed = ((local_expert_idx + 1u) << 20) | (m_b & 0xfffffu);
                                        asm volatile("st.global.release.gpu.u32 [%0], %1;"
                                                     :: "l"(interleave_ready_mb_queue + wave_mb_base + q_off + m_b), "r"(packed)
                                                     : "memory");
                                    }
                                }
                            } else {
                                const uint32_t mb = smem_expert_cache->m_blocks[local_expert_idx];
                                const uint32_t prev = ptx::atomic_add_rel(
                                    interleave_l1_done_base + local_expert_idx, 1u);
                                if (prev + 1u == mb * kL1BlockNs) {
                                    const uint32_t q_base = ptx::atomic_add(ready_mb_write_ptr, mb);
                                    for (uint32_t m_b = 0u; m_b < mb; ++ m_b) {
                                        const uint32_t packed = ((local_expert_idx + 1u) << 20) | (m_b & 0xfffffu);
                                        asm volatile("st.global.release.gpu.u32 [%0], %1;"
                                                     :: "l"(interleave_ready_mb_queue + q_base + m_b), "r"(packed)
                                                     : "memory");
                                    }
                                    ptx::atomic_add_rel(ready_l2_avail_ptr, mb * kL2BlockNs);
                                }
                            }
                        }
                    }
                }
            } else {
                // L2 epilogue: BF16 cast and NVLink scatter
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kScatDispActive) {
                    if (valid_m > 0u) {
                        if (not scat_slot_free) {
                            sm90_fp8_mega_moe_scat_bar_wait(scat_free_bar, math_scat_phase);
                            scat_slot_free = true;
                        }
                        if (epilogue_warp_idx == 0 and lane_idx == 0) {
                            ptx::st_shared(scat_meta_ptr, m_idx, n_block_idx, valid_m, 0u);
                        }
                        if (row_block_offset < valid_m) {
                            if constexpr (kSwapABActive) {
                                auto store_scat_u16 = [&](const uint32_t& token,
                                                          const uint32_t& col,
                                                          const float& value) {
                                    const nv_bfloat16 bf = __float2bfloat16_rn(value);
                                    const uint16_t raw = *reinterpret_cast<const uint16_t*>(&bf);
                                    const uint32_t idx = get_scat_l2_elem_idx(epilogue_wg_idx, token, col);
                                    asm volatile("st.shared.u16 [%0], %1;"
                                                 :: "l"(__cvta_generic_to_shared(smem_scat_l2 + idx)),
                                                    "h"(raw));
                                };
                                auto store_l2_swap_chunk = [&](const uint32_t& i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        store_scat_u16(token_0, r_0, final_accum[i * 4 + 0]);
                                        store_scat_u16(token_0, r_1, final_accum[i * 4 + 2]);
                                    }
                                    if (token_1 < valid_m) {
                                        store_scat_u16(token_1, r_0, final_accum[i * 4 + 1]);
                                        store_scat_u16(token_1, r_1, final_accum[i * 4 + 3]);
                                    }
                                };
                                const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                                store_l2_swap_chunk(0);
                                if (valid_m > 8) {
                                    #pragma unroll
                                    for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                        if (i < num_swap_token_chunks)
                                            store_l2_swap_chunk(i);
                                    }
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                    const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                    auto write_scat_pair = [&](const uint32_t& row,
                                                               const uint32_t& col,
                                                               float value0,
                                                               float value1) {
                                        const uint32_t packed = math::cast_into_bf16_and_pack(value0, value1);
                                        const uint32_t idx = get_scat_l2_elem_idx(epilogue_wg_idx, row, col);
                                        ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_scat_l2 + idx), packed);
                                    };
                                    if (valid_r0) {
                                        write_scat_pair(r_0, chunk_lo * 8 + col_idx * 2,
                                            final_accum[chunk_lo * 4 + 0],
                                            final_accum[chunk_lo * 4 + 1]);
                                        write_scat_pair(r_0, chunk_hi * 8 + col_idx * 2,
                                            final_accum[chunk_hi * 4 + 0],
                                            final_accum[chunk_hi * 4 + 1]);
                                    }
                                    if (valid_r1) {
                                        write_scat_pair(r_1, chunk_lo * 8 + col_idx * 2,
                                            final_accum[chunk_lo * 4 + 2],
                                            final_accum[chunk_lo * 4 + 3]);
                                        write_scat_pair(r_1, chunk_hi * 8 + col_idx * 2,
                                            final_accum[chunk_hi * 4 + 2],
                                            final_accum[chunk_hi * 4 + 3]);
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        if (lane_idx == 0) {
                            sm90_fp8_mega_moe_scat_bar_arrive(scat_ready_bar);
                        }
                        scat_ever_used = true;
                        scat_slot_free = false;
                    }
                } else if (row_block_offset < valid_m) {
                if constexpr (kDirectL2Scatter) {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 128, "Direct L2 scatter only supports N128");

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row,
                                                  const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            uint32_t dst_rank_idx = 0, dst_token_idx = 0, dst_topk_idx = 0;
                            const uint32_t row_group_base = lane_idx - col_idx;
                            if (col_idx == 0) {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row_offset);
                                dst_rank_idx = src_metadata.rank_idx;
                                dst_token_idx = src_metadata.token_idx;
                                dst_topk_idx = kAgyDedupScatReq ? (src_metadata.topk_idx & 0xffu) : src_metadata.topk_idx;
                            }
                            const uint32_t row_group_mask = 0xfu << row_group_base;
                            const int src_lane = static_cast<int>(row_group_base);
                            dst_rank_idx = __shfl_sync(row_group_mask, dst_rank_idx, src_lane);
                            dst_token_idx = __shfl_sync(row_group_mask, dst_token_idx, src_lane);
                            dst_topk_idx = __shfl_sync(row_group_mask, dst_topk_idx, src_lane);
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * kCombineElementBytes);
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1]);
                                const uint32_t packed_hi = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1]);
                                *reinterpret_cast<uint32_t*>(
                                    mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(
                                    mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                } else {
                    auto store_l2_scalar = [&](const uint32_t& elem_idx, const float& value) {
                        smem_cd_l2[elem_idx] = __float2bfloat16_rn(value);
                    };
                    auto store_l2_pair = [&](const uint32_t& elem_idx,
                                             float value0, float value1) {
                        *reinterpret_cast<uint32_t*>(smem_cd_l2 + elem_idx) =
                            math::cast_into_bf16_and_pack(value0, value1);
                    };
                    if constexpr (kSwapABActive) {
                        constexpr bool kSwapL2XorSwz = kAgyFastSwapEpi and not kAgyFastSwapL2;
                        const uint32_t wg_smem_base =
                            epilogue_wg_idx * WG_BLOCK_M * kL2SmemStrideN;
                        const uint32_t r0_col = kSwapL2XorSwz ? (r_0 ^ (col_idx * 16u)) : r_0;
                        const uint32_t r1_col = kSwapL2XorSwz ? (r_1 ^ (col_idx * 16u)) : r_1;
                        auto store_value = [&](const uint32_t& token,
                                               const uint32_t& col,
                                               const float value) {
                            store_l2_scalar(
                                wg_smem_base + token * kL2SmemStrideN + col,
                                value);
                        };

                        auto store_l2_swap_chunk = [&](const uint32_t& i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                store_value(token_0, r0_col, final_accum[i * 4 + 0]);
                                store_value(token_0, r1_col, final_accum[i * 4 + 2]);
                            }
                            if (token_1 < valid_m) {
                                store_value(token_1, r0_col, final_accum[i * 4 + 1]);
                                store_value(token_1, r1_col, final_accum[i * 4 + 3]);
                            }
                        };

                        const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                        store_l2_swap_chunk(0);
                        if (valid_m > 8) {
                            #pragma unroll
                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                if (i < num_swap_token_chunks)
                                    store_l2_swap_chunk(i);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                            const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                            auto write_pair = [&](const uint32_t row,
                                                  const uint32_t col,
                                                  const float value0,
                                                  const float value1) {
                                const uint32_t elem_idx =
                                    epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N +
                                    row * WG_BLOCK_N + col;
                                store_l2_pair(elem_idx, value0, value1);
                            };
                            if (valid_r0) {
                                write_pair(r_0, chunk_lo * 8 + col_idx * 2,
                                    final_accum[chunk_lo * 4 + 0],
                                    final_accum[chunk_lo * 4 + 1]);
                                write_pair(r_0, chunk_hi * 8 + col_idx * 2,
                                    final_accum[chunk_hi * 4 + 0],
                                    final_accum[chunk_hi * 4 + 1]);
                            }
                            if (valid_r1) {
                                write_pair(r_1, chunk_lo * 8 + col_idx * 2,
                                    final_accum[chunk_lo * 4 + 2],
                                    final_accum[chunk_lo * 4 + 3]);
                                write_pair(r_1, chunk_hi * 8 + col_idx * 2,
                                    final_accum[chunk_hi * 4 + 2],
                                    final_accum[chunk_hi * 4 + 3]);
                            }
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows x 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp ->1 row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    const uint32_t cols_per_lane = WG_BLOCK_N / 16;
                    static_assert(WG_BLOCK_N == 64 or WG_BLOCK_N == 128 or WG_BLOCK_N == 256,
                                  "L2 scatter supports per-WG N64/N128/N256");

                    const uint32_t half_mask = 0xffffu << (row_in_warp_block * 16u);
                    const uint32_t half_base_lane = row_in_warp_block * 16u;
                    const uint32_t l2_wg_tile_idx = n_idx / WG_BLOCK_N;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = (kAgyFastSwapEpi or kAgyFastSwapL2)
                            ? (j * 8u + warp_idx_in_wg * 2u + row_in_warp_block)
                            : (warp_idx_in_wg * (kNumRowsPerWarp * 2) + j * 2 + row_in_warp_block);
                        const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t raw_topk_meta = src_metadata.topk_idx;
                        const uint32_t dst_topk_idx = kAgyDedupScatReq ? (raw_topk_meta & 0xffu) : raw_topk_meta;

                        constexpr bool kSwapL2XorSwz = kSwapABActive and kAgyFastSwapEpi and not kAgyFastSwapL2;
                        const uint32_t l2_smem_col = kSwapL2XorSwz
                            ? ((lane_in_row * cols_per_lane) ^ (((row_in_wg >> 1) & 3u) << 4))
                            : (lane_in_row * cols_per_lane);
                        const uint32_t smem_elem_idx =
                            epilogue_wg_idx * WG_BLOCK_M * kL2SmemStrideN +
                            row_in_wg * kL2SmemStrideN + l2_smem_col;
                        auto smem_ptr = smem_cd_base + smem_elem_idx * kCombineElementBytes;
                        constexpr uint32_t kScatterBytesPerLane =
                            (WG_BLOCK_N / 16) * kCombineElementBytes;

                        if constexpr (kDedupScatActive) {
                            const uint32_t dup_count = (raw_topk_meta >> 16) & 0xffu;
                            const uint32_t leader_topk_idx = (raw_topk_meta >> 8) & 0xffu;
                            if (dst_rank_idx != sym_buffer.rank_idx and dup_count >= 2u) {
                                uint32_t* cnt_ptr = workspace.get_src_token_topk_idx_ptr(
                                    l2_wg_tile_idx / 6u,
                                    dst_rank_idx,
                                    (2u + (l2_wg_tile_idx % 6u)) * kNumMaxTokensPerRank + dst_token_idx);
                                uint32_t ticket = 0u;
                                if (lane_in_row == 0u) {
                                    ticket = ptx::atomic_add(cnt_ptr, 1u) & 0xffu;
                                }
                                ticket = __shfl_sync(half_mask, ticket, half_base_lane);

                                auto* stage_base = reinterpret_cast<uint8_t*>(l1_token_buffer.get_base_ptr()) +
                                    (static_cast<uint64_t>(dst_rank_idx * kNumMaxTokensPerRank + dst_token_idx) *
                                     kHidden * kCombineElementBytes) +
                                    n_idx * kCombineElementBytes + lane_in_row * kScatterBytesPerLane;

                                if (ticket == 0u) {
                                    if constexpr (kScatterBytesPerLane == 32) {
                                        reinterpret_cast<uint4*>(stage_base)[0] = reinterpret_cast<const uint4*>(smem_ptr)[0];
                                        reinterpret_cast<uint4*>(stage_base)[1] = reinterpret_cast<const uint4*>(smem_ptr)[1];
                                    } else if constexpr (kScatterBytesPerLane == 16) {
                                        *reinterpret_cast<uint4*>(stage_base) = *reinterpret_cast<const uint4*>(smem_ptr);
                                    } else {
                                        *reinterpret_cast<uint2*>(stage_base) = *reinterpret_cast<const uint2*>(smem_ptr);
                                    }
                                    __syncwarp(half_mask);
                                    if (lane_in_row == 0u) {
                                        asm volatile("fence.acq_rel.gpu;" ::: "memory");
                                        ptx::atomic_add_rel(cnt_ptr, 0x40010000u);
                                    }
                                    __syncwarp(half_mask);
                                } else {
                                    if (lane_in_row == 0u) {
                                        while ((ptx::ld_acq(cnt_ptr) & 0x40000000u) == 0u) {}
                                    }
                                    __syncwarp(half_mask);
                                    auto* stage_bf162 = reinterpret_cast<__nv_bfloat162*>(stage_base);
                                    const auto* smem_bf162 = reinterpret_cast<const __nv_bfloat162*>(smem_ptr);
                                    #pragma unroll
                                    for (uint32_t p = 0; p < kScatterBytesPerLane / 4u; ++ p) {
                                        atomicAdd(stage_bf162 + p, smem_bf162[p]);
                                    }
                                    __syncwarp(half_mask);
                                    uint32_t is_last = 0u;
                                    if (lane_in_row == 0u) {
                                        asm volatile("fence.acq_rel.gpu;" ::: "memory");
                                        const uint32_t prev_fin = ptx::atomic_add_rel(cnt_ptr, 0x00010000u);
                                        is_last = ((((prev_fin >> 16) & 0xffu) + 1u) == dup_count) ? 1u : 0u;
                                        if (is_last) {
                                            (void)ptx::ld_acq(cnt_ptr);
                                            asm volatile("fence.acq_rel.gpu;" ::: "memory");
                                        }
                                    }
                                    is_last = __shfl_sync(half_mask, is_last, half_base_lane);
                                    if (is_last) {
                                        const auto dst_token = combine_token_buffer.get_rank_buffer(leader_topk_idx)
                                                               .get_data_buffer(dst_token_idx);
                                        auto dst_ptr = math::advance_ptr<uint8_t>(
                                            dst_token.get_base_ptr(),
                                            n_idx * kCombineElementBytes + lane_in_row * kScatterBytesPerLane);
                                        auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);
                                        if constexpr (kScatterBytesPerLane == 32) {
                                            const auto p0 = reinterpret_cast<const uint4*>(stage_base)[0];
                                            const auto p1 = reinterpret_cast<const uint4*>(stage_base)[1];
                                            __syncwarp(half_mask);
                                            if (lane_in_row == 0u) *cnt_ptr = 0u;
                                            reinterpret_cast<uint4*>(mapped_dst_ptr)[0] = p0;
                                            reinterpret_cast<uint4*>(mapped_dst_ptr)[1] = p1;
                                        } else if constexpr (kScatterBytesPerLane == 16) {
                                            const auto p0 = *reinterpret_cast<const uint4*>(stage_base);
                                            __syncwarp(half_mask);
                                            if (lane_in_row == 0u) *cnt_ptr = 0u;
                                            *reinterpret_cast<uint4*>(mapped_dst_ptr) = p0;
                                        } else {
                                            const auto p0 = *reinterpret_cast<const uint2*>(stage_base);
                                            __syncwarp(half_mask);
                                            if (lane_in_row == 0u) *cnt_ptr = 0u;
                                            *reinterpret_cast<uint2*>(mapped_dst_ptr) = p0;
                                        }
                                    }
                                }
                                continue;
                            }
                        }

                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        auto dst_ptr = math::advance_ptr<uint8_t>(
                            dst_token.get_base_ptr(),
                            n_idx * kCombineElementBytes + lane_in_row * kScatterBytesPerLane);
                        auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);

                        if constexpr (kScatterBytesPerLane == 32) {
                            const auto packed0 = *reinterpret_cast<uint4*>(smem_ptr);
                            const auto packed1 = *(reinterpret_cast<uint4*>(smem_ptr) + 1);
                            reinterpret_cast<uint4*>(mapped_dst_ptr)[0] = packed0;
                            reinterpret_cast<uint4*>(mapped_dst_ptr)[1] = packed1;
                        } else if constexpr (kScatterBytesPerLane == 16) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            *reinterpret_cast<uint4*>(mapped_dst_ptr) = packed;
                        } else if constexpr (kScatterBytesPerLane == 8) {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            *reinterpret_cast<uint2*>(mapped_dst_ptr) = packed;
                        } else {
                            DG_STATIC_ASSERT(kScatterBytesPerLane == 4,
                                             "Unexpected L2 scatter width");
                            *reinterpret_cast<uint32_t*>(mapped_dst_ptr) =
                                *reinterpret_cast<uint32_t*>(smem_ptr);
                        }
                    }

                    if constexpr (kAgyFastSwapL2 and not kAgyInterleaveActive) {
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    } else if constexpr (not kAgyInterleaveActive) {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                }
                } // end if (row_block_offset < valid_m) on L2
                if constexpr (kAgyInterleaveActive and not kScatDispActive) {
                    // CTA-wide barrier protects smem_cd_base across interleaved L2 -> L1 transitions!
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            }
            AGY_PROF_TILE_STORE_END(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx,
                                    is_linear1_phase, local_expert_idx, m_block_idx, n_block_idx);
        });

        if constexpr (!MegaMoEPhase::runs_linear2) {
            AGY_PROF_MATH_CLOSE(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx);
            return;
        }

        if constexpr (kScatDispActive) {
            if (not scat_slot_free) {
                sm90_fp8_mega_moe_scat_bar_wait(scat_free_bar, math_scat_phase);
                scat_slot_free = true;
            }
            if (epilogue_warp_idx == 0 and lane_idx == 0) {
                ptx::st_shared(scat_meta_ptr, 0u, 0u, 0u, 1u);
            }
            __syncwarp();
            if (lane_idx == 0) {
                sm90_fp8_mega_moe_scat_bar_arrive(scat_ready_bar);
            }
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }
        } // end kRoleMode != 3 (GEMM + Epilogue/Scatter)
        if constexpr (!MegaMoEPhase::runs_linear2 or kRoleMode == 2) {
            return;
        }

        // Combine
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        AGY_PROF_BEFORE_BAR2(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx);
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumEpilogueThreads,
            kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag,
            comm::BarrierTimeoutPolicy::TrapOnly>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );
        AGY_PROF_AFTER_BAR2(MegaMoEPhase, sm_idx, epilogue_thread_idx);
        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        AGY_PROF_COMBINE_BEGIN(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx);

        constexpr uint32_t kNumChunks = kCombineNumChunks;
        constexpr uint32_t kInputChunkBytes = kCombineInputChunkBytes;
        constexpr uint32_t kOutputChunkBytes = kCombineOutputChunkBytes;
        constexpr uint32_t kElemsPerVector = 8;
        constexpr uint32_t kInputVectorBytes = kElemsPerVector * kCombineElementBytes;
        constexpr uint32_t kNumVectorsPerLane =
            kCombineChunkElems / (32 * kElemsPerVector);
        constexpr uint32_t kNumBF16PairsPerVector = kElemsPerVector / 2;
        DG_STATIC_ASSERT(kInputChunkBytes % 16 == 0,
                         "Combine input chunk must be TMA-aligned");
        DG_STATIC_ASSERT(kOutputChunkBytes % 16 == 0,
                         "Combine output chunk must be TMA-aligned");
        DG_STATIC_ASSERT(kCombineChunkElems % (32 * kElemsPerVector) == 0,
                         "Combine chunk must distribute evenly across a warp");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint8_t>(
                smem_buffer,
                (epilogue_warp_idx + i * kNumEpilogueWarps) * kInputChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(smem_buffer,
            2 * kNumEpilogueWarps * kInputChunkBytes +
                epilogue_warp_idx * kOutputChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        const auto compute_token_combine_mask = [&](const uint32_t& token_idx) -> uint32_t {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
            if constexpr (kDedupScatActive) {
                bool keep_slot = false;
                if (stored_topk_slot_idx >= 0) {
                    const uint32_t slot_dst_rank = static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank;
                    const uint32_t peer_mask = __match_any_sync(total_mask, slot_dst_rank);
                    const uint32_t leader_lane = static_cast<uint32_t>(__ffs(peer_mask) - 1);
                    keep_slot = (slot_dst_rank == sym_buffer.rank_idx) or (lane_idx == leader_lane);
                }
                total_mask = __ballot_sync(0xffffffff, keep_slot);
            }
            return total_mask;
        };

        const auto run_flattened_combine = [&]<uint32_t kRunChunks>() {
            constexpr uint32_t kRunChunkElems = kHidden / kRunChunks;
            constexpr uint32_t kRunInputBytes = kCombineInputHiddenBytes / kRunChunks;
            constexpr uint32_t kRunOutputBytes = kCombineOutputHiddenBytes / kRunChunks;
            constexpr uint32_t kRunVecsPerLane = kRunChunkElems / (32u * kElemsPerVector);
            DG_STATIC_ASSERT(kRunInputBytes <= kInputChunkBytes and kRunOutputBytes <= kOutputChunkBytes,
                             "Flattened combine chunk exceeds warp SMEM staging buffer");

            const uint32_t total_items = num_tokens * kRunChunks;
            const bool y_aligned_16 = (reinterpret_cast<uintptr_t>(y) & 15u) == 0u;
            uint32_t cached_token = 0xffffffffu;
            uint32_t cached_mask = 0u;
            for (uint32_t work_id = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
                 work_id < total_items;
                 work_id += kNumSMs * kNumEpilogueWarps) {
                const uint32_t token_idx = work_id / kRunChunks;
                const uint32_t chunk = work_id - token_idx * kRunChunks;
                if (token_idx != cached_token) {
                    cached_token = token_idx;
                    cached_mask = compute_token_combine_mask(token_idx);
                }
                const uint32_t input_chunk_byte_offset = chunk * kRunInputBytes;
                const uint32_t output_chunk_byte_offset = chunk * kRunOutputBytes;

                if constexpr (kAgyRegComb and kRunVecsPerLane <= 2u) {
                    uint32_t mask = cached_mask;
                    float2 reduced[kRunVecsPerLane * kNumBF16PairsPerVector] = {};
                    while (mask & (mask - 1u)) {
                        const uint32_t slot0 = static_cast<uint32_t>(__ffs(mask) - 1);
                        mask &= (mask - 1u);
                        const uint32_t slot1 = static_cast<uint32_t>(__ffs(mask) - 1);
                        mask &= (mask - 1u);

                        const auto* src0 = reinterpret_cast<const uint4*>(
                            math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot0)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                input_chunk_byte_offset));
                        const auto* src1 = reinterpret_cast<const uint4*>(
                            math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot1)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                input_chunk_byte_offset));

                        uint4 v0[kRunVecsPerLane];
                        uint4 v1[kRunVecsPerLane];
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j)
                            v0[j] = ptx::ld_evict_first(src0 + j * 32u + lane_idx);
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j)
                            v1[j] = ptx::ld_evict_first(src1 + j * 32u + lane_idx);

                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            const auto* bf0 = reinterpret_cast<const nv_bfloat162*>(&v0[j]);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                ptx::accumulate(
                                    reduced[j * kNumBF16PairsPerVector + l],
                                    bf0[l]);
                        }
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            const auto* bf1 = reinterpret_cast<const nv_bfloat162*>(&v1[j]);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                ptx::accumulate(
                                    reduced[j * kNumBF16PairsPerVector + l],
                                    bf1[l]);
                        }
                    }
                    if (mask) {
                        const uint32_t slot0 = static_cast<uint32_t>(__ffs(mask) - 1);
                        const auto* src0 = reinterpret_cast<const uint4*>(
                            math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot0)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                input_chunk_byte_offset));
                        uint4 v0[kRunVecsPerLane];
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j)
                            v0[j] = ptx::ld_evict_first(src0 + j * 32u + lane_idx);
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            const auto* bf0 = reinterpret_cast<const nv_bfloat162*>(&v0[j]);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                ptx::accumulate(
                                    reduced[j * kNumBF16PairsPerVector + l],
                                    bf0[l]);
                        }
                    }

                    auto* dst_y_ptr = math::advance_ptr<uint8_t>(
                        y,
                        static_cast<uint64_t>(token_idx) * kCombineOutputHiddenBytes +
                            output_chunk_byte_offset);
                    if (y_aligned_16) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            uint4 casted;
                            auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                casted_bf16[l] = __float22bfloat162_rn(
                                    reduced[j * kNumBF16PairsPerVector + l]);
                            *reinterpret_cast<uint4*>(dst_y_ptr + (j * 32u + lane_idx) * 16u) = casted;
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            auto* dst_bf16 = reinterpret_cast<nv_bfloat16*>(
                                dst_y_ptr + (j * 32u + lane_idx) * 16u);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l) {
                                const float2 r_pair = reduced[j * kNumBF16PairsPerVector + l];
                                dst_bf16[2u * l + 0u] = __float2bfloat16_rn(r_pair.x);
                                dst_bf16[2u * l + 1u] = __float2bfloat16_rn(r_pair.y);
                            }
                        }
                    }
                } else {
                    uint32_t mask = cached_mask;
                    const auto move_mask_and_load = [&](const uint32_t& i) {
                        if (mask) {
                            const uint32_t slot_idx = __ffs(mask) - 1;
                            mask ^= 1u << slot_idx;
                            __syncwarp();
                            if (cute::elect_one_sync()) {
                                const auto src_ptr = math::advance_ptr<uint8_t>(
                                    combine_token_buffer.get_rank_buffer(slot_idx)
                                                        .get_data_buffer(token_idx).get_base_ptr(),
                                    input_chunk_byte_offset);
                                ptx::tma_load_1d(
                                    combine_load_buffer[i], src_ptr,
                                    combine_load_barriers[i], kRunInputBytes);
                                ptx::mbarrier_arrive_and_set_tx(
                                    combine_load_barriers[i], kRunInputBytes);
                            }
                            __syncwarp();
                            return true;
                        }
                        return false;
                    };

                    bool do_reduce = move_mask_and_load(load_stage_idx);
                    float2 reduced[kRunVecsPerLane * kNumBF16PairsPerVector] = {};
                    while (do_reduce) {
                        do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                        combine_load_barriers[load_stage_idx]->wait(combine_phase);
                        #pragma unroll
                        for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                            const uint32_t vector_idx = j * 32 + lane_idx;
                            const auto packed = *reinterpret_cast<const uint4*>(
                                combine_load_buffer[load_stage_idx] +
                                vector_idx * kInputVectorBytes);
                            const auto bf16_values =
                                reinterpret_cast<const nv_bfloat162*>(&packed);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                ptx::accumulate(
                                    reduced[j * kNumBF16PairsPerVector + l],
                                    bf16_values[l]);
                        }
                        combine_phase ^= load_stage_idx;
                        load_stage_idx ^= 1;
                    }

                    #pragma unroll
                    for (uint32_t j = 0; j < kRunVecsPerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(
                                reduced[j * kNumBF16PairsPerVector + l]);

                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                       casted.x, casted.y, casted.z, casted.w);
                    }
                    __syncwarp();

                    if (cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        ptx::tma_store_1d(
                            math::advance_ptr(
                                y,
                                static_cast<uint64_t>(token_idx) * kCombineOutputHiddenBytes +
                                    output_chunk_byte_offset),
                            combine_store_buffer, kRunOutputBytes);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
            }
        };

        if constexpr (kAgyAllIdleCombActive) {
            constexpr uint32_t kTotalCombWarpsPerSM = kNumDispatchWarps + kNumEpilogueWarps;
            run_all_idle_reg_combine(
                sm_idx * kTotalCombWarpsPerSM + kNumDispatchWarps + epilogue_warp_idx,
                kNumSMs * kTotalCombWarpsPerSM);
        } else if constexpr (kAgyFastComb) {
            constexpr uint32_t kTotalEpiWarps = kNumSMs * kNumEpilogueWarps;
            constexpr uint32_t kFineChunks = kHidden / 256u;
            if constexpr (kAgyRegComb) {
                constexpr uint32_t kHalfFineCand =
                    ((kHidden / 256u) % 2u == 0u) ? (kHidden / 512u) : kFineChunks;
                constexpr uint32_t kDualVecChunks =
                    (kHalfFineCand >= kNumChunks) ? kHalfFineCand : kFineChunks;
                if (num_tokens * kFineChunks <= kTotalEpiWarps) {
                    run_flattened_combine.template operator()<kFineChunks>();
                } else if (kAgyRegCombMode >= 2 or num_tokens * 2u <= kTotalEpiWarps) {
                    run_flattened_combine.template operator()<kDualVecChunks>();
                } else {
                    run_flattened_combine.template operator()<kNumChunks>();
                }
            } else {
                constexpr uint32_t kMedCandidate =
                    ((kHidden / 256u) % 4u == 0u) ? 4u : (((kHidden / 256u) % 2u == 0u) ? 2u : 1u);
                constexpr uint32_t kMedChunks = kMedCandidate > kNumChunks ? kMedCandidate : kNumChunks;

                if (num_tokens * kMedChunks <= kTotalEpiWarps / 2u) {
                    run_flattened_combine.template operator()<kFineChunks>();
                } else if (num_tokens * kNumChunks < kTotalEpiWarps) {
                    run_flattened_combine.template operator()<kMedChunks>();
                } else {
                    run_flattened_combine.template operator()<kNumChunks>();
                }
            }
        } else {
            for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
                 token_idx < num_tokens;
                 token_idx += kNumSMs * kNumEpilogueWarps) {
                const uint32_t total_mask = compute_token_combine_mask(token_idx);

                for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                    const uint32_t input_chunk_byte_offset = chunk * kInputChunkBytes;
                    const uint32_t output_chunk_byte_offset = chunk * kOutputChunkBytes;

                    uint32_t mask = total_mask;
                    const auto move_mask_and_load = [&](const uint32_t& i) {
                        if (mask) {
                            const uint32_t slot_idx = __ffs(mask) - 1;
                            mask ^= 1 << slot_idx;
                            __syncwarp();
                            if (cute::elect_one_sync()) {
                                const auto src_ptr = math::advance_ptr<uint8_t>(
                                    combine_token_buffer.get_rank_buffer(slot_idx)
                                                        .get_data_buffer(token_idx).get_base_ptr(),
                                    input_chunk_byte_offset);
                                ptx::tma_load_1d(
                                    combine_load_buffer[i], src_ptr,
                                    combine_load_barriers[i], kInputChunkBytes);
                                ptx::mbarrier_arrive_and_set_tx(
                                    combine_load_barriers[i], kInputChunkBytes);
                            }
                            __syncwarp();
                            return true;
                        }
                        return false;
                    };

                    bool do_reduce = move_mask_and_load(load_stage_idx);

                    float2 reduced[kNumVectorsPerLane * kNumBF16PairsPerVector] = {};
                    while (do_reduce) {
                        do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                        combine_load_barriers[load_stage_idx]->wait(combine_phase);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumVectorsPerLane; ++ j) {
                            const uint32_t vector_idx = j * 32 + lane_idx;
                            const auto packed = *reinterpret_cast<const uint4*>(
                                combine_load_buffer[load_stage_idx] +
                                vector_idx * kInputVectorBytes);
                            const auto bf16_values =
                                reinterpret_cast<const nv_bfloat162*>(&packed);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                                ptx::accumulate(
                                    reduced[j * kNumBF16PairsPerVector + l],
                                    bf16_values[l]);
                        }
                        combine_phase ^= load_stage_idx;
                        load_stage_idx ^= 1;
                    }

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumVectorsPerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(
                                reduced[j * kNumBF16PairsPerVector + l]);

                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                       casted.x, casted.y, casted.z, casted.w);
                    }
                    __syncwarp();

                    if (cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        ptx::tma_store_1d(
                            math::advance_ptr(
                                y,
                                static_cast<uint64_t>(token_idx) * kCombineOutputHiddenBytes +
                                    output_chunk_byte_offset),
                            combine_store_buffer, kOutputChunkBytes);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
            }
        }
        AGY_PROF_COMBINE_END_AND_CLOSE(MegaMoEPhase, sm_idx, epilogue_wg_idx, warp_idx_in_wg, lane_idx);
    }
    } // end kRoleMode == 0/2/3
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
    }
};

template <typename MegaMoEPhase, DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_DEVICE void
sm90_fp8_mega_moe_core(DG_SM90_FP8_MOE_CORE_ARGS_DECL) {
    constexpr uint32_t kTotalThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr int kDispatchMode = (kTotalThreads >= 512u) ? 1 : 0;
    SM90MegaMoECoreRunner<MegaMoEPhase, DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>::template run<kDispatchMode>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS,
          bool kNMajorScheduleRequested = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1) void
sm90_fp8_mega_moe_l1_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    using Phase = MegaMoEPhasePolicy<
        MegaMoEPhaseKind::Linear1, kNMajorScheduleRequested>;
    sm90_fp8_mega_moe_core<Phase, DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) and defined(AGY_MCAST) and (AGY_MCAST == 2)
    if constexpr (Phase::is_fused and (kNumSMs % 2u == 0u)) {
        comm::cluster_sync_with_relaxed_arrive();
    }
#endif
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS,
          bool kDirectL2ScatterRequested = false,
          bool kNMajorScheduleRequested = false,
          bool kOneWarpCleanupRequested = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1) void
sm90_fp8_mega_moe_l2_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    using Phase = MegaMoEPhasePolicy<
        MegaMoEPhaseKind::Linear2, kNMajorScheduleRequested,
        kDirectL2ScatterRequested, kOneWarpCleanupRequested>;
    sm90_fp8_mega_moe_core<Phase, DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) and defined(AGY_MCAST) and (AGY_MCAST == 2)
    if constexpr (Phase::is_fused and (kNumSMs % 2u == 0u)) {
        comm::cluster_sync_with_relaxed_arrive();
    }
#endif
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS,
          bool kDirectL2ScatterRequested = false,
          bool kL2NMajorScheduleRequested = false,
          bool kOneWarpCleanupRequested = false,
          bool kL1NMajorScheduleRequested = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1) void
sm90_fp8_mega_moe_fused_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    using Phase = MegaMoEPhasePolicy<
        MegaMoEPhaseKind::Fused, kL2NMajorScheduleRequested,
        kDirectL2ScatterRequested, kOneWarpCleanupRequested,
        kL1NMajorScheduleRequested>;
    sm90_fp8_mega_moe_core<Phase, DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) and defined(AGY_MCAST) and (AGY_MCAST == 2)
    if constexpr (Phase::is_fused and (kNumSMs % 2u == 0u)) {
        comm::cluster_sync_with_relaxed_arrive();
    }
#endif
}

#undef DG_SM90_FP8_MOE_TEMPLATE_PARAMS
#undef DG_SM90_FP8_MOE_KERNEL_ARGS_DECL
#undef DG_SM90_FP8_MOE_CORE_ARGS_DECL
#undef DG_SM90_FP8_MOE_KERNEL_ARGS
#undef DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS

} // namespace deep_gemm

#pragma clang diagnostic pop



