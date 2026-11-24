// Two-Level compile gate: instantiates the SM103 fp8xfp4 MegaMoE kernel with
// the Tier-1 structural regime that enables the KEY runtime gate -- dedicated
// Middle Reduce warps (WEAVE_REDUCE_WARPS=4 + fine-grained D6 combine) and the
// online adaptive tier (WEAVE_ADAPT_BM=1).  Every modified gate site is lowered
// to PTX; run ptxas -v on the result to verify zero spills.
//
//   nvcc -std=c++20 -ptx -arch=sm_100a -I deep_gemm/include \
//        -I third-party/cutlass/include -I third-party/deep_jit/include \
//        tests/compile_sm100_two_level.cu -o sm100_two_level.ptx
//   ptxas -arch=sm_100a -v sm100_two_level.ptx -o sm100_two_level.cubin
//
// Scatter executor variants (both gate sites must compile):
//   -DWEAVE_SCATTER_MODE=0  epilogue_only, default D7 path (gate at the
//                           dedicated-reduce-warp branch)
//   -DWEAVE_SCATTER_MODE=1  comm_only SMEM handoff (gate at the post-drain
//                           streaming-reduce branches)
#ifndef T_BM
#define T_BM 128
#endif
#ifndef T_STORE_L1
#define T_STORE_L1 32
#endif
#ifndef T_STORE_L2
#define T_STORE_L2 T_STORE_L1
#endif
#ifndef T_STAGES
#define T_STAGES 4
#endif
// Routed-expert shape.  The defaults are the two-level regime; the fused
// SharedLinear2 + INT7 arm overrides them (one shared expert, top-k <= 8,
// 8-row GEMM2 stores) so its kernel path is actually instantiated.
#ifndef T_NUM_SHARED
#define T_NUM_SHARED 0
#endif
#ifndef T_TOPK
#define T_TOPK 10
#endif
#ifndef WEAVE_EPI_REG_STAGES
#define WEAVE_EPI_REG_STAGES 1
#endif
#ifndef WEAVE_TMEM_PIPELINE_MODE
#define WEAVE_TMEM_PIPELINE_MODE 200
#endif
#ifndef WEAVE_SCATTER_MODE
#define WEAVE_SCATTER_MODE 0
#endif
#if WEAVE_SCATTER_MODE == 1 || WEAVE_SCATTER_MODE == 3
#define WEAVE_SCATTER_OWNER 1
#else
#define WEAVE_SCATTER_OWNER 0
#endif
#if WEAVE_SCATTER_MODE == 3
#define WEAVE_SCATTER_STAGING 2
#define WEAVE_LOCAL_SCATTER_DEDUP 1
#elif WEAVE_SCATTER_MODE == 1
#define WEAVE_SCATTER_STAGING 1
#else
#define WEAVE_SCATTER_STAGING 0
#endif

// Two-Level Tier-1 regime knobs (the online CTA-0 tier gates the lanes).
// WEAVE_ADAPT_BM=1 requires the per-expert BM ladder (EXPERT_STEP_MODE=1);
// sub-tiled GEMM2 tasks compose only with D6 group <= 1, so the fine-grained
// combine uses one 128-column N-tile per ready group here.
#ifndef WEAVE_REDUCE_WARPS
#define WEAVE_REDUCE_WARPS 4
#endif
#ifndef WEAVE_COMBINE_GROUP_TILES
#define WEAVE_COMBINE_GROUP_TILES 1u
#endif
#ifndef WEAVE_ADAPT_BM
#define WEAVE_ADAPT_BM 1
#endif
#ifndef WEAVE_EXPERT_STEP_MODE
#define WEAVE_EXPERT_STEP_MODE 1
#endif

#define WEAVE_PRECISION_MODE 0
#define WEAVE_ABLATION_PRESET 0
#define WEAVE_LOCALITY_DOMAINS 1
#define WEAVE_DISPATCH_WARPS 4
#define WEAVE_SM0_STREAM_SCHED 0
#define WEAVE_L2_KBLOCK_DEP 1
#define WEAVE_RING_BLOCKS 0
#define WEAVE_EARLY_STREAM_REDUCE 0
#define WEAVE_HEAD_PULL_CHUNK 0
#define WEAVE_BLOCK_M_L1 0
#define WEAVE_BLOCK_M_L2 0
#define WEAVE_STORE_BLOCK_M_L1 T_STORE_L1
#define WEAVE_DISPATCH_ACTIVE_SMS 0
#define WEAVE_COMBINE_SYNC_CHUNK 0
#define WEAVE_SCHED_L2_RATIO 1
#define WEAVE_REG_BALANCE 0
#define WEAVE_READY_GROUP_SIZE 0
#define WEAVE_WARP_SPLIT_MODE 0
#define WEAVE_REG_PULL_BYTES 0u
#define WEAVE_HETERO_SM_LIMIT 74u
#define WEAVE_DISP_REGS 0u
#define WEAVE_EPI_REGS 0u
#define WEAVE_BLOCK_M 0
#define WEAVE_BLOCK_N 0
#define WEAVE_BLOCK_K 0
#define WEAVE_STORE_BLOCK_M 0
#define WEAVE_STORE_BLOCK_M_L2 T_STORE_L2
#define WEAVE_DYN_EPILOGUE 0
#define WEAVE_DYN_TMA_A 0
#define WEAVE_OVERLAP_EXPERT_SMEM 0
#define WEAVE_ACCUM_STAGES 2u
#define WEAVE_TMA_STORE_STAGES 2u
#define WEAVE_SCHED_STAGES 2u
#define DG_D3_HAS_EXTRA_TMAS 0
#define WEAVE_EXPERTS_PER_WAVE 0
#define WEAVE_STATIC_EXPERT_ORDER 0
#define WEAVE_READY_QUEUE 0
#define WEAVE_SCHED_MODE 0
#define WEAVE_DEDUP_MODE 0
#define WEAVE_DEDUP_SCHED 0
#define WEAVE_COMBINE_REDUCE_WARPS 0u
#define WEAVE_SCATTER_SMEM_ALIAS 1
#define WEAVE_SCATTER_WARPS 0
#define WEAVE_SCATTER_HANDOFF_WAIT 1
#define WEAVE_SCATTER_ACTIVE_SMS 0
#define WEAVE_SHARED_L2_EARLY_BLOCKS 10u
#define WEAVE_ENABLE_MULTICAST 1
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

constexpr uint32_t kSfBlockM = ((T_BM + 127) / 128) * 128;
constexpr uint32_t kRingTokens = T_BM * 64;
constexpr uint32_t kSfRingTokens = (kRingTokens / T_BM) * kSfBlockM;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        1024,
        8192, 2048,
        512, T_NUM_SHARED,
        T_TOPK,
        T_BM, 128, 128,
        T_STORE_L1,
        kSfBlockM, 128,
        kRingTokens,
        kSfRingTokens,
        T_STAGES,
        4096,
        128, 128, 256,
        148, 4,
        10.0f,
        true>);
    (void)ptr;
}
