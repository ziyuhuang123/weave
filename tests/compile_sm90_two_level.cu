// Two-Level compile gate for the H20 SM90 path (project README PTX convention).
// Enables the dedicated streaming Reduce warps (WEAVE_REDUCE_WARPS=4) on top of
// the chunked-combine + HBM-dedup scatter structure that implements them, plus
// the online adaptive tier (WEAVE_ADAPT_BM=1).  Every modified gate site --
// the reduce-warp dispatch gate, the runtime stream stride and the online
// v_head load -- is type-checked and lowered to PTX.
//
//   nvcc -std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda \
//        -ptx -arch=sm_90a -I deep_gemm/include \
//        -I third-party/cutlass/include -I third-party/deep_jit/include \
//        tests/compile_sm90_two_level.cu -o sm90_two_level.ptx
#define WEAVE_PRECISION_MODE 2
#define WEAVE_ABLATION_PRESET 0
#define WEAVE_DISPATCH_WARPS 4
#define WEAVE_SCATTER_ACTIVE_SMS 16
#define WEAVE_SM0_STREAM_SCHED 1
#define WEAVE_L2_KBLOCK_DEP 1
#define WEAVE_RING_BLOCKS 0
#define WEAVE_EARLY_STREAM_REDUCE 1
#define WEAVE_HEAD_PULL_CHUNK 64
#define WEAVE_LOCAL_SCATTER_DEDUP 1
#define WEAVE_HBM_SCATTER_QUEUE 1
#define WEAVE_ADAPT_BM 1
#define WEAVE_BLOCK_M_L1 64
#define WEAVE_BLOCK_M_L2 64
#define WEAVE_STORE_BLOCK_M_L1 64
#define WEAVE_SCATTER_OWNER 1
#define WEAVE_SCATTER_STAGING 2
#define WEAVE_DISPATCH_ACTIVE_SMS 0
#define WEAVE_COMBINE_SYNC_CHUNK 32
#define WEAVE_SCATTER_MODE 2
#define WEAVE_INTERLEAVE_L1_L2 5
// Two-Level Tier-1 regime: R_max=4 dedicated Reduce warps; the CTA-0 online
// tier (WEAVE_ADAPT_BM=1) gates them per step on the live routing histogram.
#define WEAVE_REDUCE_WARPS 4
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>
