// Compile-time instantiation check of the SM103 fp8xfp4 MegaMoE kernel with the
// CTA-wide (all 16 warps) tail Reduce: WEAVE_REDUCE_WARPS>0 (spatial streaming
// Reduce) plus a two-warpgroup epilogue => 4 dispatch + 4 MMA non-epilogue +
// 8 epilogue warps = 512 threads, W00..W15.
//
// The macro block and template arguments below are exactly what the host JIT
// emits for the w6 EP4 / S=1024 / num_shared_experts=0 arm; only the epilogue
// thread count is raised from 128 to 256 (WEAVE_EPILOGUE_WARPGROUPS=2), which is
// the geometry that makes W04..W07 and W08..W15 join the same tail Reduce.
//
//   nvcc -cubin -arch=sm_103a -std=c++20 --expt-relaxed-constexpr \
//        -Xptxas -v -I deep_gemm/include -I <cutlass>/include \
//        tests/compile_sm103_all16warp_reduce.cu
//
// The ptxas spill line must read "0 bytes spill stores, 0 bytes spill loads".
//
// Four -D arms cover the reduce/scatter space that ships (all must stay at
// <=128 registers / 0 spill / 0 LDL / 18 UTMACCTL.PF / 16 UTMALDG):
//   * rw4 (spatial streaming Middle + 16-warp TMA tail, the production arm):
//       -DWEAVE_REDUCE_WARPS=4
//   * rw0 (legacy all-warp SMEM tail, the reference all-16-warp arm):
//       -DWEAVE_REDUCE_WARPS=0 -DWEAVE_COMBINE_REDUCE_WARPS=16u
//   * d7q (WEAVE_LOCAL_SCATTER_DEDUP pre-Reduce over the HBM scatter queue):
//       -DWEAVE_REDUCE_WARPS=4 -DWEAVE_LOCAL_SCATTER_DEDUP=1
//       -DDG_D7_QUEUE_SCATTER=2 -DWEAVE_SCATTER_MODE=3 -DWEAVE_SCATTER_STAGING=2
//       -DWEAVE_SCATTER_OWNER=1 -DWEAVE_DEDUP_MODE=0 -DWEAVE_DEDUP_SCHED=0
//   * bm_dec (decoupled GEMM1/GEMM2 tile heights: 128-row envelope, GEMM2 at 64)
//       -DWEAVE_GATE_BM_DECOUPLED=1 -DWEAVE_GATE_T_BLOCK_M=128
//       -DWEAVE_GATE_T_STORE_BLOCK_M=16 -DWEAVE_GATE_T_STAGES=8
//       -DWEAVE_GATE_BLOCK_M_L1=128 -DWEAVE_GATE_BLOCK_M_L2=64
//       -DWEAVE_BLOCK_M=128 -DWEAVE_STORE_BLOCK_M=16
//       -DWEAVE_DEDUP_MODE=0 -DWEAVE_DEDUP_SCHED=0
// The bm_dec arm is a 128-row envelope, so its UTMACCTL.PF/UTMALDG counts
// follow the 128-row MMA shape (23/28) instead of the 64-row 18/16.
//
// A mode-2 dedup arm also compiles: -DWEAVE_LOCAL_SCATTER_DEDUP=1
// -DWEAVE_SCATTER_OWNER=0 -DWEAVE_SCATTER_STAGING=0 -DWEAVE_COMBINE_SYNC_CHUNK=0
// -DWEAVE_COMBINE_GROUP_TILES=0 -DWEAVE_EARLY_STREAM_REDUCE=0.
#define WEAVE_LOCALITY_DOMAINS 1
#define WEAVE_PRECISION_MODE 0u
#define WEAVE_DISPATCH_WARPS 4
#ifndef WEAVE_REDUCE_WARPS
#define WEAVE_REDUCE_WARPS 4
#endif
#define WEAVE_SM0_STREAM_SCHED 0
#define WEAVE_L2_KBLOCK_DEP 0
#define WEAVE_RING_BLOCKS 0
#ifndef WEAVE_EARLY_STREAM_REDUCE
#define WEAVE_EARLY_STREAM_REDUCE 1
#endif
#define WEAVE_TMEM_PIPELINE_MODE 200
#define WEAVE_HEAD_PULL_CHUNK 0
#ifndef WEAVE_LOCAL_SCATTER_DEDUP
#define WEAVE_LOCAL_SCATTER_DEDUP 0
#endif
#ifndef DG_D7_QUEUE_SCATTER
#define DG_D7_QUEUE_SCATTER 0
#endif
#define WEAVE_ADAPT_BM 0
// GEMM envelope and per-phase tile heights.  The production rw0/rw4/d7q arms
// are 64/64/64 (one pool block per phase); -DWEAVE_GATE_T_BLOCK_M=128 with
// -DWEAVE_GATE_BLOCK_M_L1=128 -DWEAVE_GATE_BLOCK_M_L2=64 compiles the
// decoupled-BM arm (128-row CTA envelope, GEMM1 at 128, GEMM2 sub-tiled at 64).
#ifndef WEAVE_GATE_T_BLOCK_M
#define WEAVE_GATE_T_BLOCK_M 64
#endif
#ifndef WEAVE_GATE_T_STORE_BLOCK_M
#define WEAVE_GATE_T_STORE_BLOCK_M 32
#endif
#ifndef WEAVE_GATE_T_STAGES
#define WEAVE_GATE_T_STAGES 9
#endif
#ifndef WEAVE_GATE_BLOCK_M_L1
#define WEAVE_GATE_BLOCK_M_L1 64
#endif
#ifndef WEAVE_GATE_BLOCK_M_L2
#define WEAVE_GATE_BLOCK_M_L2 64
#endif
#define WEAVE_BLOCK_M_L1 WEAVE_GATE_BLOCK_M_L1
#define WEAVE_BLOCK_M_L2 WEAVE_GATE_BLOCK_M_L2
// The host emits dyn-epi 2 as a 16-row store height; the static arms emit the
// compiled STORE_BLOCK_M (32).  Must agree with the kernel's STORE_BLOCK_M.
#ifndef WEAVE_STORE_BLOCK_M_L1
#define WEAVE_STORE_BLOCK_M_L1 WEAVE_GATE_T_STORE_BLOCK_M
#endif
// The host JIT recomputes this from its SMEM planner for the 8-warp epilogue;
// the compile check only needs a value that covers sizeof(SharedStorage).
#ifndef WEAVE_HOST_SMEM_BYTES
#define WEAVE_HOST_SMEM_BYTES 232448u
#endif
#define WEAVE_DISPATCH_ACTIVE_SMS 0
#ifndef WEAVE_SCATTER_OWNER
#define WEAVE_SCATTER_OWNER 1
#endif
#ifndef WEAVE_SCATTER_STAGING
#define WEAVE_SCATTER_STAGING 1
#endif
#ifndef WEAVE_COMBINE_SYNC_CHUNK
#define WEAVE_COMBINE_SYNC_CHUNK 32
#endif
#define WEAVE_EPI_REG_STAGES 1
#define WEAVE_SCHED_L2_RATIO 1
#define WEAVE_REG_BALANCE 0
// Default to the production (non-instrumented) kernel so the 0-spill gate below
// matches what ships; pass -DDG_MEGA_TILE_PROF=1 for the tile-profiler build,
// which carries its own known (pre-existing) register spills.
#ifndef DG_MEGA_TILE_PROF
#define DG_MEGA_TILE_PROF 0
#endif

#define WEAVE_READY_GROUP_SIZE 4
#define WEAVE_WARP_SPLIT_MODE 0
#define WEAVE_REG_PULL_BYTES 1024u
#define WEAVE_HETERO_SM_LIMIT 74u
#define WEAVE_DISP_REGS 0u
#define WEAVE_EPI_REGS 0u
#define WEAVE_BLOCK_M 0
#define WEAVE_BLOCK_N 0
#define WEAVE_BLOCK_K 0
#ifndef WEAVE_STORE_BLOCK_M
#define WEAVE_STORE_BLOCK_M 32
#endif
#ifndef WEAVE_STORE_BLOCK_M_L2
#define WEAVE_STORE_BLOCK_M_L2 32
#endif
#ifndef WEAVE_DYN_EPILOGUE
#define WEAVE_DYN_EPILOGUE 0
#endif
#ifndef WEAVE_DYN_TMA_A
#define WEAVE_DYN_TMA_A 0
#endif
#define WEAVE_OVERLAP_EXPERT_SMEM 0
#define WEAVE_EXPERT_STEP_MODE 0
#define WEAVE_ACCUM_STAGES 2u
#define WEAVE_TMA_STORE_STAGES 2u
#define WEAVE_SCHED_STAGES 2u
#ifndef DG_D3_HAS_EXTRA_TMAS
#define DG_D3_HAS_EXTRA_TMAS 0
#endif
#define WEAVE_EXPERTS_PER_WAVE 0
#define WEAVE_STATIC_EXPERT_ORDER 0
#define WEAVE_READY_QUEUE 0
#define WEAVE_SCHED_MODE 0

// Sub-tiled GEMM2 (WEAVE_BLOCK_M_L2 < BLOCK_M) is implemented on the default
// scheduler path only and cannot carry D1 group metadata, D5 dedup or a D6
// N-tile group > 1, so the decoupled-BM arm compiles the clean default block.
#ifdef WEAVE_GATE_BM_DECOUPLED
#undef WEAVE_READY_GROUP_SIZE
#undef WEAVE_STORE_BLOCK_M
#undef WEAVE_STORE_BLOCK_M_L2
#undef WEAVE_DYN_EPILOGUE
#undef WEAVE_DYN_TMA_A
#undef DG_D3_HAS_EXTRA_TMAS
#undef WEAVE_COMBINE_SYNC_CHUNK
#undef WEAVE_COMBINE_GROUP_TILES
#undef WEAVE_EARLY_STREAM_REDUCE
#define WEAVE_READY_GROUP_SIZE 0
#define WEAVE_STORE_BLOCK_M 16
#define WEAVE_STORE_BLOCK_M_L2 16
#define WEAVE_DYN_EPILOGUE 2
#define WEAVE_DYN_TMA_A 2
#define DG_D3_HAS_EXTRA_TMAS 1
#define WEAVE_COMBINE_SYNC_CHUNK 0
#define WEAVE_COMBINE_GROUP_TILES 0
#define WEAVE_EARLY_STREAM_REDUCE 0
#endif

#ifndef WEAVE_DEDUP_MODE
#define WEAVE_DEDUP_MODE 2
#endif
#define WEAVE_DEDUP_SCHED 1
#ifndef WEAVE_COMBINE_GROUP_TILES
#define WEAVE_COMBINE_GROUP_TILES 2u
#endif
// The host heuristic picks 16 for the WEAVE_REDUCE_WARPS=0 arm (legacy all-warp
// SMEM combine) and 0 for the spatial streaming arms; -D selects either here.
#ifndef WEAVE_COMBINE_REDUCE_WARPS
#define WEAVE_COMBINE_REDUCE_WARPS 0u
#endif
// Scatter-executor mode block: the default is the SMEM handoff (mode 1); the
// -D overrides switch the same instantiation to the HBM scatter queue (mode 3)
// or the epilogue-owned reg_2stage path (mode 0) so the WEAVE_LOCAL_SCATTER_DEDUP
// pre-Reduce copies compile too.
#ifndef WEAVE_SCATTER_MODE
#define WEAVE_SCATTER_MODE 1
#endif
#ifndef WEAVE_SCATTER_SMEM_ALIAS
#define WEAVE_SCATTER_SMEM_ALIAS 1
#endif
#define WEAVE_SCATTER_WARPS 4
#define WEAVE_SCATTER_HANDOFF_WAIT 1
#define WEAVE_SCATTER_ACTIVE_SMS 0

#define WEAVE_SHARED_L2_EARLY_BLOCKS 10u
#define WEAVE_ENABLE_MULTICAST 1
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        7680,
        7168, 2048,
        256, 0,
        8,
        WEAVE_GATE_T_BLOCK_M, 128, 128,
        WEAVE_GATE_T_STORE_BLOCK_M,
        128, 128,
        122880,
        1966080,
        WEAVE_GATE_T_STAGES,
        1024,
        128, 128, 256,   // 4 dispatch + 4 MMA + 8 epilogue warps = 512 threads
        148, 4,
        0x1.4p+3f,
        true
    >);
};