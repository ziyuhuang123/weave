#pragma once

#ifndef WEAVE_LOCALITY_DOMAINS
#define WEAVE_LOCALITY_DOMAINS 0
#endif

// Verified architectural knobs consumed by this translation unit.  Host
// dispatchers always define them explicitly; the defaults keep direct header
// users on the historical behavior.
#ifndef WEAVE_DISPATCH_WARPS
#define WEAVE_DISPATCH_WARPS 0
#endif
#ifndef WEAVE_SM0_STREAM_SCHED
#define WEAVE_SM0_STREAM_SCHED 0
#endif
#ifndef WEAVE_L2_KBLOCK_DEP
#define WEAVE_L2_KBLOCK_DEP 0
#endif
#ifndef WEAVE_RING_BLOCKS
#define WEAVE_RING_BLOCKS 0
#endif
#ifndef WEAVE_EARLY_STREAM_REDUCE
#define WEAVE_EARLY_STREAM_REDUCE 0
#endif
#ifndef WEAVE_TMEM_PIPELINE_MODE
#define WEAVE_TMEM_PIPELINE_MODE 0
#endif
#ifndef WEAVE_HEAD_PULL_CHUNK
#define WEAVE_HEAD_PULL_CHUNK 0
#endif
#ifndef WEAVE_LOCAL_SCATTER_DEDUP
#define WEAVE_LOCAL_SCATTER_DEDUP 0
#endif
#ifndef WEAVE_ADAPT_BM
#define WEAVE_ADAPT_BM 0
#endif
#ifndef WEAVE_BLOCK_M_L1
#define WEAVE_BLOCK_M_L1 0
#endif
#ifndef WEAVE_BLOCK_M_L2
#define WEAVE_BLOCK_M_L2 0
#endif
#ifndef WEAVE_STORE_BLOCK_M_L1
#define WEAVE_STORE_BLOCK_M_L1 0
#endif

// Reset both per-domain queues even when the build does not use them; the cost
// is four stores once per launch and keeps the workspace byte layout stable.
static constexpr unsigned int kNumTaskCountDomains = (WEAVE_LOCALITY_DOMAINS ? 2u : 1u);

// ============================================================================
// Public B300 compile-time knobs. WEAVE_* is the primary spelling; archived
// DG_* names are accepted only as source-level compatibility aliases.
// ============================================================================
#ifndef WEAVE_READY_GROUP_SIZE
#  if defined(DG_TUNE_D1_G)
#    define WEAVE_READY_GROUP_SIZE DG_TUNE_D1_G
#  elif defined(DG_ISOLATED_D1_G)
#    define WEAVE_READY_GROUP_SIZE DG_ISOLATED_D1_G
#  else
#    define WEAVE_READY_GROUP_SIZE 0
#  endif
#endif
#ifndef DG_TUNE_D1_G
#define DG_TUNE_D1_G WEAVE_READY_GROUP_SIZE
#endif
#ifndef DG_ISOLATED_D1_G
#define DG_ISOLATED_D1_G WEAVE_READY_GROUP_SIZE
#endif

#ifndef WEAVE_WARP_SPLIT_MODE
#  if defined(DG_TUNE_D2_MODE)
#    define WEAVE_WARP_SPLIT_MODE DG_TUNE_D2_MODE
#  elif defined(DG_ISOLATED_D2_MODE)
#    define WEAVE_WARP_SPLIT_MODE DG_ISOLATED_D2_MODE
#  else
#    define WEAVE_WARP_SPLIT_MODE 0
#  endif
#endif
#ifndef DG_TUNE_D2_MODE
#define DG_TUNE_D2_MODE WEAVE_WARP_SPLIT_MODE
#endif
#ifndef DG_ISOLATED_D2_MODE
#define DG_ISOLATED_D2_MODE WEAVE_WARP_SPLIT_MODE
#endif

#ifndef WEAVE_REG_PULL_BYTES
#  if defined(DG_TUNE_D2_REG_PULL_BYTES)
#    define WEAVE_REG_PULL_BYTES DG_TUNE_D2_REG_PULL_BYTES
#  elif defined(DG_ISOLATED_D2_REG_PULL_BYTES)
#    define WEAVE_REG_PULL_BYTES DG_ISOLATED_D2_REG_PULL_BYTES
#  else
#    define WEAVE_REG_PULL_BYTES 0u
#  endif
#endif
#ifndef DG_TUNE_D2_REG_PULL_BYTES
#define DG_TUNE_D2_REG_PULL_BYTES WEAVE_REG_PULL_BYTES
#endif
#ifndef DG_ISOLATED_D2_REG_PULL_BYTES
#define DG_ISOLATED_D2_REG_PULL_BYTES WEAVE_REG_PULL_BYTES
#endif

#ifndef WEAVE_HETERO_SM_LIMIT
#  if defined(DG_ISOLATED_D2_D8_SM_LIMIT)
#    define WEAVE_HETERO_SM_LIMIT DG_ISOLATED_D2_D8_SM_LIMIT
#  else
#    define WEAVE_HETERO_SM_LIMIT 74u
#  endif
#endif
#ifndef DG_ISOLATED_D2_D8_SM_LIMIT
#define DG_ISOLATED_D2_D8_SM_LIMIT WEAVE_HETERO_SM_LIMIT
#endif

#ifndef WEAVE_DISP_REGS
#  if defined(DG_ISOLATED_D2_DISP_REGS)
#    define WEAVE_DISP_REGS DG_ISOLATED_D2_DISP_REGS
#  else
#    define WEAVE_DISP_REGS 0u
#  endif
#endif
#ifndef DG_ISOLATED_D2_DISP_REGS
#define DG_ISOLATED_D2_DISP_REGS WEAVE_DISP_REGS
#endif

#ifndef WEAVE_EPI_REGS
#  if defined(DG_ISOLATED_D2_EPI_REGS)
#    define WEAVE_EPI_REGS DG_ISOLATED_D2_EPI_REGS
#  else
#    define WEAVE_EPI_REGS 0u
#  endif
#endif
#ifndef DG_ISOLATED_D2_EPI_REGS
#define DG_ISOLATED_D2_EPI_REGS WEAVE_EPI_REGS
#endif

#ifndef WEAVE_BLOCK_M
#  if defined(DG_TUNE_D3_BM)
#    define WEAVE_BLOCK_M DG_TUNE_D3_BM
#  elif defined(DG_D3_BM)
#    define WEAVE_BLOCK_M DG_D3_BM
#  else
#    define WEAVE_BLOCK_M 0
#  endif
#endif
#ifndef DG_TUNE_D3_BM
#define DG_TUNE_D3_BM WEAVE_BLOCK_M
#endif
#ifndef DG_D3_BM
#define DG_D3_BM WEAVE_BLOCK_M
#endif

#ifndef WEAVE_BLOCK_N
#  if defined(DG_TUNE_D3_BN)
#    define WEAVE_BLOCK_N DG_TUNE_D3_BN
#  elif defined(DG_D3_BN)
#    define WEAVE_BLOCK_N DG_D3_BN
#  else
#    define WEAVE_BLOCK_N 0
#  endif
#endif
#ifndef DG_TUNE_D3_BN
#define DG_TUNE_D3_BN WEAVE_BLOCK_N
#endif
#ifndef DG_D3_BN
#define DG_D3_BN WEAVE_BLOCK_N
#endif

#ifndef WEAVE_BLOCK_K
#  if defined(DG_TUNE_D3_BK)
#    define WEAVE_BLOCK_K DG_TUNE_D3_BK
#  elif defined(DG_D3_BK)
#    define WEAVE_BLOCK_K DG_D3_BK
#  else
#    define WEAVE_BLOCK_K 0
#  endif
#endif
#ifndef DG_TUNE_D3_BK
#define DG_TUNE_D3_BK WEAVE_BLOCK_K
#endif
#ifndef DG_D3_BK
#define DG_D3_BK WEAVE_BLOCK_K
#endif

#ifndef WEAVE_STORE_BLOCK_M
#  if defined(DG_TUNE_D3_STORE_M)
#    define WEAVE_STORE_BLOCK_M DG_TUNE_D3_STORE_M
#  elif defined(DG_D3_STORE_M)
#    define WEAVE_STORE_BLOCK_M DG_D3_STORE_M
#  else
#    define WEAVE_STORE_BLOCK_M 0
#  endif
#endif
#ifndef DG_TUNE_D3_STORE_M
#define DG_TUNE_D3_STORE_M WEAVE_STORE_BLOCK_M
#endif
#ifndef DG_D3_STORE_M
#define DG_D3_STORE_M WEAVE_STORE_BLOCK_M
#endif

// Asymmetric L2 epilogue store block.  0 = mirror WEAVE_STORE_BLOCK_M.
#ifndef WEAVE_STORE_BLOCK_M_L2
#  if defined(DG_TUNE_D3_STORE_M_L2)
#    define WEAVE_STORE_BLOCK_M_L2 DG_TUNE_D3_STORE_M_L2
#  elif defined(DG_D3_STORE_M_L2)
#    define WEAVE_STORE_BLOCK_M_L2 DG_D3_STORE_M_L2
#  else
#    define WEAVE_STORE_BLOCK_M_L2 0
#  endif
#endif
#ifndef DG_TUNE_D3_STORE_M_L2
#define DG_TUNE_D3_STORE_M_L2 WEAVE_STORE_BLOCK_M_L2
#endif
#ifndef DG_D3_STORE_M_L2
#define DG_D3_STORE_M_L2 WEAVE_STORE_BLOCK_M_L2
#endif

#ifndef WEAVE_DYN_EPILOGUE
#  if defined(DG_TUNE_D3_DYN_EPI)
#    define WEAVE_DYN_EPILOGUE DG_TUNE_D3_DYN_EPI
#  elif defined(DG_D3_DYN_EPI)
#    define WEAVE_DYN_EPILOGUE DG_D3_DYN_EPI
#  else
#    define WEAVE_DYN_EPILOGUE 0
#  endif
#endif
#ifndef DG_TUNE_D3_DYN_EPI
#define DG_TUNE_D3_DYN_EPI WEAVE_DYN_EPILOGUE
#endif
#ifndef DG_D3_DYN_EPI
#define DG_D3_DYN_EPI WEAVE_DYN_EPILOGUE
#endif

#ifndef WEAVE_DYN_TMA_A
#  if defined(DG_TUNE_D3_DYN_TMA_A)
#    define WEAVE_DYN_TMA_A DG_TUNE_D3_DYN_TMA_A
#  elif defined(DG_D3_DYN_TMA_A)
#    define WEAVE_DYN_TMA_A DG_D3_DYN_TMA_A
#  else
#    define WEAVE_DYN_TMA_A 0
#  endif
#endif
#ifndef DG_TUNE_D3_DYN_TMA_A
#define DG_TUNE_D3_DYN_TMA_A WEAVE_DYN_TMA_A
#endif
#ifndef DG_D3_DYN_TMA_A
#define DG_D3_DYN_TMA_A WEAVE_DYN_TMA_A
#endif

#ifndef WEAVE_OVERLAP_EXPERT_SMEM
#  if defined(DG_TUNE_D3_OVERLAP_SMEM)
#    define WEAVE_OVERLAP_EXPERT_SMEM DG_TUNE_D3_OVERLAP_SMEM
#  elif defined(DG_D3_OVERLAP_EXPERT_SMEM)
#    define WEAVE_OVERLAP_EXPERT_SMEM DG_D3_OVERLAP_EXPERT_SMEM
#  else
#    define WEAVE_OVERLAP_EXPERT_SMEM 0
#  endif
#endif
#ifndef DG_TUNE_D3_OVERLAP_SMEM
#define DG_TUNE_D3_OVERLAP_SMEM WEAVE_OVERLAP_EXPERT_SMEM
#endif
#ifndef DG_D3_OVERLAP_EXPERT_SMEM
#define DG_D3_OVERLAP_EXPERT_SMEM WEAVE_OVERLAP_EXPERT_SMEM
#endif

#ifndef WEAVE_EXPERT_STEP_MODE
#  if defined(DG_TUNE_D3_EXPERT_STEP_MODE)
#    define WEAVE_EXPERT_STEP_MODE DG_TUNE_D3_EXPERT_STEP_MODE
#  elif defined(DG_D3_EXPERT_STEP_MODE)
#    define WEAVE_EXPERT_STEP_MODE DG_D3_EXPERT_STEP_MODE
#  else
#    define WEAVE_EXPERT_STEP_MODE 0
#  endif
#endif
#ifndef DG_TUNE_D3_EXPERT_STEP_MODE
#define DG_TUNE_D3_EXPERT_STEP_MODE WEAVE_EXPERT_STEP_MODE
#endif
#ifndef DG_D3_EXPERT_STEP_MODE
#define DG_D3_EXPERT_STEP_MODE WEAVE_EXPERT_STEP_MODE
#endif

#ifndef WEAVE_ACCUM_STAGES
#  if defined(DG_D3_ACCUM_STAGES)
#    define WEAVE_ACCUM_STAGES DG_D3_ACCUM_STAGES
#  else
#    define WEAVE_ACCUM_STAGES 2u
#  endif
#endif
#ifndef DG_D3_ACCUM_STAGES
#define DG_D3_ACCUM_STAGES WEAVE_ACCUM_STAGES
#endif

#ifndef WEAVE_TMA_STORE_STAGES
#  if defined(DG_D3_TMA_STORE_STAGES)
#    define WEAVE_TMA_STORE_STAGES DG_D3_TMA_STORE_STAGES
#  else
#    define WEAVE_TMA_STORE_STAGES 2u
#  endif
#endif
#ifndef DG_D3_TMA_STORE_STAGES
#define DG_D3_TMA_STORE_STAGES WEAVE_TMA_STORE_STAGES
#endif

#ifndef WEAVE_SCHED_STAGES
#  if defined(DG_TUNE_D3_SCHED_STAGES)
#    define WEAVE_SCHED_STAGES DG_TUNE_D3_SCHED_STAGES
#  elif defined(DG_D3_SCHED_STAGES)
#    define WEAVE_SCHED_STAGES DG_D3_SCHED_STAGES
#  else
#    define WEAVE_SCHED_STAGES 2u
#  endif
#endif
#ifndef DG_TUNE_D3_SCHED_STAGES
#define DG_TUNE_D3_SCHED_STAGES WEAVE_SCHED_STAGES
#endif
#ifndef DG_D3_SCHED_STAGES
#define DG_D3_SCHED_STAGES WEAVE_SCHED_STAGES
#endif

#ifndef WEAVE_EXPERTS_PER_WAVE
#  if defined(DG_TUNE_D4_EPW)
#    define WEAVE_EXPERTS_PER_WAVE DG_TUNE_D4_EPW
#  elif defined(DG_ISOLATED_D4_EPW)
#    define WEAVE_EXPERTS_PER_WAVE DG_ISOLATED_D4_EPW
#  else
#    define WEAVE_EXPERTS_PER_WAVE 0
#  endif
#endif
#ifndef DG_TUNE_D4_EPW
#define DG_TUNE_D4_EPW WEAVE_EXPERTS_PER_WAVE
#endif
#ifndef DG_ISOLATED_D4_EPW
#define DG_ISOLATED_D4_EPW WEAVE_EXPERTS_PER_WAVE
#endif

#ifndef WEAVE_STATIC_EXPERT_ORDER
#  if defined(DG_TUNE_D4_STATIC)
#    define WEAVE_STATIC_EXPERT_ORDER DG_TUNE_D4_STATIC
#  elif defined(DG_ISOLATED_D4_STATIC)
#    define WEAVE_STATIC_EXPERT_ORDER DG_ISOLATED_D4_STATIC
#  else
#    define WEAVE_STATIC_EXPERT_ORDER 0
#  endif
#endif
#ifndef DG_TUNE_D4_STATIC
#define DG_TUNE_D4_STATIC WEAVE_STATIC_EXPERT_ORDER
#endif
#ifndef DG_ISOLATED_D4_STATIC
#define DG_ISOLATED_D4_STATIC WEAVE_STATIC_EXPERT_ORDER
#endif

#ifndef WEAVE_READY_QUEUE
#  if defined(DG_TUNE_D4_READYQ)
#    define WEAVE_READY_QUEUE DG_TUNE_D4_READYQ
#  elif defined(DG_ISOLATED_D4_READYQ)
#    define WEAVE_READY_QUEUE DG_ISOLATED_D4_READYQ
#  else
#    define WEAVE_READY_QUEUE 0
#  endif
#endif
#ifndef DG_TUNE_D4_READYQ
#define DG_TUNE_D4_READYQ WEAVE_READY_QUEUE
#endif
#ifndef DG_ISOLATED_D4_READYQ
#define DG_ISOLATED_D4_READYQ WEAVE_READY_QUEUE
#endif

#ifndef WEAVE_SCHED_MODE
#  if defined(DG_TUNE_D4_MODE)
#    define WEAVE_SCHED_MODE DG_TUNE_D4_MODE
#  elif defined(DG_ISOLATED_D4_MODE)
#    define WEAVE_SCHED_MODE DG_ISOLATED_D4_MODE
#  else
#    define WEAVE_SCHED_MODE 0
#  endif
#endif
#ifndef DG_TUNE_D4_MODE
#define DG_TUNE_D4_MODE WEAVE_SCHED_MODE
#endif
#ifndef DG_ISOLATED_D4_MODE
#define DG_ISOLATED_D4_MODE WEAVE_SCHED_MODE
#endif

#ifndef WEAVE_DEDUP_MODE
#  if defined(DG_TUNE_D5_DEDUP)
#    define WEAVE_DEDUP_MODE DG_TUNE_D5_DEDUP
#  elif defined(DG_ISOLATED_D5_DEDUP)
#    define WEAVE_DEDUP_MODE DG_ISOLATED_D5_DEDUP
#  else
#    define WEAVE_DEDUP_MODE 0
#  endif
#endif
#ifndef DG_TUNE_D5_DEDUP
#define DG_TUNE_D5_DEDUP WEAVE_DEDUP_MODE
#endif
#ifndef DG_ISOLATED_D5_DEDUP
#define DG_ISOLATED_D5_DEDUP WEAVE_DEDUP_MODE
#endif

#ifndef WEAVE_DEDUP_SCHED
#  if defined(DG_TUNE_D5_SCHED)
#    define WEAVE_DEDUP_SCHED DG_TUNE_D5_SCHED
#  elif defined(DG_ISOLATED_D5_SCHED)
#    define WEAVE_DEDUP_SCHED DG_ISOLATED_D5_SCHED
#  else
#    define WEAVE_DEDUP_SCHED 0
#  endif
#endif
#ifndef DG_TUNE_D5_SCHED
#define DG_TUNE_D5_SCHED WEAVE_DEDUP_SCHED
#endif
#ifndef DG_ISOLATED_D5_SCHED
#define DG_ISOLATED_D5_SCHED WEAVE_DEDUP_SCHED
#endif

#ifndef WEAVE_COMBINE_GROUP_TILES
#  if defined(DG_TUNE_D6_G)
#    define WEAVE_COMBINE_GROUP_TILES DG_TUNE_D6_G
#  elif defined(DG_ISOLATED_D6_G)
#    define WEAVE_COMBINE_GROUP_TILES DG_ISOLATED_D6_G
#  else
#    define WEAVE_COMBINE_GROUP_TILES 0u
#  endif
#endif
#ifndef DG_TUNE_D6_G
#define DG_TUNE_D6_G WEAVE_COMBINE_GROUP_TILES
#endif
#ifndef DG_ISOLATED_D6_G
#define DG_ISOLATED_D6_G WEAVE_COMBINE_GROUP_TILES
#endif

#ifndef WEAVE_COMBINE_REDUCE_WARPS
#  if defined(DG_TUNE_D6_REDUCE_WARPS)
#    define WEAVE_COMBINE_REDUCE_WARPS DG_TUNE_D6_REDUCE_WARPS
#  elif defined(DG_ISOLATED_D6_REDUCE_WARPS)
#    define WEAVE_COMBINE_REDUCE_WARPS DG_ISOLATED_D6_REDUCE_WARPS
#  else
#    define WEAVE_COMBINE_REDUCE_WARPS 0u
#  endif
#endif
#ifndef DG_TUNE_D6_REDUCE_WARPS
#define DG_TUNE_D6_REDUCE_WARPS WEAVE_COMBINE_REDUCE_WARPS
#endif
#ifndef DG_ISOLATED_D6_REDUCE_WARPS
#define DG_ISOLATED_D6_REDUCE_WARPS WEAVE_COMBINE_REDUCE_WARPS
#endif

#ifndef WEAVE_SCATTER_MODE
#  if defined(DG_TUNE_D7_SCATTER_MODE)
#    define WEAVE_SCATTER_MODE DG_TUNE_D7_SCATTER_MODE
#  else
#    define WEAVE_SCATTER_MODE 0
#  endif
#endif
#ifndef DG_TUNE_D7_SCATTER_MODE
#define DG_TUNE_D7_SCATTER_MODE WEAVE_SCATTER_MODE
#endif

#ifndef WEAVE_SCATTER_SMEM_ALIAS
#  if defined(DG_TUNE_D7_SMEM_ALIAS)
#    define WEAVE_SCATTER_SMEM_ALIAS DG_TUNE_D7_SMEM_ALIAS
#  else
#    define WEAVE_SCATTER_SMEM_ALIAS 1
#  endif
#endif
#ifndef DG_TUNE_D7_SMEM_ALIAS
#define DG_TUNE_D7_SMEM_ALIAS WEAVE_SCATTER_SMEM_ALIAS
#endif

#ifndef WEAVE_SCATTER_WARPS
#  if defined(DG_TUNE_D7_WARPS)
#    define WEAVE_SCATTER_WARPS DG_TUNE_D7_WARPS
#  elif defined(DG_TUNE_D7_SCATTER_WARPS)
#    define WEAVE_SCATTER_WARPS DG_TUNE_D7_SCATTER_WARPS
#  else
#    define WEAVE_SCATTER_WARPS 0
#  endif
#endif
#ifndef DG_TUNE_D7_WARPS
#define DG_TUNE_D7_WARPS WEAVE_SCATTER_WARPS
#endif
#ifndef DG_TUNE_D7_SCATTER_WARPS
#define DG_TUNE_D7_SCATTER_WARPS WEAVE_SCATTER_WARPS
#endif

#ifndef WEAVE_SCATTER_HANDOFF_WAIT
#  if defined(DG_TUNE_D7_HANDOFF_WAIT)
#    define WEAVE_SCATTER_HANDOFF_WAIT DG_TUNE_D7_HANDOFF_WAIT
#  else
#    define WEAVE_SCATTER_HANDOFF_WAIT 1
#  endif
#endif
#ifndef DG_TUNE_D7_HANDOFF_WAIT
#define DG_TUNE_D7_HANDOFF_WAIT WEAVE_SCATTER_HANDOFF_WAIT
#endif

#ifndef WEAVE_SCATTER_ACTIVE_SMS
#  if defined(DG_D7_ACTIVE_SMS)
#    define WEAVE_SCATTER_ACTIVE_SMS DG_D7_ACTIVE_SMS
#  else
#    define WEAVE_SCATTER_ACTIVE_SMS 0
#  endif
#endif
#ifndef DG_D7_ACTIVE_SMS
#define DG_D7_ACTIVE_SMS WEAVE_SCATTER_ACTIVE_SMS
#endif

#ifndef DG_D7_QUEUE_SCATTER
#define DG_D7_QUEUE_SCATTER 0
#endif

#ifndef WEAVE_SHARED_L2_EARLY_BLOCKS
#  if defined(DG_SHARED_L2_EARLY_BLOCKS)
#    define WEAVE_SHARED_L2_EARLY_BLOCKS DG_SHARED_L2_EARLY_BLOCKS
#  else
#    define WEAVE_SHARED_L2_EARLY_BLOCKS 10u
#  endif
#endif
#ifndef DG_SHARED_L2_EARLY_BLOCKS
#define DG_SHARED_L2_EARLY_BLOCKS WEAVE_SHARED_L2_EARLY_BLOCKS
#endif

#ifndef WEAVE_ENABLE_MULTICAST
#define WEAVE_ENABLE_MULTICAST 1
#endif



#include <cstdint>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <type_traits>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/scheduler/weave_policy_runtime.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

// ============================================================================
// Unified 7-Axis Parameter Normalization (DG_TUNE_* <-> DG_ISOLATED_* / DG_D3_*)
// ============================================================================

#ifndef DG_P1_HEAD_HELPER_K
#define DG_P1_HEAD_HELPER_K 0u
#endif
#ifndef DG_P1_HEAD_TOKENS
#define DG_P1_HEAD_TOKENS 0u
#endif

// Axis 1 (D1): Fine-grained group metadata & flat-slot pull
#if defined(DG_TUNE_D1_G) && !defined(DG_ISOLATED_D1_G)
#define DG_ISOLATED_D1_G DG_TUNE_D1_G
#endif
#ifndef DG_ISOLATED_D1_G
#define DG_ISOLATED_D1_G 0
#endif
#ifndef DG_TUNE_D1_G
#define DG_TUNE_D1_G DG_ISOLATED_D1_G
#endif

// Axis 2 (D2): Warp ratio + Register-Direct Pull
#if defined(DG_TUNE_D2_MODE) && !defined(DG_ISOLATED_D2_MODE)
#define DG_ISOLATED_D2_MODE DG_TUNE_D2_MODE
#endif
#ifndef DG_ISOLATED_D2_MODE
#define DG_ISOLATED_D2_MODE 0
#endif
#ifndef DG_TUNE_D2_MODE
#define DG_TUNE_D2_MODE DG_ISOLATED_D2_MODE
#endif

#if defined(DG_TUNE_D2_REG_PULL_BYTES) && !defined(DG_ISOLATED_D2_REG_PULL_BYTES)
#define DG_ISOLATED_D2_REG_PULL_BYTES DG_TUNE_D2_REG_PULL_BYTES
#endif
#ifndef DG_ISOLATED_D2_REG_PULL_BYTES
#define DG_ISOLATED_D2_REG_PULL_BYTES 0u
#endif
#ifndef DG_TUNE_D2_REG_PULL_BYTES
#define DG_TUNE_D2_REG_PULL_BYTES DG_ISOLATED_D2_REG_PULL_BYTES
#endif

// Host launches this kernel with cluster_size=2 for the 2-CTA multicast path.
constexpr bool kEnableMulticast = (WEAVE_ENABLE_MULTICAST != 0);
DG_STATIC_ASSERT(kEnableMulticast, "B300 FP8xFP4 requires 2-CTA multicast");

#ifndef DG_ISOLATED_D2_D8_SM_LIMIT
#define DG_ISOLATED_D2_D8_SM_LIMIT 74u
#endif
#ifndef DG_ISOLATED_D2_DISP_REGS
#define DG_ISOLATED_D2_DISP_REGS 0u
#endif
#ifndef DG_ISOLATED_D2_EPI_REGS
#define DG_ISOLATED_D2_EPI_REGS 0u
#endif

// Axis 3 (D3): Tile sizes (BM/BN/BK/STORE_M), Dynamic Epilogue, Dynamic TMA-A, SMEM Overlap
#if defined(DG_D3_BM) && !defined(DG_TUNE_D3_BM)
#define DG_TUNE_D3_BM DG_D3_BM
#endif
#ifndef DG_TUNE_D3_BM
#define DG_TUNE_D3_BM 0
#endif

#if defined(DG_D3_BN) && !defined(DG_TUNE_D3_BN)
#define DG_TUNE_D3_BN DG_D3_BN
#endif
#ifndef DG_TUNE_D3_BN
#define DG_TUNE_D3_BN 0
#endif

#if defined(DG_D3_BK) && !defined(DG_TUNE_D3_BK)
#define DG_TUNE_D3_BK DG_D3_BK
#endif
#ifndef DG_TUNE_D3_BK
#define DG_TUNE_D3_BK 0
#endif

#if defined(DG_D3_STORE_M) && !defined(DG_TUNE_D3_STORE_M)
#define DG_TUNE_D3_STORE_M DG_D3_STORE_M
#endif
#ifndef DG_TUNE_D3_STORE_M
#define DG_TUNE_D3_STORE_M 0
#endif

#if defined(DG_TUNE_D3_DYN_EPI) && !defined(DG_D3_DYN_EPI)
#define DG_D3_DYN_EPI DG_TUNE_D3_DYN_EPI
#endif
#ifndef DG_D3_DYN_EPI
#define DG_D3_DYN_EPI 0
#endif
#ifndef DG_TUNE_D3_DYN_EPI
#define DG_TUNE_D3_DYN_EPI DG_D3_DYN_EPI
#endif

#if defined(DG_TUNE_D3_DYN_TMA_A) && !defined(DG_D3_DYN_TMA_A)
#define DG_D3_DYN_TMA_A DG_TUNE_D3_DYN_TMA_A
#endif
#ifndef DG_D3_DYN_TMA_A
#define DG_D3_DYN_TMA_A 0
#endif
#ifndef DG_TUNE_D3_DYN_TMA_A
#define DG_TUNE_D3_DYN_TMA_A DG_D3_DYN_TMA_A
#endif

#if defined(DG_TUNE_D3_OVERLAP_SMEM) && !defined(DG_D3_OVERLAP_EXPERT_SMEM)
#define DG_D3_OVERLAP_EXPERT_SMEM DG_TUNE_D3_OVERLAP_SMEM
#endif
#ifndef DG_D3_OVERLAP_EXPERT_SMEM
#define DG_D3_OVERLAP_EXPERT_SMEM 0
#endif
#ifndef DG_TUNE_D3_OVERLAP_SMEM
#define DG_TUNE_D3_OVERLAP_SMEM DG_D3_OVERLAP_EXPERT_SMEM
#endif

#ifndef DG_D3_ACCUM_STAGES
#define DG_D3_ACCUM_STAGES 2u
#endif
#ifndef DG_D3_TMA_STORE_STAGES
#define DG_D3_TMA_STORE_STAGES 2u
#endif
#ifndef DG_D3_EARLY_ACK_TASK
#define DG_D3_EARLY_ACK_TASK 0
#endif
#ifndef DG_D3_HAS_EXTRA_TMAS
#define DG_D3_HAS_EXTRA_TMAS 0
#endif

// Axis 6 (D6): Fine-grained group combine
#if defined(DG_TUNE_D6_G) && !defined(DG_ISOLATED_D6_G)
#define DG_ISOLATED_D6_G DG_TUNE_D6_G
#endif
#ifndef DG_ISOLATED_D6_G
#define DG_ISOLATED_D6_G 0u
#endif
#ifndef DG_TUNE_D6_G
#define DG_TUNE_D6_G DG_ISOLATED_D6_G
#endif

#if defined(DG_TUNE_D6_REDUCE_WARPS) && !defined(DG_ISOLATED_D6_REDUCE_WARPS)
#define DG_ISOLATED_D6_REDUCE_WARPS DG_TUNE_D6_REDUCE_WARPS
#endif
#ifndef DG_ISOLATED_D6_REDUCE_WARPS
#define DG_ISOLATED_D6_REDUCE_WARPS 0u
#endif

// Tile-Granularity Lock-Free Profiler (zero overhead when DG_MEGA_TILE_PROF == 0)
#if defined(DG_SHARED_L2_EARLY_BLOCKS) && (DG_SHARED_L2_EARLY_BLOCKS == 77u)
#define DG_MEGA_TILE_PROF 1
#endif
#ifndef DG_MEGA_TILE_PROF
#define DG_MEGA_TILE_PROF 0
#endif

#if DG_MEGA_TILE_PROF
namespace deep_gemm::tile_prof {
static constexpr uint32_t kHeaderWords = 1024u;
static constexpr uint32_t kMaxWarpsPerSM = 20u;
static constexpr uint32_t kMaxRecordsPerWarp = 2048u;

CUTLASS_DEVICE uint32_t read_gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return static_cast<uint32_t>(t);
}

CUTLASS_DEVICE void record(
        int* stats_ptr,
        const uint32_t& sm_idx,
        const uint32_t& warp_idx,
        const uint32_t& lane_idx,
        uint32_t& rec_idx,
        const uint32_t& t0_ns,
        const uint32_t& ts_ns,
        const uint32_t& te_ns,
        const uint32_t& stage_id,
        const uint32_t& tile_id) {
    (void)t0_ns;
    if (lane_idx == 0u and stats_ptr != nullptr and warp_idx < kMaxWarpsPerSM and rec_idx < kMaxRecordsPerWarp) {
        const uint32_t slot = (sm_idx * kMaxWarpsPerSM + warp_idx) * kMaxRecordsPerWarp + (rec_idx++);
        uint4* base = reinterpret_cast<uint4*>(stats_ptr + kHeaderWords);
        base[slot] = make_uint4(stage_id, ts_ns, te_ns, tile_id);
    }
}
} // namespace deep_gemm::tile_prof
#endif

// Axis 7 (D7): 3-Mode FC2 Scatter (Mode 0 = Epilogue inline, Mode 1 = Idle Dispatch SMEM handoff, Mode 2 = Local HBM/L2 Dedup/Pre-reduce)
#if defined(DG_TUNE_D7_WARPS) && !defined(DG_TUNE_D7_SCATTER_WARPS)
#define DG_TUNE_D7_SCATTER_WARPS DG_TUNE_D7_WARPS
#endif
#if defined(DG_SCATTER_WARPS) && !defined(DG_TUNE_D7_SCATTER_WARPS)
#define DG_TUNE_D7_SCATTER_WARPS DG_SCATTER_WARPS
#endif
#ifndef DG_TUNE_D7_SCATTER_WARPS
#define DG_TUNE_D7_SCATTER_WARPS 0
#endif
#ifndef DG_TUNE_D7_WARPS
#define DG_TUNE_D7_WARPS DG_TUNE_D7_SCATTER_WARPS
#endif

#ifndef DG_TUNE_D7_SMEM_ALIAS
#define DG_TUNE_D7_SMEM_ALIAS 1
#endif

#ifndef DG_TUNE_D7_HANDOFF_WAIT
#define DG_TUNE_D7_HANDOFF_WAIT 1
#endif

#if defined(DG_TUNE_D7_SCATTER_MODE)
    #if DG_TUNE_D7_SCATTER_MODE == 0
        #define DG_D7_HANDOFF_MODE 0
        #define DG_D7_LOCAL_DEDUP 0
    #elif DG_TUNE_D7_SCATTER_MODE == 1
        #define DG_D7_HANDOFF_MODE DG_TUNE_D7_HANDOFF_WAIT
        #define DG_D7_LOCAL_DEDUP 0
    #elif DG_TUNE_D7_SCATTER_MODE == 2
        #define DG_D7_HANDOFF_MODE 0
        #define DG_D7_LOCAL_DEDUP 1
    #elif DG_TUNE_D7_SCATTER_MODE == 3
        #define DG_D7_HANDOFF_MODE 0
        #define DG_D7_LOCAL_DEDUP 1
        #undef DG_D7_QUEUE_SCATTER
        #define DG_D7_QUEUE_SCATTER 1
    #else
        #define DG_D7_HANDOFF_MODE 0
        #define DG_D7_LOCAL_DEDUP 0
    #endif
    #ifndef DG_ISOLATED_D7_SCATTER
        #define DG_ISOLATED_D7_SCATTER DG_D7_HANDOFF_MODE
    #endif
#else
    #ifndef DG_ISOLATED_D7_SCATTER
        #define DG_ISOLATED_D7_SCATTER 0
    #endif
    #define DG_TUNE_D7_SCATTER_MODE ((DG_ISOLATED_D7_SCATTER != 0) ? 1 : 0)
    #define DG_D7_HANDOFF_MODE DG_ISOLATED_D7_SCATTER
    #define DG_D7_LOCAL_DEDUP 0
#endif

namespace deep_gemm {

#if DG_ISOLATED_D5_DEDUP != 1
template <uint32_t kNumRanksPerLane>
CUTLASS_DEVICE void find_round_robin_pull_source(
    uint32_t (&remaining)[kNumRanksPerLane], const uint32_t token_idx_in_expert,
    uint32_t& current_rank_in_expert_idx, uint32_t& token_idx_in_rank) {
    uint32_t offset = 0;
    uint32_t slot_idx = token_idx_in_expert;
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
        // The per-rank receive-count snapshot can be transiently all-zero for a
        // token that the summed histogram still reports.  Without this guard the
        // selector divides by zero (slot_idx % 0) and wedges the CTA.
        if (num_active_ranks == 0u) {
            current_rank_in_expert_idx = 0u;
            token_idx_in_rank = token_idx_in_expert;
            return;
        }
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
            break;
        }

        slot_idx -= num_round_tokens;
        offset += length;
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
            remaining[i] -= cute::min(remaining[i], length);
    }
}
#endif

struct alignas(16) D7ScatterJobInfo {
    uint32_t pool_m_idx;
    uint32_t n_idx;
    uint32_t packed_meta;
    uint32_t pad;
};

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t T_BLOCK_M, uint32_t T_BLOCK_N, uint32_t T_BLOCK_K,
    uint32_t T_STORE_BLOCK_M,
    uint32_t T_SF_BLOCK_M, uint32_t T_SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t T_kNumStages,
    uint32_t T_kNumBytesPerPull,
    uint32_t T_kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t T_kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t BLOCK_M = (DG_TUNE_D3_BM > 0) ? static_cast<uint32_t>(DG_TUNE_D3_BM) : T_BLOCK_M,
    uint32_t BLOCK_N = (DG_TUNE_D3_BN > 0) ? static_cast<uint32_t>(DG_TUNE_D3_BN) : T_BLOCK_N,
    uint32_t BLOCK_K = (DG_TUNE_D3_BK > 0) ? static_cast<uint32_t>(DG_TUNE_D3_BK) : T_BLOCK_K,
    uint32_t kNumDispatchThreads =
        (T_kNumDispatchThreads == 192u) ? 192u :
        (((2 <= DG_ISOLATED_D2_MODE and DG_ISOLATED_D2_MODE <= 8) or
          (11 <= DG_ISOLATED_D2_MODE and DG_ISOLATED_D2_MODE <= 15) or
          (DG_TUNE_D7_SCATTER_WARPS == 8)) ? 256u : T_kNumDispatchThreads),
    uint32_t kNumEpilogueThreads =
        (1 <= DG_ISOLATED_D2_MODE and DG_ISOLATED_D2_MODE <= 8) ? 256u : T_kNumEpilogueThreads,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
#if DG_D3_HAS_EXTRA_TMAS
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output_m8,
#endif
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
#if DG_D3_HAS_EXTRA_TMAS
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_m16,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_m16,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_m8,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_m8,
#endif
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf,
                            const uint8_t* sm_locality_domains) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks & 6D compile-time rejection
    DG_STATIC_ASSERT(T_kNumDispatchThreads != 192 and kNumDispatchWarps != 6 and
                     (kNumDispatchWarps == 4 or kNumDispatchWarps == 8),
                     "6D is illegal; kNumDispatchWarps must be 4 or 8");
    DG_STATIC_ASSERT(WEAVE_DISPATCH_WARPS == 0 or
                     WEAVE_DISPATCH_WARPS == kNumDispatchWarps,
                     "WEAVE_DISPATCH_WARPS disagrees with the launch thread layout");
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    static constexpr uint32_t WG_BLOCK_M_RAW = BLOCK_M / kNumEpilogueWarpgroups;
    static constexpr uint32_t RAW_STORE_M = (DG_TUNE_D3_STORE_M > 0)
        ? static_cast<uint32_t>(DG_TUNE_D3_STORE_M)
        : T_STORE_BLOCK_M;
    static constexpr uint32_t STORE_BLOCK_M =
        (DG_D3_DYN_EPI == 2) ? 16u : ((RAW_STORE_M <= WG_BLOCK_M_RAW) ? RAW_STORE_M : WG_BLOCK_M_RAW);
    static constexpr uint32_t RAW_STORE_M_L2 = (DG_TUNE_D3_STORE_M_L2 > 0)
        ? static_cast<uint32_t>(DG_TUNE_D3_STORE_M_L2)
        : ((WEAVE_STORE_BLOCK_M_L2 > 0) ? static_cast<uint32_t>(WEAVE_STORE_BLOCK_M_L2) : STORE_BLOCK_M);
    static constexpr uint32_t STORE_BLOCK_M_L2 =
        (DG_D3_DYN_EPI > 0) ? STORE_BLOCK_M
                            : ((RAW_STORE_M_L2 <= WG_BLOCK_M_RAW) ? RAW_STORE_M_L2 : WG_BLOCK_M_RAW);
    static constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, 128u);
    static constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, 128u);
    static constexpr uint32_t kNumBytesPerPull =
        (DG_ISOLATED_D2_REG_PULL_BYTES > 0u) ? static_cast<uint32_t>(DG_ISOLATED_D2_REG_PULL_BYTES) :
        ((T_kNumBytesPerPull > 0u) ? T_kNumBytesPerPull : 4096u);

#if DG_ISOLATED_D5_DEDUP == 2
    DG_STATIC_ASSERT(kNumRanks == 4 or kNumRanks == 8, "D5 dedup layout requires 4 or 8 EP ranks");
    DG_STATIC_ASSERT(kNumExpertsPerRank >= 18, "D5 dedup layout requires >= 18 experts per rank");
    DG_STATIC_ASSERT(kNumTopk <= 10, "D5 dedup requires topk <= 10");
    DG_STATIC_ASSERT(kHidden % 512 == 0, "D5 vectorized SF fanout requires kHidden % 512 == 0");
    DG_STATIC_ASSERT(kNumBytesPerPull % 1024 == 0, "D5 activation fanout requires pull bytes % 1024 == 0");
#endif

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
#if DG_D3_HAS_EXTRA_TMAS
        if constexpr (DG_D3_DYN_EPI == 2)
            cute::prefetch_tma_descriptor(&tensor_map_l1_output_m8);
        if constexpr (DG_D3_DYN_TMA_A >= 1) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_m16);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_m16);
        }
        if constexpr (DG_D3_DYN_TMA_A >= 2) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_m8);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_m8);
        }
#endif
        if constexpr (kHasShared) {
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
        }
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");

    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
    using shared_b_dtype_t = cutlass::float_e4m3_t;

    // MMA configs (supports BN in {64, 128, 256}, BK in {128, 256}, BM in {32, 64, 128, 256})
    constexpr uint32_t LAYOUT_AD_M = (BLOCK_N == 64) ? 64 : 128;
    constexpr uint32_t kNumNSubtiles = BLOCK_N / LAYOUT_AD_M;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 128;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t kNumKSubtilesPerBlock = BLOCK_K / UMMA_BLOCK_K;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == 64 or BLOCK_N == 128 or BLOCK_N == 256, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128 or BLOCK_K == 256, "Invalid block K");
    DG_STATIC_ASSERT(WEAVE_RING_BLOCKS == 0 or WEAVE_RING_BLOCKS == kNumRingBlocks,
                     "WEAVE_RING_BLOCKS must match the allocated ring");
    DG_STATIC_ASSERT(WEAVE_BLOCK_M_L1 == 0 or WEAVE_BLOCK_M_L1 == BLOCK_M,
                     "SM103 has one static BLOCK_M; L1/L2 envelopes must match");
    DG_STATIC_ASSERT(WEAVE_BLOCK_M_L2 == 0 or WEAVE_BLOCK_M_L2 == BLOCK_M,
                     "SM103 has one static BLOCK_M; L1/L2 envelopes must match");
    DG_STATIC_ASSERT(WEAVE_STORE_BLOCK_M_L1 == 0 or WEAVE_STORE_BLOCK_M_L1 == STORE_BLOCK_M,
                     "WEAVE_STORE_BLOCK_M_L1 disagrees with the compiled store tile");
    DG_STATIC_ASSERT(WEAVE_ADAPT_BM == 0 or DG_D3_EXPERT_STEP_MODE > 0,
                     "WEAVE_ADAPT_BM requires the per-expert BM ladder");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;

    // Epilogue & Schedule stages
    constexpr uint32_t kNumAccumColsPerStage = kNumNSubtiles * UMMA_N;
    constexpr uint32_t kNumSFATmemColsPerSubtile = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemColsPerSubtile = SF_BLOCK_N / 32;
    constexpr uint32_t kNumSFATmemCols = kNumSFATmemColsPerSubtile * kNumKSubtilesPerBlock;
    constexpr uint32_t kNumSFBTmemCols = kNumSFBTmemColsPerSubtile * kNumKSubtilesPerBlock;
    constexpr uint32_t kMaxLegalAccumStages = (512u - (kNumSFATmemCols + kNumSFBTmemCols)) / kNumAccumColsPerStage;
    constexpr uint32_t kNumEpilogueStages =
        (DG_D3_ACCUM_STAGES <= kMaxLegalAccumStages) ? DG_D3_ACCUM_STAGES :
        (kMaxLegalAccumStages >= 1u ? kMaxLegalAccumStages : 1u);
    constexpr uint32_t kNumTMAStoreStages = DG_D3_TMA_STORE_STAGES;
    constexpr uint32_t kNumScheduleStages = DG_D3_SCHED_STAGES;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t L1_OUT_BLOCK_N = 64; // 64B swizzle subtile
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M / 2; // float2

    // D2 flags
    static constexpr bool kD2Active = (DG_ISOLATED_D2_MODE >= 1) or (DG_ISOLATED_D2_REG_PULL_BYTES > 0u);
    static constexpr bool kD2Enable8D = (kNumDispatchWarps == 8);
    static constexpr bool kD2AllD8 = kD2Enable8D and (
        DG_ISOLATED_D2_MODE == 0 or DG_ISOLATED_D2_MODE == 2 or DG_ISOLATED_D2_MODE == 5 or
        DG_ISOLATED_D2_MODE == 11 or DG_ISOLATED_D2_MODE == 14);
    static constexpr bool kD2SmidMasked = kD2Enable8D and !kD2AllD8 and (DG_ISOLATED_D1_G == 0) and (DG_ISOLATED_D5_DEDUP != 2);
    static constexpr bool kD2AllRegPull = (DG_ISOLATED_D2_REG_PULL_BYTES > 0u);
    static constexpr bool kD2SplitHybridPull = kD2Active and !kD2AllRegPull and (
        (DG_ISOLATED_D2_MODE >= 5 and DG_ISOLATED_D2_MODE <= 8) or DG_ISOLATED_D2_MODE == 14);
    static constexpr uint32_t kNumTmaDispatchWarps =
        kD2AllRegPull ? 0u : (kD2SplitHybridPull ? 4u : kNumDispatchWarps);
    static constexpr uint32_t kNumDispatchBarrierSlots =
        kD2AllRegPull ? 1u : kNumTmaDispatchWarps;

    // The host pipeline planner computes the exact SMEM footprint (including the
    // asymmetric L1/L2 epilogue buffers) and passes the largest legal stage count.
    // Trust it here: the previous fixed 48 KiB estimate rejected the 6-stage
    // BM=240 + 24/8 configuration even though its real fixed footprint is ~29 KiB.
    static constexpr uint32_t kNumStages = T_kNumStages;
    DG_STATIC_ASSERT(kNumStages >= 2u, "Invalid pipeline stage count");

    struct SharedStorage {
#if DG_D3_OVERLAP_EXPERT_SMEM && ((DG_D7_HANDOFF_MODE == 0) || (DG_TUNE_D7_SMEM_ALIAS == 0))
        union {
            alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
    #if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
            alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumTmaDispatchWarps][kNumBytesPerPull];
    #endif
        };
#else
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
    #if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumTmaDispatchWarps][kNumBytesPerPull];
    #elif (DG_D7_HANDOFF_MODE != 0) && (DG_TUNE_D7_SMEM_ALIAS != 0)
        alignas(kSharedMemoryAlignment) uint8_t d7_scatter_smem[8192];
    #endif
#endif
#if (DG_D7_HANDOFF_MODE != 0) && (DG_TUNE_D7_SMEM_ALIAS == 0)
        alignas(kSharedMemoryAlignment) uint8_t d7_dedicated_scatter_smem[2u * kNumEpilogueWarpgroups * 16u * 256u];
#endif
        union {
            alignas(kSharedMemoryAlignment) cutlass::float_e4m3_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M_L2 * LAYOUT_AD_M];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchBarrierSlots];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
#if DG_D7_HANDOFF_MODE != 0
        D7ScatterJobInfo scatter_jobs[2];
        Barrier scatter_full_barriers[2];
        Barrier scatter_empty_barriers[2];
        uint32_t epi_scatter_mode[4];
#endif
#if (DG_D7_HANDOFF_MODE != 0) || (DG_D7_QUEUE_SCATTER != 0)
        uint32_t dispatch_pull_done;
#endif
#if DG_ISOLATED_D2_MODE >= 1
        uint32_t d2_warp_base;
        uint32_t d2_total_warps;
#endif
        // Weave CTA-0 decision, copied once per CTA after the live histogram
        // publication barrier. Kept outside every DeepGEMM task-info slot.
        weave::PolicyDecision weave_policy;
        uint32_t weave_policy_ready;
        uint32_t weave_epoch_base;
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    if (thread_idx == 0u) {
        auto* const control = workspace.get_weave_policy_control_ptr();
        shared_storage.weave_epoch_base =
            *reinterpret_cast<volatile uint32_t*>(&control->published_epoch);
        shared_storage.weave_policy_ready = 0u;
    }

#if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumTmaDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));
#endif

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = kNumAccumColsPerStage * kNumEpilogueStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Physical %smid mask helper (for D2 %smid modes)
    const auto d2_check_is_d8_sm = [&]() -> bool {
        if constexpr (!kD2Enable8D) {
            return false;
        } else if constexpr (kD2AllD8) {
            return true;
        } else {
            uint32_t phys_smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(phys_smid));
            if constexpr (DG_ISOLATED_D2_MODE == 3 or DG_ISOLATED_D2_MODE == 6 or DG_ISOLATED_D2_MODE == 12) {
                return (phys_smid & 1u) == 0u;
            } else if constexpr (DG_ISOLATED_D2_MODE == 4 or DG_ISOLATED_D2_MODE == 7 or DG_ISOLATED_D2_MODE == 13) {
                return phys_smid < (kNumSMs / 2u);
            } else if constexpr (DG_ISOLATED_D2_MODE == 8 or DG_ISOLATED_D2_MODE == 15) {
                return ((phys_smid >> 1u) & 1u) == 0u;
            } else {
                return phys_smid < DG_ISOLATED_D2_D8_SM_LIMIT;
            }
        }
    };
    const auto get_active_dispatch_threads = [&]() -> uint32_t {
        if constexpr (!kD2SmidMasked) {
            return kNumDispatchThreads;
        } else {
            return d2_check_is_d8_sm() ? 256u : 128u;
        }
    };

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        if (cute::elect_one_sync()) {
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
#if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumTmaDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
#else
        if (lane_idx == 0)
            shared_storage.dispatch_barriers[0].init(1);
#endif
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                shared_storage.tmem_full_barriers[i].init(1);
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
#if (DG_D7_HANDOFF_MODE != 0) || (DG_D7_QUEUE_SCATTER != 0)
            shared_storage.dispatch_pull_done = 0u;
#endif
#if DG_D7_HANDOFF_MODE != 0
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i) {
                shared_storage.epi_scatter_mode[i] = 0u;
            }
            const uint32_t d7_init_disp_t = get_active_dispatch_threads();
            #pragma unroll
            for (uint32_t i = 0; i < 2; ++ i) {
                shared_storage.scatter_full_barriers[i].init(kNumEpilogueThreads);
                shared_storage.scatter_empty_barriers[i].init(d7_init_disp_t);
            }
#endif
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    comm::cluster_sync_with_relaxed_arrive();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;
    constexpr uint32_t kAllWarpReduceBarrierIdx = 7;
#if (DG_D7_HANDOFF_MODE != 0) || (DG_D7_QUEUE_SCATTER != 0)
    constexpr uint32_t kScatterDrainBarrierIdx = 6;
    DG_STATIC_ASSERT(kScatterDrainBarrierIdx >= kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups and
                     kScatterDrainBarrierIdx < 16, "Invalid scatter drain barrier index");
    DG_STATIC_ASSERT(kNumEpilogueWarpgroups <= 2, "D7 scatter supports 1 or 2 epilogue warpgroups");
#if DG_D7_HANDOFF_MODE != 0
    #if DG_TUNE_D7_SMEM_ALIAS == 0
    constexpr uint32_t kD7DeadDispatchBytes = sizeof(SharedStorage::d7_dedicated_scatter_smem);
    #else
    constexpr uint32_t kD7DeadDispatchBytes = offsetof(SharedStorage, smem_d);
    #endif
    constexpr uint32_t kD7FitChunkM32 =
        (STORE_BLOCK_M % 32u == 0 and 2u * kNumEpilogueWarpgroups * 32u * 256u <= kD7DeadDispatchBytes) ? 32u : 0u;
    constexpr uint32_t kD7FitChunkM16 =
        (STORE_BLOCK_M % 16u == 0 and 2u * kNumEpilogueWarpgroups * 16u * 256u <= kD7DeadDispatchBytes) ? 16u : 0u;
    constexpr uint32_t kD7FitChunkM8 =
        (STORE_BLOCK_M % 8u == 0 and 2u * kNumEpilogueWarpgroups * 8u * 256u <= kD7DeadDispatchBytes) ? 8u : 0u;
    constexpr bool kD7UseDualDeadSlots = (kD7FitChunkM32 > 0u or kD7FitChunkM16 > 0u or kD7FitChunkM8 > 0u);
    constexpr uint32_t kD7DualWgChunkM =
        kD7FitChunkM32 ? 32u : (kD7FitChunkM16 ? 16u : 8u);
    constexpr bool kD7AllowDynamicWg0Full32 =
        (kNumEpilogueWarpgroups == 2u and STORE_BLOCK_M == 32u and kD7DualWgChunkM == 16u and
         2u * 32u * 256u <= kD7DeadDispatchBytes);
    constexpr uint32_t kD7SlotBytes =
        kD7AllowDynamicWg0Full32 ? (32u * 256u) : (kNumEpilogueWarpgroups * kD7DualWgChunkM * 256u);
    constexpr uint32_t kD7MaxSlotRows =
        kD7AllowDynamicWg0Full32 ? 32u : (kNumEpilogueWarpgroups * kD7DualWgChunkM);
    const auto d7_get_scatter_base_ptr = [&]() -> uint8_t* {
        #if DG_TUNE_D7_SMEM_ALIAS == 0
        return shared_storage.d7_dedicated_scatter_smem;
        #else
        return smem_buffer;
        #endif
    };
#endif
#endif

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Grid sync index assignments
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // D6 group tile count
    constexpr uint32_t kD6RawGroupTiles = DG_ISOLATED_D6_G;
    constexpr uint32_t kD6GroupTiles =
        (kD6RawGroupTiles > (L2_SHAPE_N / BLOCK_N)) ? (L2_SHAPE_N / BLOCK_N) : kD6RawGroupTiles;
    constexpr uint32_t kD6SafeDiv = (kD6GroupTiles > 0u) ? kD6GroupTiles : 1u;
    constexpr uint32_t kD6ReduceWarps = DG_ISOLATED_D6_REDUCE_WARPS;
    DG_STATIC_ASSERT(WEAVE_EARLY_STREAM_REDUCE == 0 or kD6GroupTiles > 0u,
                     "WEAVE_EARLY_STREAM_REDUCE requires the fine-grained D6 combine");
    static constexpr bool kD7LocalDedupActive =
        ((DG_D7_LOCAL_DEDUP != 0) or (WEAVE_LOCAL_SCATTER_DEDUP != 0)) and
        (kD6GroupTiles == 0u);
    DG_STATIC_ASSERT(WEAVE_LOCAL_SCATTER_DEDUP == 0 or DG_D7_QUEUE_SCATTER != 0,
                     "WEAVE_LOCAL_SCATTER_DEDUP requires the HBM scatter-queue path");
    // Queue scatter is an epilogue-local-dedup path with a single dispatch-side
    // NVLink writer pool.  It is only legal for 128-column epilogue subtiles.
    static constexpr bool kD7QueueScatterActive =
        (DG_D7_QUEUE_SCATTER != 0) and (kD6GroupTiles == 0u) and (LAYOUT_AD_M == 128u);
    static constexpr uint32_t kD7QueueActiveSms =
        (DG_D7_ACTIVE_SMS <= 0 or static_cast<uint32_t>(DG_D7_ACTIVE_SMS) >= kNumSMs)
            ? kNumSMs : static_cast<uint32_t>(DG_D7_ACTIVE_SMS);
    static constexpr uint32_t kD7QueueReserveBatch = 64u;
    DG_STATIC_ASSERT(kD6GroupTiles == 0u or
                     ((kD6GroupTiles & (kD6GroupTiles - 1u)) == 0u and
                      (L2_SHAPE_N / BLOCK_N) % kD6SafeDiv == 0u),
                     "Invalid D6 group tile count");

    // Zero-spill register allocation across all 7 axes (strictly <= 64,512 total per SM)
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    static constexpr bool kD2AnyRegPull = kD2AllRegPull or kD2SplitHybridPull;
    static constexpr bool kD2Is20Warp = (kNumDispatchWarps == 8u and kNumEpilogueWarps == 8u);
    static constexpr bool kAnyAxisActive =
        (DG_ISOLATED_D1_G > 0) or kD2Active or kD2Enable8D or
        (DG_TUNE_D3_BM > 0) or (DG_TUNE_D3_BN > 0) or (DG_TUNE_D3_BK > 0) or
        (DG_TUNE_D3_STORE_M > 0) or (DG_D3_DYN_EPI > 0) or (DG_D3_DYN_TMA_A > 0) or
        (DG_D3_OVERLAP_EXPERT_SMEM > 0) or DG_ISOLATED_D4_LUT_ACTIVE or
        (DG_ISOLATED_D5_DEDUP > 0) or (DG_ISOLATED_D6_G > 0) or
        (DG_D7_HANDOFF_MODE != 0) or (DG_D7_LOCAL_DEDUP != 0);

    static constexpr bool kSchedNeedsMoreRegs = DG_ISOLATED_D4_LUT_ACTIVE or (DG_D3_EXPERT_STEP_MODE > 0);
#if DG_MEGA_TILE_PROF
    static constexpr uint32_t kAutoNonEpiRegs = kD2Is20Warp
        ? (kUseMoreEpilogueRegisters ? 48u : 56u)
        : (kUseMoreEpilogueRegisters ? 56u : 88u);
    static constexpr uint32_t kAutoDispRegs = kD2Is20Warp
        ? (kUseMoreEpilogueRegisters ? 64u : 72u)
        : (kUseMoreEpilogueRegisters ? 64u : 88u);
#else
    static constexpr uint32_t kAutoNonEpiRegs = !kAnyAxisActive
        ? (kUseMoreEpilogueRegisters ? 40u : 88u)
        : (kD2Is20Warp
            ? (kUseMoreEpilogueRegisters ? (kSchedNeedsMoreRegs ? 48u : 40u) : (kD2AnyRegPull ? 40u : 56u))
            : (kUseMoreEpilogueRegisters ? (kSchedNeedsMoreRegs ? 56u : 40u) : 88u));

    static constexpr uint32_t kAutoDispRegs = !kAnyAxisActive
        ? (kUseMoreEpilogueRegisters ? 48u : 96u)
        : (kD2Is20Warp
            ? (!kUseMoreEpilogueRegisters ? (kD2AnyRegPull ? 72u : 64u) :
               (DG_ISOLATED_D4_LUT_ACTIVE ? 64u : (kD2AnyRegPull ? 72u : 56u)))
            : (!kUseMoreEpilogueRegisters ? 88u :
               (kNumEpilogueWarps == 4u ? ((kD2AnyRegPull or DG_ISOLATED_D1_G > 0 or DG_ISOLATED_D5_DEDUP > 0 or DG_ISOLATED_D7_SCATTER != 0) ? 80u : 64u) :
                ((kD2AnyRegPull or DG_ISOLATED_D1_G > 0 or DG_ISOLATED_D5_DEDUP > 0) ? 72u : 56u))));
#endif

    static constexpr uint32_t kMaxRemainingEpiRegs =
        ((64512u - kAutoDispRegs * kNumDispatchThreads - kAutoNonEpiRegs * kNumNonEpilogueThreads) / kNumEpilogueThreads) & ~7u;
    static constexpr uint32_t kTargetEpiRegs = !kAnyAxisActive
        ? (kUseMoreEpilogueRegisters ? 208u : 160u)
        : (kD2Is20Warp ? 168u : (kUseMoreEpilogueRegisters ? 208u : 160u));
    static constexpr uint32_t kAutoEpiRegs = (kTargetEpiRegs <= kMaxRemainingEpiRegs) ? kTargetEpiRegs : kMaxRemainingEpiRegs;

#if DG_P1_HEAD_HELPER_K > 0
    DG_STATIC_ASSERT(kNumDispatchThreads == 128 and kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128,
                     "P1 head pull requires 4 dispatch, 4 task, and 4 epilogue warps");
    DG_STATIC_ASSERT(DG_ISOLATED_D1_G == 0 and not kD2Active and
                     DG_ISOLATED_D2_DISP_REGS == 0 and DG_ISOLATED_D2_EPI_REGS == 0 and
                     DG_D3_EXPERT_STEP_MODE == 0 and DG_ISOLATED_D4_READYQ == 0 and DG_ISOLATED_D5_DEDUP == 0,
                     "Unsupported metadata, register, or ready queue configuration for P1 head pull");
    constexpr uint32_t kNumDispatchRegisters = 168u;
    constexpr uint32_t kNumNonEpilogueRegisters = 72u;
    constexpr uint32_t kNumEpilogueRegisters = 208u;
#else
    constexpr uint32_t kNumDispatchRegisters = (DG_ISOLATED_D2_DISP_REGS > 0u) ? DG_ISOLATED_D2_DISP_REGS : kAutoDispRegs;
    constexpr uint32_t kNumNonEpilogueRegisters = kAutoNonEpiRegs;
    constexpr uint32_t kNumEpilogueRegisters = (DG_ISOLATED_D2_EPI_REGS > 0u) ? DG_ISOLATED_D2_EPI_REGS : kAutoEpiRegs;
#endif
    DG_STATIC_ASSERT(kNumDispatchRegisters % 8 == 0 and kNumNonEpilogueRegisters % 8 == 0 and kNumEpilogueRegisters % 8 == 0,
                     "warpgroup_reg_alloc/dealloc requires multiples of 8");
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

#if DG_MEGA_TILE_PROF
    constexpr uint32_t prof_t0_ns = 0u;
    uint32_t prof_rec_idx = 0u;
#endif

#if DG_P1_HEAD_HELPER_K > 0
    // The existing 128-byte workspace header has free padding after the
    // schedule counters and the D5 telemetry at bytes 64..87.
    constexpr uint32_t kP1ControlOffset = 96u;
    DG_STATIC_ASSERT(kP1ControlOffset + 2u * sizeof(uint32_t) <= layout::Workspace::kNumBarrierSignalBytes,
                     "P1 control exceeds workspace header");
    auto* const p1_cursor = math::advance_ptr<uint32_t>(workspace.base, kP1ControlOffset);
    auto* const p1_published_limit = p1_cursor + 1u;

    // All warps rendezvous after dispatch finishes publishing cross-rank metadata.
    const auto p1_wait_for_metadata = [&]() {
        asm volatile("barrier.sync 14, %0;" :: "n"(kNumThreads) : "memory");
        uint32_t limit = 0u;
        if (lane_idx == 0u) {
            do {
                limit = ptx::ld_acq(p1_published_limit);
            } while (limit == 0u);
        }
        return __shfl_sync(0xffffffffu, limit, 0) - 1u;
    };

    const auto p1_count_head_tokens = [&]() {
        constexpr uint64_t kHelperCapacity =
            static_cast<uint64_t>(DG_P1_HEAD_HELPER_K) *
            (kNumMMANonEpilogueWarps + kNumEpilogueWarps) * kNumSMs;
        constexpr uint32_t kRequested = DG_P1_HEAD_TOKENS > 0u ? DG_P1_HEAD_TOKENS :
            static_cast<uint32_t>(kHelperCapacity < 0xffffffffull ? kHelperCapacity : 0xffffffffull);
        uint32_t tokens = 0u, used_blocks = 0u;
        for (uint32_t base = 0u; base < kNumExpertsPerRank and
             used_blocks < kNumRingBlocks and tokens < kRequested; base += 32u) {
            const uint32_t expert = base + lane_idx;
            const uint32_t count = expert < kNumExpertsPerRank ?
                static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(expert)) : 0u;
            const uint32_t blocks = math::ceil_div(count, BLOCK_M);
            const uint32_t end = math::warp_inclusive_sum(blocks, lane_idx);
            const uint32_t start = used_blocks + end - blocks;
            // Ring reuse depends on GEMM, which the helpers have yet to start.
            const uint32_t available = start < kNumRingBlocks ? (kNumRingBlocks - start) * BLOCK_M : 0u;
            tokens += __reduce_add_sync(0xffffffffu, cute::min(count, available));
            used_blocks += __shfl_sync(0xffffffffu, end, 31);
        }
        return cute::min(tokens, kRequested);
    };

    const auto run_p1_head_pull = [&]<uint32_t kRegisters, bool kDispatch>(const uint32_t head_end) {
        constexpr uint32_t kVecPerWave = kRegisters == 72u ? 2u : (kRegisters == 168u ? 4u : 8u);
        constexpr uint32_t kWaveBytes = kVecPerWave * 32u * sizeof(uint4);
        constexpr uint32_t kFullWaves = kHidden / kWaveBytes;
        constexpr uint32_t kTailVectors = (kHidden % kWaveBytes) / sizeof(uint4);
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        constexpr uint32_t kNumSFUint32 = kHidden / 128u;
        DG_STATIC_ASSERT(kRegisters == 72u or kRegisters == 168u or kRegisters == 208u,
                         "Invalid P1 register budget");
        DG_STATIC_ASSERT(kHidden % 128u == 0u, "Invalid P1 activation width");

        uint32_t expert_idx = 0u, expert_start = 0u, expert_end = 0u, pool_block_base = 0u;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        for (uint32_t pulled = 0u; head_end > 0u and (kDispatch or pulled < DG_P1_HEAD_HELPER_K); ++pulled) {
            uint32_t token = 0u;
            if (lane_idx == 0u)
                token = atomicAdd(p1_cursor, 1u);
            token = __shfl_sync(0xffffffffu, token, 0);
            if (token >= head_end)
                break;

#if DG_MEGA_TILE_PROF
            const uint64_t prof_pull_s = tile_prof::read_gtimer();
#endif
            if (token >= expert_end) {
                do {
                    pool_block_base += math::ceil_div(expert_end - expert_start, BLOCK_M);
                    expert_start = expert_end;
                    uint32_t count = 0u;
                    if (lane_idx == 0u)
                        count = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(expert_idx));
                    expert_end += __shfl_sync(0xffffffffu, count, 0);
                    ++expert_idx;
                } while (token >= expert_end);

                #pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    const uint32_t rank = i * 32u + lane_idx;
                    stored_rank_count[i] = rank < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(rank, expert_idx - 1u)) : 0u;
                }
            }

            const uint32_t token_idx_in_expert = token - expert_start;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0u; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];
            uint32_t src_rank, token_idx_in_rank;
            find_round_robin_pull_source(remaining, token_idx_in_expert, src_rank, token_idx_in_rank);

            const uint32_t src_token_topk_raw = *workspace.get_src_token_topk_idx_ptr(
                expert_idx - 1u, src_rank, token_idx_in_rank);
            const uint32_t src_token_topk_idx = src_token_topk_raw & 0x0003ffffu;
            const uint32_t d7_topk_bm = kD7LocalDedupActive ? ((src_token_topk_raw >> 18) & 0x3ffu) : 0u;
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = (src_token_topk_idx % kNumTopk) | (d7_topk_bm << 8);
            const uint32_t pool_token_idx = pool_block_base * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
            const auto* src = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint4>(), src_rank);
            auto* dst = buffer.l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr<uint4>();

            #pragma unroll 1
            for (uint32_t wave = 0u; wave < kFullWaves; ++wave) {
                const uint32_t base = wave * kVecPerWave * 32u + lane_idx;
                uint4 values[kVecPerWave];
                #pragma unroll
                for (uint32_t v = 0u; v < kVecPerWave; ++v) {
                    auto& value = values[v];
                    asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                                 : "l"(src + base + v * 32u));
                }
                #pragma unroll
                for (uint32_t v = 0u; v < kVecPerWave; ++v) {
                    const auto& value = values[v];
                    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                 :: "l"(dst + base + v * 32u),
                                    "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
                }
            }
            // Keep the same load-then-store ordering in a partial wave.
            if constexpr (kTailVectors > 0u) {
                constexpr uint32_t kTailVecPerLane = math::constexpr_ceil_div(kTailVectors, 32u);
                const uint32_t base = kFullWaves * kVecPerWave * 32u + lane_idx;
                uint4 values[kTailVecPerLane];
                #pragma unroll
                for (uint32_t v = 0u; v < kTailVecPerLane; ++v) {
                    if (v * 32u + lane_idx < kTailVectors) {
                        auto& value = values[v];
                        asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                                     : "l"(src + base + v * 32u));
                    }
                }
                #pragma unroll
                for (uint32_t v = 0u; v < kTailVecPerLane; ++v) {
                    if (v * 32u + lane_idx < kTailVectors) {
                        const auto& value = values[v];
                        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                     :: "l"(dst + base + v * 32u),
                                        "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
                    }
                }
            }

            const auto* remote_sf = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(), src_rank);
            auto* local_sf = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t sf_ring_token_idx =
                pool_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert % BLOCK_M);
            #pragma unroll
            for (uint32_t i = 0u; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++i) {
                const uint32_t j = i * 32u + lane_idx;
                if (j < kNumSFUint32)
                    local_sf[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf[j];
            }

            if (lane_idx == 0u) {
                const auto* remote_weight = sym_buffer.map(
                    buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, src_rank);
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = *remote_weight;
                *workspace.get_token_src_metadata_ptr(pool_token_idx) = {src_rank, src_token_idx, src_topk_idx};
            }

            // Publish the padded block-ready count after the stores by all lanes.
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            __syncwarp();
            if (lane_idx == 0u) {
                const uint32_t full_delta =
                    token == expert_end - 1u ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u;
                ptx::red_add_rel(workspace.get_l1_full_count_ptr(pool_block_idx), full_delta);
            }
            __syncwarp();
#if DG_MEGA_TILE_PROF
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_pull_s, tile_prof::read_gtimer(),
                              3u, pool_block_idx);
#endif
        }
    };

    if (warp_idx >= kNumDispatchWarps and warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        run_p1_head_pull.template operator()<kNumNonEpilogueRegisters, false>(p1_wait_for_metadata());
    }
#endif

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        if constexpr (kD2SmidMasked) {
            if (!d2_check_is_d8_sm() and warp_idx >= 4) {
                cutlass::arch::warpgroup_reg_dealloc<24>();
                return;
            }
        }
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
#if DG_MEGA_TILE_PROF
        prof_rec_idx = 0u;
#endif

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumScanWarps = kD2Enable8D ? 4u : kNumDispatchWarps;
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            if (warp_idx < kNumScanWarps) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumScanWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumScanWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            __ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    }
                    uint32_t d7_packed_mask = 0u;
                    if constexpr (kD7LocalDedupActive) {
                        const uint32_t tok_grp = lane_idx / kNumTopk;
                        const uint32_t dst_r = (expert_idx >= 0)
                            ? (static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank)
                            : (0xffff0000u | lane_idx);
                        const uint32_t same_m = __match_any_sync(0xffffffffu, (tok_grp << 8) | dst_r);
                        const uint32_t topk_bm = (same_m >> (tok_grp * kNumTopk)) & ((1u << kNumTopk) - 1u);
                        d7_packed_mask = topk_bm << 18;
                    }
                    if (expert_idx >= 0)
                        process((i * kNumTopk + lane_idx) | d7_packed_mask, expert_idx);
                    __syncwarp();
                }
            }
        };

        if constexpr (kD6GroupTiles > 0) {
            if (*buffer.get_d6_epoch_ptr() >= 65535u) {
                const uint32_t total_u32 = buffer.get_d6_ready_epoch_num_u32();
                uint32_t* const ready_u32 = buffer.get_d6_ready_epoch_u32_ptr();
                const uint32_t act_disp_t = get_active_dispatch_threads();
                for (uint32_t idx = sm_idx * act_disp_t + thread_idx;
                     idx < total_u32;
                     idx += kNumSMs * act_disp_t) {
                    ready_u32[idx] = 0u;
                }
                ptx::fence_acq_rel_sys();
            }
        }

        // Helper lambda for token chunk + SF + weight pull (supports both SMEM TMA and D2 Zero-SMEM Register-Direct Pull)
        const auto execute_token_pull = [&](const uint32_t& token_idx,
                                            const uint32_t& current_expert_idx,
                                            const uint32_t& current_rank_in_expert_idx,
                                            const uint32_t& src_token_topk_idx,
                                            const uint32_t& src_token_idx,
                                            const uint32_t& src_topk_idx,
                                            const uint32_t& token_idx_in_expert,
                                            const uint32_t& expert_end_idx,
                                            const uint32_t& pool_token_idx,
                                            const uint32_t& pool_block_idx,
                                            uint32_t& pull_mbarrier_phase) {
            (void)current_expert_idx;
#if DG_MEGA_TILE_PROF
            const uint64_t prof_pull_s = tile_prof::read_gtimer();
#endif
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");

            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();

            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const bool use_reg_pull = kD2AllRegPull or (kD2SplitHybridPull and (warp_idx >= 4));

            if (!use_reg_pull) {
#if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
                const uint32_t tma_warp_slot = warp_idx < kNumTmaDispatchWarps ? warp_idx : 0u;
                const auto pull_buffer = smem_send_buffers.get_rank_buffer(tma_warp_slot).get_data_buffer(0);
                const auto pull_mbarrier = &shared_storage.dispatch_barriers[tma_warp_slot];
                const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                    ptx::tma_store_1d(
                        math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                        pull_buffer.get_base_ptr(), kNumBytesPerPull
                    );
                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                };
#if DG_ISOLATED_D5_DEDUP == 1
                // Issue Chunk 0 TMA load + remote SF loads + remote weight load simultaneously to collapse NVLink RTTs
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        src_base_ptr,
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                }

                constexpr uint32_t kNumSFIters = math::constexpr_ceil_div(kNumSFUint32, 32u);
                const auto remote_sf_ptr = sym_buffer.map(
                    buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                    current_rank_in_expert_idx);
                uint32_t sf_regs[kNumSFIters];
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFIters; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    sf_regs[i] = j < kNumSFUint32 ? remote_sf_ptr[j] : 0u;
                }

                float weight_reg = 0.0f;
                if (cute::elect_one_sync()) {
                    weight_reg = *sym_buffer.map(
                        buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    #pragma unroll
                    for (uint32_t i = 0; i + 1 < kNumChunks; ++ i) {
                        issue_and_wait_pull_store(i);
                        ptx::tma_load_1d(
                            pull_buffer.get_base_ptr(),
                            math::advance_ptr(src_base_ptr, (i + 1) * kNumBytesPerPull),
                            pull_mbarrier, kNumBytesPerPull
                        );
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    }
                }
                __syncwarp();

                const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
                const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
                const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
                const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                    transform_sf_token_idx(token_idx_in_block);
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFIters; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint32)
                        local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = sf_regs[i];
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight_reg;
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    issue_and_wait_pull_store(kNumChunks - 1);
#else
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumChunks; ++ i) {
                        ptx::tma_load_1d(
                            pull_buffer.get_base_ptr(),
                            math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                            pull_mbarrier, kNumBytesPerPull
                        );
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                        i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                    }
                }
                __syncwarp();

                const auto remote_sf_ptr = sym_buffer.map(
                    buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
                const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
                const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
                const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                    transform_sf_token_idx(token_idx_in_block);
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint32)
                        local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    issue_and_wait_pull_store(kNumChunks - 1);
#endif
                    const bool is_last_token = (token_idx == expert_end_idx - 1);
                    const uint32_t full_delta = is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u;
                    #if DG_ISOLATED_D4_READYQ
                    if (scheduler.num_total_m_blocks <= kNumRingBlocks) {
                        uint32_t prev_full = 0u;
                        uint32_t* full_ptr = workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks);
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                                     : "=r"(prev_full) : "l"(full_ptr), "r"(full_delta) : "memory");
                        if (prev_full + full_delta == BLOCK_M) {
                            const uint32_t q_slot = ptx::atomic_add(workspace.get_d4_ready_write_count_ptr(), 1u);
                            const uint32_t m_in_exp = token_idx_in_expert / BLOCK_M;
                            const uint32_t packed = 0x80000000u |
                                ((static_cast<uint32_t>(current_expert_idx) & 0x7fu) << 24) |
                                ((m_in_exp & 0xfffu) << 12) |
                                (pool_block_idx & 0xfffu);
                            uint32_t* q_ptr = workspace.get_d4_ready_queue_ptr(q_slot);
                            asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                         :: "l"(q_ptr), "r"(packed) : "memory");
                        }
                    } else {
                        ptx::red_add_rel(workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), full_delta);
                    }
                    #else
                    ptx::red_add_rel(workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), full_delta);
                    #endif
                }
                __syncwarp();
#endif
            } else if constexpr (kD2AnyRegPull) {
                // Zero-SMEM Register-Direct Pull (auto-caps per-iteration register wave to 1024B when D1/D7/20-warp active for 0 spill)
                static constexpr bool kTightDispRegs =
                    kD2Is20Warp or (DG_ISOLATED_D1_G > 0) or (DG_ISOLATED_D7_SCATTER != 0);
                static constexpr uint32_t kEffectiveRegPullBytes =
                    kTightDispRegs ? 1024u : (kD2AllRegPull ? kNumBytesPerPull : (kNumEpilogueWarps == 8u ? 1024u : 4096u));
                static constexpr uint32_t kVecPerRegChunk = kEffectiveRegPullBytes / (32u * sizeof(uint4));
                static constexpr uint32_t kNumRegChunks = kHidden / kEffectiveRegPullBytes;
                static constexpr uint32_t kSFPerLane = math::constexpr_ceil_div(kNumSFUint32, 32u);
                DG_STATIC_ASSERT(kEffectiveRegPullBytes % (32u * sizeof(uint4)) == 0, "Invalid reg pull chunk bytes");
                DG_STATIC_ASSERT(kHidden % kEffectiveRegPullBytes == 0, "kEffectiveRegPullBytes must divide kHidden");

                const auto* src_u4 = reinterpret_cast<const uint4*>(src_base_ptr);
                auto* dst_u4 = reinterpret_cast<uint4*>(dst_base_ptr);

                #pragma unroll 1
                for (uint32_t c = 0; c + 1u < kNumRegChunks; ++c) {
                    const uint32_t chunk_u4 = c * (kEffectiveRegPullBytes / sizeof(uint4));
                    uint4 regs[kVecPerRegChunk];
                    #pragma unroll
                    for (uint32_t v = 0; v < kVecPerRegChunk; ++v) {
                        asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(regs[v].x), "=r"(regs[v].y), "=r"(regs[v].z), "=r"(regs[v].w)
                                     : "l"(src_u4 + chunk_u4 + v * 32u + lane_idx));
                    }
                    #pragma unroll
                    for (uint32_t v = 0; v < kVecPerRegChunk; ++v) {
                        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                     :: "l"(dst_u4 + chunk_u4 + v * 32u + lane_idx),
                                        "r"(regs[v].x), "r"(regs[v].y), "r"(regs[v].z), "r"(regs[v].w)
                                     : "memory");
                    }
                }

                constexpr uint32_t kLastChunkU4 = (kNumRegChunks - 1u) * (kEffectiveRegPullBytes / sizeof(uint4));
                uint4 last_regs[kVecPerRegChunk];
                #pragma unroll
                for (uint32_t v = 0; v < kVecPerRegChunk; ++v) {
                    asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(last_regs[v].x), "=r"(last_regs[v].y), "=r"(last_regs[v].z), "=r"(last_regs[v].w)
                                 : "l"(src_u4 + kLastChunkU4 + v * 32u + lane_idx));
                }

                const auto remote_sf_ptr = sym_buffer.map(
                    buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                    current_rank_in_expert_idx);
                uint32_t sf_regs[kSFPerLane] = {};
                #pragma unroll
                for (uint32_t i = 0; i < kSFPerLane; ++i) {
                    const uint32_t j = i * 32u + lane_idx;
                    if (j < kNumSFUint32) {
                        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(sf_regs[i]) : "l"(remote_sf_ptr + j));
                    }
                }

                float remote_weight = 0.0f;
                if (lane_idx == 0) {
                    const auto* remote_w_ptr = sym_buffer.map(
                        buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(remote_weight) : "l"(remote_w_ptr));
                }

                #pragma unroll
                for (uint32_t v = 0; v < kVecPerRegChunk; ++v) {
                    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                 :: "l"(dst_u4 + kLastChunkU4 + v * 32u + lane_idx),
                                    "r"(last_regs[v].x), "r"(last_regs[v].y), "r"(last_regs[v].z), "r"(last_regs[v].w)
                                 : "memory");
                }

                const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
                const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
                const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
                const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                    transform_sf_token_idx(token_idx_in_block);
                #pragma unroll
                for (uint32_t i = 0; i < kSFPerLane; ++i) {
                    const uint32_t j = i * 32u + lane_idx;
                    if (j < kNumSFUint32) {
                        asm volatile("st.global.cg.u32 [%0], %1;"
                                     :: "l"(local_sf_ptr + j * kNumSFRingTokens + sf_ring_token_idx), "r"(sf_regs[i])
                                     : "memory");
                    }
                }

                if (lane_idx == 0) {
                    *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = remote_weight;
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                }

                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                __syncwarp();
                if (lane_idx == 0) {
                    const bool is_last_token = (token_idx == expert_end_idx - 1);
                    const uint32_t full_delta = is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u;
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                        full_delta
                    );
                }
                __syncwarp();
            }
#if DG_MEGA_TILE_PROF
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_pull_s, tile_prof::read_gtimer(),
                              3u, pool_block_idx);
#endif
        };

#if (DG_ISOLATED_D1_G > 0) && (DG_ISOLATED_D5_DEDUP != 2)
        // ====================================================================
        // Axis 1 (D1): Fine-grained group metadata & flat per-expert pull
        // ====================================================================
#if DG_MEGA_TILE_PROF
        const uint64_t prof_open_s = tile_prof::read_gtimer();
#endif
        constexpr uint32_t kD1GroupBlocks = static_cast<uint32_t>(DG_ISOLATED_D1_G);
        constexpr uint32_t kD1GroupTokens = kD1GroupBlocks * BLOCK_M;
        constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
        DG_STATIC_ASSERT(kD1GroupBlocks == 1 or kD1GroupBlocks == 2 or kD1GroupBlocks == 4 or
                         kD1GroupBlocks == 8 or kD1GroupBlocks == 16, "Invalid DG_ISOLATED_D1_G");
        DG_STATIC_ASSERT(kNumRanks * kNumMaxTokensPerRank < (1u << 20), "Too many recv tokens for 20-bit slot");

        const uint32_t cta_first_token = sm_idx * kNumScanWarps * kNumTokensPerWarp;
        if (cta_first_token < num_tokens) {
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1u);
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t local_cnt = shared_storage.expert_token_count[i];
                if (local_cnt > 0) {
                    const uint32_t dst_rank_idx = i / kNumExpertsPerRank;
                    const uint32_t dst_local_expert = i % kNumExpertsPerRank;
                    const uint32_t base_slot = sched::d1_ptx::atomic_add_sys_u32(
                        sym_buffer.map(workspace.get_d1_expert_write_cursor_ptr(dst_local_expert), dst_rank_idx),
                        local_cnt);
                    shared_storage.expert_token_count[i] = base_slot & 0xfffffu;
                }
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const uint32_t dst_rank_idx = static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
                const uint32_t dst_local_expert = static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank;
                const uint32_t old_packed = atomicAdd_block(
                    shared_storage.expert_token_count + expert_idx, 1u << 20);
                const uint32_t dst_slot_idx = (old_packed & 0xfffffu) + (old_packed >> 20);
                const uint32_t desc = (static_cast<uint32_t>(sym_buffer.rank_idx) << 28) | (token_topk_idx & 0x0fffffffu);
                *sym_buffer.map(workspace.get_src_token_topk_idx_ptr(dst_local_expert, 0, dst_slot_idx), dst_rank_idx) = desc;
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t packed = shared_storage.expert_token_count[i];
                const uint32_t local_cnt = packed >> 20;
                if (local_cnt > 0) {
                    const uint32_t base_slot = packed & 0xfffffu;
                    const uint32_t end_slot = base_slot + local_cnt;
                    const uint32_t dst_rank_idx = i / kNumExpertsPerRank;
                    const uint32_t dst_local_expert = i % kNumExpertsPerRank;
                    const uint32_t g0 = base_slot / kD1GroupTokens;
                    const uint32_t g0_end = (g0 + 1u) * kD1GroupTokens;
                    const uint32_t add0 = cute::min(end_slot, g0_end) - base_slot;
                    const uint32_t old0 = sched::d1_ptx::atomic_add_rel_sys_u32(
                        sym_buffer.map(workspace.get_d1_group_meta_count_ptr(dst_local_expert, g0), dst_rank_idx),
                        add0);
                    if (old0 + add0 == kD1GroupTokens) {
                        ptx::red_or_rel_sys(
                            sym_buffer.map(workspace.get_d1_ready_bitmap_word_ptr(dst_local_expert, g0 >> 6), dst_rank_idx),
                            1ull << (g0 & 63u));
                    }
                    if (end_slot > g0_end) {
                        const uint32_t g1 = g0 + 1u;
                        const uint32_t add1 = end_slot - g0_end;
                        const uint32_t old1 = sched::d1_ptx::atomic_add_rel_sys_u32(
                            sym_buffer.map(workspace.get_d1_group_meta_count_ptr(dst_local_expert, g1), dst_rank_idx),
                            add1);
                        if (old1 + add1 == kD1GroupTokens) {
                            ptx::red_or_rel_sys(
                                sym_buffer.map(workspace.get_d1_ready_bitmap_word_ptr(dst_local_expert, g1 >> 6), dst_rank_idx),
                                1ull << (g1 & 63u));
                        }
                    }
                }
            }
        }

#if DG_D7_QUEUE_SCATTER != 0
        if constexpr (kD7QueueScatterActive) {
            if (sm_idx == 0u and thread_idx == 0u) {
                auto* const queue_epoch_ptr = buffer.get_d7_queue_epoch_ptr();
                uint32_t next_epoch = *reinterpret_cast<volatile uint32_t*>(queue_epoch_ptr);
                next_epoch = (next_epoch == 0xffffffffu) ? 1u : (next_epoch + 1u);
                *queue_epoch_ptr = next_epoch;
                *buffer.get_d7_queue_tail_ptr() = 0u;
                *buffer.get_d7_queue_head_ptr() = 0u;
                *buffer.get_d7_queue_producer_done_ptr() = 0u;
                *buffer.get_d7_queue_consumer_done_ptr() = 0u;
            }
        }
#endif
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
#if DG_MEGA_TILE_PROF
        const uint64_t prof_sync_s = tile_prof::read_gtimer();
        tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                          prof_rec_idx, prof_t0_ns, prof_open_s, prof_sync_s, 1u, 0u);
#endif

        if (sm_idx == 0) {
            if constexpr (kD6GroupTiles > 0) {
                if (thread_idx == 0) {
                    const uint32_t prev_ep = *buffer.get_d6_epoch_ptr();
                    *buffer.get_d6_epoch_ptr() = (prev_ep >= 65535u) ? 1u : (prev_ep + 1u);
                }
            }
            auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
            const auto status = (*counter_ptr) & 3;
            const auto signal_phase = status & 1, signal_sign = status >> 1;
            auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

            if (thread_idx < kNumRanks)
                ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            if (thread_idx == 0) {
                ptx::red_add(counter_ptr, 1);
                const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
                const auto start_clock = clock64();
                while (ptx::ld_acq_sys(signal_ptr) != target) {
                    if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                        DG_DEVICE_ASSERT(false and "D1 NVLink barrier timeout");
                    }
                }
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            #pragma unroll
            for (uint32_t e = thread_idx; e < kNumExpertsPerRank; e += kNumDispatchThreads) {
                const uint32_t total_tok = *workspace.get_d1_expert_write_cursor_ptr(e);
                const uint32_t rem = total_tok % kD1GroupTokens;
                if (rem > 0) {
                    const uint32_t g_tail = total_tok / kD1GroupTokens;
                    ptx::red_or_rel_gpu(
                        workspace.get_d1_ready_bitmap_word_ptr(e, g_tail >> 6),
                        1ull << (g_tail & 63u));
                }
                const uint64_t pub_val =
                    (static_cast<uint64_t>(kNumSMs * kNumRanks) << 32) |
                    static_cast<uint64_t>(total_tok);
                sched::d1_ptx::st_rel_gpu_u64(workspace.get_expert_recv_count_sum_ptr(e), pub_val);
            }
        }

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#if DG_MEGA_TILE_PROF
        tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                          prof_rec_idx, prof_t0_ns, prof_sync_s, tile_prof::read_gtimer(), 2u, 0u);
#endif

        uint32_t pull_mbarrier_phase = 0;
        uint32_t expert_tok_end[kNumExpertsPerLane];
        uint32_t expert_blk_offset[kNumExpertsPerLane];
        uint32_t running_tok = 0;
        uint32_t running_blk = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            uint32_t tok = 0;
            if (expert_idx < kNumExpertsPerRank) {
                tok = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(expert_idx));
                const uint64_t bmap = *workspace.get_d1_ready_bitmap_word_ptr(expert_idx, 0);
                const uint32_t ready_groups = static_cast<uint32_t>(__popcll(bmap));
                tok = cute::min(tok, ready_groups * kD1GroupTokens);
            }
            scheduler.stored_num_tokens_per_expert[i] = tok;
            const uint32_t blks = math::ceil_div(tok, BLOCK_M);
            const uint32_t incl_tok = math::warp_inclusive_sum(tok, lane_idx);
            const uint32_t incl_blk = math::warp_inclusive_sum(blks, lane_idx);
            expert_tok_end[i] = running_tok + incl_tok;
            expert_blk_offset[i] = running_blk + incl_blk - blks;
            running_tok += ptx::exchange(incl_tok, 31);
            running_blk += ptx::exchange(incl_blk, 31);
        }
        const uint32_t total_recv_tokens = running_tok;

        constexpr uint32_t kPullNumSms =
            WEAVE_SM0_STREAM_SCHED ? (kNumSMs - 1u) : kNumSMs;
        constexpr uint32_t kNumGlobalWarps = kPullNumSms * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;
             token_idx < total_recv_tokens;
             token_idx += kNumGlobalWarps) {
            uint32_t current_expert_idx = 0;
            uint32_t expert_start_idx = 0;
            uint32_t expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;
            bool found = false;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                const uint32_t hit_mask = __ballot_sync(0xffffffff, not found and token_idx < expert_tok_end[i]);
                if (hit_mask) {
                    const uint32_t owner_lane = static_cast<uint32_t>(__ffs(hit_mask) - 1);
                    const uint32_t lane_exp = i * 32 + lane_idx;
                    const uint32_t lane_start = expert_tok_end[i] - scheduler.stored_num_tokens_per_expert[i];
                    current_expert_idx = ptx::exchange(lane_exp, owner_lane);
                    expert_start_idx = ptx::exchange(lane_start, owner_lane);
                    expert_end_idx = ptx::exchange(expert_tok_end[i], owner_lane);
                    expert_pool_block_offset = ptx::exchange(expert_blk_offset[i], owner_lane);
                    found = true;
                }
            }

            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            const uint32_t desc = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, 0, token_idx_in_expert);
            const uint32_t current_rank_in_expert_idx = desc >> 28;
            const uint32_t src_token_topk_idx = desc & 0x0003ffffu;
            const uint32_t d7_topk_bm = kD7LocalDedupActive ? ((desc >> 18) & 0x3ffu) : 0u;
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = (src_token_topk_idx % kNumTopk) | (d7_topk_bm << 8);
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            execute_token_pull(
                token_idx, current_expert_idx, current_rank_in_expert_idx,
                src_token_topk_idx, src_token_idx, src_topk_idx,
                token_idx_in_expert, expert_end_idx, pool_token_idx, pool_block_idx,
                pull_mbarrier_phase);
        }
#else
        // ====================================================================
        // Standard / D2 / D5 Phase 1 Opening
        // ====================================================================
#if DG_MEGA_TILE_PROF
        const uint64_t prof_open_s = tile_prof::read_gtimer();
#endif
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
#if DG_ISOLATED_D2_MODE >= 1
        if constexpr (kD2SmidMasked) {
            if (thread_idx == 0) {
                shared_storage.d2_warp_base = ptx::atomic_add(
                    workspace.get_d2_pull_warp_count_ptr(), get_active_dispatch_threads() / 32u);
            }
        }
#endif
        ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx);

        for (uint32_t i = thread_idx; i < kNumExperts; i += get_active_dispatch_threads()) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx);

        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

#if DG_P1_HEAD_HELPER_K > 0
        // The existing dispatch grid rendezvous publishes the reset to all SMs.
        if (sm_idx == 0u and thread_idx == 0u) {
            *p1_cursor = 0u;
            *p1_published_limit = 0u;
        }
#endif
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [&]() { ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx); }
        );
#if DG_MEGA_TILE_PROF
        const uint64_t prof_sync_s = tile_prof::read_gtimer();
        tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                          prof_rec_idx, prof_t0_ns, prof_open_s, prof_sync_s, 1u, 0u);
#endif

#if DG_ISOLATED_D2_MODE >= 1
        if constexpr (kD2SmidMasked) {
            if (thread_idx == 0) {
                shared_storage.d2_total_warps = ptx::ld_acq(workspace.get_d2_pull_warp_count_ptr());
            }
        }
#endif

        if (sm_idx == 0) {
            if constexpr (kD6GroupTiles > 0) {
                if (thread_idx == 0) {
                    const uint32_t prev_ep = *buffer.get_d6_epoch_ptr();
                    *buffer.get_d6_epoch_ptr() = (prev_ep >= 65535u) ? 1u : (prev_ep + 1u);
                }
            }
            for (uint32_t i = thread_idx; i < kNumExperts; i += get_active_dispatch_threads()) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
#if (DG_ISOLATED_D1_G > 0) && (DG_ISOLATED_D5_DEDUP == 2)
                if (sym_buffer.rank_idx == 0) {
                    *sym_buffer.map(workspace.get_d1_ready_bitmap_word_ptr(dst_local_expert_idx, 0), dst_rank_idx) = ~0ull;
                }
#endif
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, 128,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [&]() { ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx); },
            false,
            true
        );

        // -----------------------------------------------------------------
        // Weave CTA-0 runtime selection.  expert_recv_count_sum is live
        // after the post-dispatch NVLink/grid barrier above.  Only thread 0
        // of CTA 0 evaluates Eq.(1)-(3); every other CTA acquire-waits on
        // the published epoch and copies the decision.
        // -----------------------------------------------------------------
        if (thread_idx == 0u) {
            auto* const control = workspace.get_weave_policy_control_ptr();
            const uint32_t weave_epoch = shared_storage.weave_epoch_base + 1u;
            bool policy_ok = false;
            if (sm_idx == 0u) {
                const weave::WorkloadSpec wl{
                    kNumExpertsPerRank,
                    static_cast<uint32_t>(kHidden),
                    static_cast<uint32_t>(kIntermediateHidden)};
                const weave::WeaveKnobs knobs =
                    weave::default_weave_knobs(weave::HardwareArch::B300_SM100, weave::PrecisionMode::FP8xFP4);
                const uint32_t m_out = weave::count_remote_expert_sends(
                    workspace, kNumExperts, kNumExpertsPerRank,
                    static_cast<uint32_t>(sym_buffer.rank_idx));
                shared_storage.weave_policy = weave::cta0_select_from_packed_hist(
                    control, weave::HardwareArch::B300_SM100, wl, knobs,
                    workspace.get_expert_recv_count_sum_ptr(),
                    0u, m_out, weave_epoch);
                policy_ok = weave::policy_is_valid(shared_storage.weave_policy);
            } else {
                (void) weave::load_published_policy(control);
                policy_ok = weave::wait_and_copy_policy(
                    control, weave_epoch, &shared_storage.weave_policy);
            }
            if (policy_ok)
                shared_storage.weave_policy_ready = 1u;
        }
        ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx);

#if DG_P1_HEAD_HELPER_K > 0
        if (sm_idx == 0u and warp_idx == 0u) {
            const uint32_t encoded_limit = p1_count_head_tokens() + 1u;
            if (lane_idx == 0u)
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(p1_published_limit), "r"(encoded_limit) : "memory");
        }
        const uint32_t p1_head_end = p1_wait_for_metadata();
#else
        ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#endif
#if DG_MEGA_TILE_PROF
        tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                          prof_rec_idx, prof_t0_ns, prof_sync_s, tile_prof::read_gtimer(), 2u, 0u);
#endif

#if DG_P1_HEAD_HELPER_K > 0
        run_p1_head_pull.template operator()<kNumDispatchRegisters, true>(p1_head_end);
#endif

        uint32_t pull_mbarrier_phase = 0;
        scheduler.fetch_expert_recv_count();

        // Head/middle temporal split from the CTA-0 decision: all dispatch
        // warps pull the first selected_v_head tasks, then only N_c warps per
        // SM remain in the pull queue.
        const bool weave_policy_ok = shared_storage.weave_policy_ready != 0u;
        const uint32_t weave_comm_warps_per_sm = weave_policy_ok
            ? weave::r_comm_units(shared_storage.weave_policy.selected_r_mask)
            : kNumDispatchWarps;
        const uint32_t active_pull_warps =
            weave_comm_warps_per_sm == 0u ? 1u :
            (weave_comm_warps_per_sm < kNumDispatchWarps
                 ? weave_comm_warps_per_sm : kNumDispatchWarps);
        const uint32_t selected_v_head = weave_policy_ok
            ? shared_storage.weave_policy.selected_v_head : 0u;
        (void) shared_storage.weave_policy.selected_scatter_mode;
        uint32_t weave_total_tokens = 0u;
        for (uint32_t weave_e = 0u; weave_e < kNumExpertsPerRank; ++weave_e)
            weave_total_tokens += scheduler.get_num_tokens(weave_e);
        const uint32_t head_pull_end = selected_v_head == 0u
            ? weave_total_tokens
            : cute::min(selected_v_head, weave_total_tokens);
        uint32_t expert_pool_block_offset = 0;

    #if DG_ISOLATED_D5_DEDUP == 2
        // ====================================================================
        // Axis 5 (D5): Deduplicated token pull + 32-lane fanout
        // ====================================================================
        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        constexpr uint32_t kNumGlobalDispatchThreads = kNumSMs * kNumDispatchThreads;
        constexpr uint32_t kNumERPairs = kNumExpertsPerRank * kNumRanks;
        constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
        DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");
        constexpr uint32_t kNumSFUint32 = kHidden / 128u;
        constexpr uint32_t kNumSFVecs = kNumSFUint32 / 4u;
        DG_STATIC_ASSERT(kNumSFVecs <= 16u and kNumSFVecs * 4u == kNumSFUint32, "Expected <= 16 uint4 SF vectors");
        constexpr uint32_t kPanelBytes = 1024u;
        constexpr uint32_t kPanelsPerChunk = kNumBytesPerPull / kPanelBytes;
        constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;

        const uint32_t warp_gid = (WEAVE_SM0_STREAM_SCHED and sm_idx == 0u)
            ? 0xffffffffu
            : ((WEAVE_SM0_STREAM_SCHED ? (sm_idx - 1u) : sm_idx) * kNumDispatchWarps + warp_idx);
        const uint32_t disp_tid = sm_idx * kNumDispatchThreads + thread_idx;
        const bool use_three_tier_queue =
            (scheduler.num_total_m_blocks > 0u and scheduler.num_total_m_blocks <= kNumRingBlocks);
        const uint32_t num_windows = math::ceil_div(scheduler.num_total_m_blocks, kNumRingBlocks);
        uint32_t warp_fanout_stores = 0u;

        if (sm_idx == 0u and warp_idx == 0u) {
            constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
            uint32_t lane_m_sum = 0u;
            #pragma unroll
            for (uint32_t i = 0u; i < kNumExpertsPerLane; ++i) {
                if (i * 32u + lane_idx < kNumExpertsPerRank)
                    lane_m_sum += scheduler.stored_num_tokens_per_expert[i];
            }
            const uint32_t total_m = __reduce_add_sync(0xffffffff, lane_m_sum);
            if (lane_idx == 0u)
                *d5::total_members_ptr(workspace) = total_m;
        }

        #pragma unroll 1
        for (uint32_t win = 0u; win < num_windows; ++win) {
            const uint32_t tile_lo = win * kNumRingBlocks;
            const uint32_t tile_hi = cute::min(tile_lo + kNumRingBlocks, scheduler.num_total_m_blocks);

            #pragma unroll 1
            for (uint32_t er = warp_gid; er < kNumERPairs; er += kNumGlobalWarps) {
                const uint32_t e = er / kNumRanks;
                const uint32_t r = er % kNumRanks;
                const uint32_t c_lane = (lane_idx < kNumRanks) ?
                    static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(lane_idx, e)) : 0u;
                const uint32_t c_re = __shfl_sync(0xffffffff, c_lane, static_cast<int>(r));
                if (c_re == 0u)
                    continue;

                const uint32_t e_pool_blk = scheduler.get_pool_block_offset(e);
                if (num_windows > 1u) {
                    const uint32_t e_blk_hi = e_pool_blk + math::ceil_div(scheduler.get_num_tokens(e), BLOCK_M);
                    if (e_blk_hi <= tile_lo or e_pool_blk >= tile_hi)
                        continue;
                }
                const uint32_t e_pool_tok_base = e_pool_blk * BLOCK_M;

                #pragma unroll 1
                for (uint32_t s_base = 0u; s_base < c_re; s_base += 32u) {
                    const uint32_t s = s_base + lane_idx;
                    uint32_t row_in_expert = 0u;
                    #pragma unroll
                    for (uint32_t j = 0u; j < kNumRanks; ++j) {
                        const uint32_t cj = __shfl_sync(0xffffffff, c_lane, static_cast<int>(j));
                        const uint32_t cap = s + (j < r ? 1u : 0u);
                        row_in_expert += (cj < cap) ? cj : cap;
                    }
                    if (s < c_re) {
                        const uint32_t pool_tok = e_pool_tok_base + row_in_expert;
                        const uint32_t tile_id = pool_tok / BLOCK_M;
                        if (num_windows == 1u or (tile_id >= tile_lo and tile_id < tile_hi)) {
                            const uint32_t src_token_topk_raw = *workspace.get_src_token_topk_idx_ptr(e, r, s);
                            const uint32_t src_token_topk = src_token_topk_raw & 0x0003ffffu;
                            const uint32_t src_tok = src_token_topk / kNumTopk;
                            const uint32_t src_topk = src_token_topk % kNumTopk;

                            const uint32_t m_slot = ptx::atomic_add(d5::key_count_ptr(workspace, r, src_tok), 1u);
                            *d5::key_member_ptr(workspace, r, src_tok, m_slot) = (src_topk << 20) | pool_tok;
                            if (m_slot == 0u) {
                                ptx::red_add(d5::tile_expected_ptr(workspace, tile_id), 0x00010000u);
                                if (r == sym_buffer.rank_idx) {
                                    const uint32_t k_pos = ptx::atomic_add(d5::num_local_keys_ptr(workspace), 1u);
                                    *d5::local_key_ptr(workspace, k_pos) = (r << 24) | src_tok;
                                } else {
                                    const uint32_t k_pos = ptx::atomic_add(d5::num_remote_keys_ptr(workspace), 1u);
                                    *d5::remote_key_ptr(workspace, k_pos) = (r << 24) | src_tok;
                                    if (use_three_tier_queue)
                                        atomicOr(d5::tile_flags_ptr(workspace, tile_id), 1u);
                                }
                            } else {
                                ptx::red_add(d5::tile_expected_ptr(workspace, tile_id), 0x00000001u);
                                if (use_three_tier_queue)
                                    atomicOr(d5::tile_flags_ptr(workspace, tile_id), 2u);
                            }
                        }
                    }
                    __syncwarp();
                }
            }

            d5::fast_grid_sync<kNumSMs, kNumDispatchThreads, kDispatchBarrierIdx>(
                workspace, sm_idx, thread_idx
            );

            if (use_three_tier_queue and sm_idx == 0u) {
                uint32_t t0 = 0u, t1 = 0u, t2 = 0u;
                #pragma unroll 1
                for (uint32_t tile_id = thread_idx; tile_id < scheduler.num_total_m_blocks; tile_id += kNumDispatchThreads) {
                    const uint32_t flags = *d5::tile_flags_ptr(workspace, tile_id);
                    if (flags == 0u) {
                        t0 += 1u;
                        atomicOr(d5::tier_bitmap_word_ptr(workspace, 0u, tile_id >> 5), 1u << (tile_id & 31u));
                    } else if (flags & 1u) {
                        t1 += 1u;
                    } else {
                        t2 += 1u;
                    }
                }
                t0 = __reduce_add_sync(0xffffffff, t0);
                t1 = __reduce_add_sync(0xffffffff, t1);
                t2 = __reduce_add_sync(0xffffffff, t2);
                if (lane_idx == 0u) {
                    shared_storage.expert_token_count[warp_idx * 4u + 0u] = t0;
                    shared_storage.expert_token_count[warp_idx * 4u + 1u] = t1;
                    shared_storage.expert_token_count[warp_idx * 4u + 2u] = t2;
                }
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
                if (warp_idx == 0u) {
                    const uint32_t w0 = (lane_idx < kNumDispatchWarps) ? shared_storage.expert_token_count[lane_idx * 4u + 0u] : 0u;
                    const uint32_t w1 = (lane_idx < kNumDispatchWarps) ? shared_storage.expert_token_count[lane_idx * 4u + 1u] : 0u;
                    const uint32_t w2 = (lane_idx < kNumDispatchWarps) ? shared_storage.expert_token_count[lane_idx * 4u + 2u] : 0u;
                    const uint32_t sum0 = __reduce_add_sync(0xffffffff, w0);
                    const uint32_t sum1 = __reduce_add_sync(0xffffffff, w1);
                    const uint32_t sum2 = __reduce_add_sync(0xffffffff, w2);
                    if (lane_idx == 0u) {
                        *d5::tier0_count_ptr(workspace) = sum0;
                        *d5::tier1_count_ptr(workspace) = sum1;
                        *d5::tier2_count_ptr(workspace) = sum2;
                        d5::st_release_gpu(d5::plan_ready_ptr(workspace), 1u);
                    }
                }
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            }

            const uint32_t num_loc_keys = *d5::num_local_keys_ptr(workspace);
            const uint32_t num_rem_keys = *d5::num_remote_keys_ptr(workspace);
            uint32_t local_iter = warp_gid;

            #pragma unroll 1
            while (true) {
                uint32_t key_word = 0u;
                if (local_iter < num_loc_keys) {
                    key_word = *d5::local_key_ptr(workspace, local_iter);
                    local_iter += kNumGlobalWarps;
                } else {
                    uint32_t rem_idx = 0u;
                    if (cute::elect_one_sync())
                        rem_idx = ptx::atomic_add(d5::remote_cursor_ptr(workspace), 1u);
                    rem_idx = __shfl_sync(0xffffffff, rem_idx, 0);
                    if (rem_idx >= num_rem_keys)
                        break;
                    key_word = *d5::remote_key_ptr(workspace, rem_idx);
                }

                const uint32_t r = key_word >> 24;
                const uint32_t t = key_word & 0x00ffffffu;
                uint32_t d = 0u;
                if (lane_idx == 0u)
                    d = *d5::key_count_ptr(workspace, r, t);
                d = __shfl_sync(0xffffffff, d, 0);
                if (lane_idx == 0u)
                    warp_fanout_stores += (d - 1u);

                const uint32_t my_mem = (lane_idx < d) ? *d5::key_member_ptr(workspace, r, t, lane_idx) : 0u;

                if (win > 0u and lane_idx < d) {
                    const uint32_t p_blk = (my_mem & 0xfffffu) / BLOCK_M;
                    const uint32_t empty_target = (p_blk / kNumRingBlocks) * kNumL1BlockNs;
                    const auto empty_ptr = workspace.get_l1_empty_count_ptr(p_blk % kNumRingBlocks);
                    while (ptx::ld_acq(empty_ptr) < empty_target) {}
                }
                __syncwarp();

                const auto src_base_ptr = sym_buffer.map(
                    buffer.input_token_buffer.get_data_buffer(t).get_base_ptr(), r);

        #if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
                const uint32_t tma_warp_slot = warp_idx < kNumTmaDispatchWarps ? warp_idx : 0u;
                const auto pull_buffer = smem_send_buffers.get_rank_buffer(tma_warp_slot).get_data_buffer(0);
                const auto pull_mbarrier = &shared_storage.dispatch_barriers[tma_warp_slot];
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        src_base_ptr,
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                }
        #endif

                {
                    const float* remote_w_ptr = sym_buffer.map(
                        buffer.input_topk_weights_buffer.get_base_ptr<float>() + t * kNumTopk, r);
                    const float lane_w = (lane_idx < kNumTopk) ? remote_w_ptr[lane_idx] : 0.0f;
                    const uint32_t my_src_topk = (my_mem >> 20) & 0x1fu;
                    const uint32_t d7_topk_bm = kD7LocalDedupActive
                        ? __reduce_or_sync(0xffffffffu, (lane_idx < d) ? (1u << my_src_topk) : 0u)
                        : 0u;
                    const float my_weight = __shfl_sync(0xffffffff, lane_w, static_cast<int>(my_src_topk));
                    if (lane_idx < d) {
                        const uint32_t p_tok = my_mem & 0xfffffu;
                        *buffer.l1_topk_weights_buffer.get_data_buffer(p_tok % kNumRingTokens).template get_base_ptr<float>() = my_weight;
                        *workspace.get_token_src_metadata_ptr(p_tok) = {r, t, my_src_topk | (d7_topk_bm << 8)};
                    }
                }

                {
                    const auto* remote_sf_vec_ptr = reinterpret_cast<const uint4*>(
                        sym_buffer.map(buffer.input_sf_buffer.get_data_buffer(t).get_base_ptr<uint32_t>(), r));
                    uint4 sf_vec = {0u, 0u, 0u, 0u};
                    if (lane_idx < kNumSFVecs)
                        sf_vec = remote_sf_vec_ptr[lane_idx];
                    sf_vec.x = __shfl_sync(0xffffffff, sf_vec.x, static_cast<int>(lane_idx & 15u));
                    sf_vec.y = __shfl_sync(0xffffffff, sf_vec.y, static_cast<int>(lane_idx & 15u));
                    sf_vec.z = __shfl_sync(0xffffffff, sf_vec.z, static_cast<int>(lane_idx & 15u));
                    sf_vec.w = __shfl_sync(0xffffffff, sf_vec.w, static_cast<int>(lane_idx & 15u));

                    uint32_t* local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
                    const uint32_t hw = lane_idx >> 4;
                    const uint32_t sub_lane = lane_idx & 15u;
                    #pragma unroll 1
                    for (uint32_t m_base = 0u; m_base < d; m_base += 2u) {
                        const uint32_t m_idx = m_base + hw;
                        const uint32_t src_lane = (m_idx < d) ? m_idx : 0u;
                        const uint32_t m_pk = __shfl_sync(0xffffffff, my_mem, static_cast<int>(src_lane));
                        if (m_idx < d and sub_lane < kNumSFVecs) {
                            const uint32_t p_tok = m_pk & 0xfffffu;
                            const uint32_t ring_blk = (p_tok / BLOCK_M) % kNumRingBlocks;
                            const uint32_t sf_ring_tok = ring_blk * SF_BLOCK_M + transform_sf_token_idx(p_tok % BLOCK_M);
                            const uint32_t j = sub_lane * 4u;
                            local_sf_ptr[(j + 0u) * kNumSFRingTokens + sf_ring_tok] = sf_vec.x;
                            local_sf_ptr[(j + 1u) * kNumSFRingTokens + sf_ring_tok] = sf_vec.y;
                            local_sf_ptr[(j + 2u) * kNumSFRingTokens + sf_ring_tok] = sf_vec.z;
                            local_sf_ptr[(j + 3u) * kNumSFRingTokens + sf_ring_tok] = sf_vec.w;
                        }
                    }
                }
                __syncwarp();

                #pragma unroll 1
                for (uint32_t c = 0u; c < kNumChunks; ++c) {
        #if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
                    if (cute::elect_one_sync())
                        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                    __syncwarp();
        #endif

                    #pragma unroll 1
                    for (uint32_t p = 0u; p < kPanelsPerChunk; ++p) {
                        const uint32_t byte_off = c * kNumBytesPerPull + p * kPanelBytes;
        #if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
                        const auto* smem_panel = math::advance_ptr<const uint4>(
                            pull_buffer.get_base_ptr(), p * kPanelBytes);
                        const uint4 v0 = ptx::ld_shared(smem_panel + lane_idx);
                        const uint4 v1 = ptx::ld_shared(smem_panel + 32u + lane_idx);

                        if (p == kPanelsPerChunk - 1u and c + 1u < kNumChunks) {
                            __syncwarp();
                            if (cute::elect_one_sync()) {
                                ptx::tma_load_1d(
                                    pull_buffer.get_base_ptr(),
                                    math::advance_ptr(src_base_ptr, (c + 1u) * kNumBytesPerPull),
                                    pull_mbarrier, kNumBytesPerPull
                                );
                                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                            }
                        }
        #else
                        const auto* glb_panel = math::advance_ptr<const uint4>(src_base_ptr, byte_off);
                        uint4 v0, v1;
                        asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                                     : "l"(glb_panel + lane_idx));
                        asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                                     : "l"(glb_panel + 32u + lane_idx));
        #endif

                        #pragma unroll 1
                        for (uint32_t m = 0u; m < d; ++m) {
                            const uint32_t m_pk = __shfl_sync(0xffffffff, my_mem, static_cast<int>(m));
                            const uint32_t ring_tok = (m_pk & 0xfffffu) % kNumRingTokens;
                            auto* dst_panel = math::advance_ptr<uint4>(
                                buffer.l1_token_buffer.get_data_buffer(ring_tok).get_base_ptr(), byte_off);
                            dst_panel[lane_idx] = v0;
                            dst_panel[32u + lane_idx] = v1;
                        }
                    }
                    __syncwarp();
                }

                if (lane_idx < d) {
                    const uint32_t p_tok = my_mem & 0xfffffu;
                    const uint32_t tile_id = p_tok / BLOCK_M;
                    const uint32_t delta = (lane_idx == 0u) ? 0x00010000u : 0x00000001u;
                    const uint32_t expected = *d5::tile_expected_ptr(workspace, tile_id);
                    const uint32_t prev = d5::atomic_add_acq_rel_gpu(d5::tile_ready_ptr(workspace, tile_id), delta);
                    if (prev + delta == expected) {
                        d5::st_release_gpu(d5::tile_ready_ptr(workspace, tile_id), 0x80000000u);
                        if (use_three_tier_queue) {
                            const uint32_t flags = *d5::tile_flags_ptr(workspace, tile_id);
                            if (flags != 0u) {
                                const uint32_t tier = (flags & 1u) ? 1u : 2u;
                                d5::red_or_rel_gpu_u32(
                                    d5::tier_bitmap_word_ptr(workspace, tier, tile_id >> 5),
                                    1u << (tile_id & 31u));
                            }
                        }
                    }
                }
                __syncwarp();
            }

            if (win + 1u < num_windows) {
                d5::fast_grid_sync<kNumSMs, kNumDispatchThreads, kDispatchBarrierIdx>(
                    workspace, sm_idx, thread_idx
                );
                #pragma unroll 1
                for (uint32_t i = disp_tid; i < num_loc_keys; i += kNumGlobalDispatchThreads) {
                    const uint32_t kw = *d5::local_key_ptr(workspace, i);
                    *d5::key_count_ptr(workspace, kw >> 24, kw & 0x00ffffffu) = 0u;
                }
                #pragma unroll 1
                for (uint32_t i = disp_tid; i < num_rem_keys; i += kNumGlobalDispatchThreads) {
                    const uint32_t kw = *d5::remote_key_ptr(workspace, i);
                    *d5::key_count_ptr(workspace, kw >> 24, kw & 0x00ffffffu) = 0u;
                }
                if (sm_idx == 0u and thread_idx == 0u) {
                    *d5::total_loc_keys_ptr(workspace) += num_loc_keys;
                    *d5::total_rem_keys_ptr(workspace) += num_rem_keys;
                    *d5::num_local_keys_ptr(workspace) = 0u;
                    *d5::num_remote_keys_ptr(workspace) = 0u;
                    *d5::remote_cursor_ptr(workspace) = 0u;
                }
                d5::fast_grid_sync<kNumSMs, kNumDispatchThreads, kDispatchBarrierIdx>(
                    workspace, sm_idx, thread_idx
                );
            }
        }
    #else
        // ====================================================================
        // Standard / D2 / D5==1 Phase 2 Pull Loop
        // ====================================================================
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        #if DG_ISOLATED_D5_DEDUP == 1
        DG_STATIC_ASSERT(kNumRanks <= 32, "D5 fast rank prefix mapping requires kNumRanks <= 32");
        uint32_t rank_start_offset = 0u, rank_end_offset = 0u;
        #else
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        #endif
        uint32_t expert_start_idx = 0, expert_end_idx = 0;

        // SM 0 streaming scheduler: SM 0's communication warps leave the pull
        // queue and the remaining SMs evenly own the full token range.
        constexpr bool kSm0StreamSchedActive = WEAVE_SM0_STREAM_SCHED != 0;
        const uint32_t pull_sm_idx = kSm0StreamSchedActive
            ? (sm_idx == 0u ? kNumSMs : sm_idx - 1u) : sm_idx;
        const uint32_t pull_num_sms = kSm0StreamSchedActive ? (kNumSMs - 1u) : kNumSMs;
        uint32_t initial_pull_warp_idx = (kSm0StreamSchedActive and sm_idx == 0u)
            ? 0xffffffffu : pull_sm_idx * kNumDispatchWarps + warp_idx;
        #if DG_P1_HEAD_HELPER_K > 0
        if (initial_pull_warp_idx != 0xffffffffu)
            initial_pull_warp_idx += p1_head_end;
        #endif
        #if DG_ISOLATED_D2_MODE >= 1
        if constexpr (kD2SmidMasked) {
            initial_pull_warp_idx = ptx::ld_shared(&shared_storage.d2_warp_base) + warp_idx;
        }
        #endif
        const auto get_pull_warp_stride = [&]() -> uint32_t {
        #if DG_ISOLATED_D2_MODE >= 1
            if constexpr (kD2SmidMasked) {
                return ptx::ld_shared(&shared_storage.d2_total_warps);
            }
        #endif
            return pull_num_sms * kNumDispatchWarps;
        };

        for (uint32_t token_idx = initial_pull_warp_idx; ; token_idx += get_pull_warp_stride()) {
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

            // Temporal head/middle split: after the opening head chunk only
            // active_pull_warps warps per SM continue pulling.  A non-zero
            // WEAVE_HEAD_PULL_CHUNK bounds the all-warp prefix.
            constexpr uint32_t kHeadPullChunk = WEAVE_HEAD_PULL_CHUNK;
            const uint32_t effective_head_pull_end = kHeadPullChunk == 0u
                ? head_pull_end : cute::min(head_pull_end, kHeadPullChunk);
            if (token_idx >= effective_head_pull_end and warp_idx >= active_pull_warps)
                continue;

            #if DG_ISOLATED_D5_DEDUP == 1
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                const uint32_t rot_rank = (sym_buffer.rank_idx + lane_idx) % kNumRanks;
                const uint32_t rank_cnt = lane_idx < kNumRanks ?
                    static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(rot_rank, current_expert_idx)) : 0u;
                rank_end_offset = math::warp_inclusive_sum(rank_cnt, lane_idx);
                rank_start_offset = rank_end_offset - rank_cnt;
            }
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            const uint32_t hit_mask = __ballot_sync(
                0xffffffff,
                lane_idx < kNumRanks and token_idx_in_expert >= rank_start_offset and token_idx_in_expert < rank_end_offset);
            const uint32_t owner_lane = static_cast<uint32_t>(__ffs(static_cast<int>(hit_mask)) - 1);
            const uint32_t current_rank_in_expert_idx = (sym_buffer.rank_idx + owner_lane) % kNumRanks;
            const uint32_t token_idx_in_rank = token_idx_in_expert - __shfl_sync(0xffffffff, rank_start_offset, static_cast<int>(owner_lane));
            #else
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t token_idx_in_rank;
            find_round_robin_pull_source(
                remaining, token_idx_in_expert, current_rank_in_expert_idx, token_idx_in_rank);
            if (current_rank_in_expert_idx >= kNumRanks)
                current_rank_in_expert_idx = static_cast<uint32_t>(kNumRanks - 1u);
            #endif

            const uint32_t src_token_topk_raw = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_topk_idx = src_token_topk_raw & 0x0003ffffu;
            const uint32_t d7_topk_bm = kD7LocalDedupActive ? ((src_token_topk_raw >> 18) & 0x3ffu) : 0u;
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = (src_token_topk_idx % kNumTopk) | (d7_topk_bm << 8);
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            execute_token_pull(
                token_idx, current_expert_idx, current_rank_in_expert_idx,
                src_token_topk_idx, src_token_idx, src_topk_idx,
                token_idx_in_expert, expert_end_idx, pool_token_idx, pool_block_idx,
                pull_mbarrier_phase);
        }
    #endif
#endif

#if (DG_D7_HANDOFF_MODE != 0) || (DG_D7_QUEUE_SCATTER != 0)
        // ====================================================================
        // Axis 7 (D7): publish 'opening pull finished' and wake the scatter role.
        // ====================================================================
        const uint32_t d7_act_disp_t = get_active_dispatch_threads();
        ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
        if (thread_idx == 0) {
            ptx::st_shared(&shared_storage.dispatch_pull_done, 1u);
        }
        ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
#endif

#if DG_D7_HANDOFF_MODE != 0
        {
            constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
            constexpr uint32_t kNumBankGroupBytes = 16u;
            const uint32_t rows_per_disp_wave = (d7_act_disp_t / 32u) * 2u;
            constexpr uint32_t kBatchWaves = 1u;
            const uint32_t warp_row_offset = warp_idx * 2u + (lane_idx >> 4);
            const uint32_t half_base = (lane_idx & 16u);
            const uint32_t lane_in_half = (lane_idx & 15u);
            const uint32_t bank_group_idx = (lane_idx & 7u);

            uint32_t disp_slot = 0u;
            uint32_t disp_phase = 0u;
            while (true) {
                shared_storage.scatter_full_barriers[disp_slot].wait(disp_phase);
                const uint4 job_raw = ptx::ld_shared(
                    reinterpret_cast<const uint4*>(&shared_storage.scatter_jobs[disp_slot]));
                const uint32_t packed_meta = job_raw.z;
                if (packed_meta & 0x80000000u) {
                    break;
                }
#if DG_MEGA_TILE_PROF
                const uint64_t prof_d7_s = tile_prof::read_gtimer();
#endif

                const uint32_t pool_m_idx = job_raw.x;
                const uint32_t n_idx = job_raw.y;
                const uint32_t valid_m = packed_meta & 0xffffu;
                const uint32_t s_row_offset = (packed_meta >> 16) & 0x1fffu;
                const bool wg0_full32 = kD7AllowDynamicWg0Full32 and ((packed_meta & 0x20000000u) != 0u);
                const uint32_t runtime_chunk_m = wg0_full32 ? 32u : kD7DualWgChunkM;
                const uint32_t half_col_offset = ((lane_in_half >> 3) * runtime_chunk_m) * kSwizzleCDMode;
                const bool is_shared = (packed_meta & 0x40000000u) != 0u;

                const uint32_t v0 = valid_m > s_row_offset ?
                    cute::min(valid_m - s_row_offset, runtime_chunk_m) : 0u;
                const uint32_t v1 = (not wg0_full32 and kNumEpilogueWarpgroups > 1u and valid_m > WG_BLOCK_M + s_row_offset) ?
                    cute::min(valid_m - (WG_BLOCK_M + s_row_offset), runtime_chunk_m) : 0u;
                const uint32_t total_valid_rows = v0 + v1;

                const uint8_t* slot_base_ptr = d7_get_scatter_base_ptr() + disp_slot * kD7SlotBytes;
                const uint32_t active_batches = total_valid_rows == 0u ? 1u :
                    math::ceil_div(total_valid_rows, kBatchWaves * rows_per_disp_wave);

                #pragma unroll 1
                for (uint32_t b = 0; b < active_batches; ++ b) {
                    const uint32_t w_base = b * kBatchWaves;
                    float4 reg_packed[kBatchWaves];
                    uint32_t reg_meta_word[kBatchWaves];
                    #pragma unroll
                    for (uint32_t wb = 0; wb < kBatchWaves; ++ wb) {
                        const uint32_t r = (w_base + wb) * rows_per_disp_wave + warp_row_offset;
                        reg_meta_word[wb] = 0u;
                        if (r < total_valid_rows) {
                            const bool in_wg0 = (r < v0);
                            const uint32_t row_in_chunk = in_wg0 ? r : (r - v0);
                            const uint32_t m_in_blk = (in_wg0 ? 0u : WG_BLOCK_M) + s_row_offset + row_in_chunk;
                            if (not is_shared and lane_in_half < 3u) {
                                const auto* meta_u32 = reinterpret_cast<const uint32_t*>(
                                    workspace.get_token_src_metadata_ptr(pool_m_idx + m_in_blk));
                                reg_meta_word[wb] = meta_u32[lane_in_half];
                            }
                            const uint8_t* wg_slot_ptr = slot_base_ptr + (in_wg0 ? 0u : (runtime_chunk_m * 256u));
                            const uint8_t* smem_ptr = wg_slot_ptr +
                                half_col_offset +
                                row_in_chunk * kSwizzleCDMode +
                                (bank_group_idx ^ (row_in_chunk & 7u)) * kNumBankGroupBytes;
                            reg_packed[wb] = ptx::ld_shared(reinterpret_cast<const float4*>(smem_ptr));
                        }
                    }

                    if (b + 1u == active_batches) {
                        shared_storage.scatter_empty_barriers[disp_slot].arrive();
                        disp_slot ^= 1u;
                        disp_phase ^= (disp_slot == 0u);
                    }

                    #pragma unroll
                    for (uint32_t wb = 0; wb < kBatchWaves; ++ wb) {
                        const uint32_t meta_rank = __shfl_sync(0xffffffff, reg_meta_word[wb], half_base + 0u);
                        const uint32_t meta_tok  = __shfl_sync(0xffffffff, reg_meta_word[wb], half_base + 1u);
                        const uint32_t meta_topk = __shfl_sync(0xffffffff, reg_meta_word[wb], half_base + 2u) & 0xffu;
                        const uint32_t r = (w_base + wb) * rows_per_disp_wave + warp_row_offset;
                        if (r < total_valid_rows) {
                            uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                            if (is_shared) {
                                const bool in_wg0 = (r < v0);
                                const uint32_t row_in_chunk = in_wg0 ? r : (r - v0);
                                const uint32_t m_in_blk = (in_wg0 ? 0u : WG_BLOCK_M) + s_row_offset + row_in_chunk;
                                dst_rank_idx = sym_buffer.rank_idx;
                                dst_token_idx = pool_m_idx + m_in_blk;
                                dst_topk_idx = kNumTopk;
                            } else {
                                dst_rank_idx = meta_rank;
                                dst_token_idx = meta_tok;
                                dst_topk_idx = meta_topk;
                            }
                            const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto dst_ptr = math::advance_ptr<float4>(
                                dst_token.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                lane_in_half * static_cast<uint32_t>(sizeof(float4)));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = reg_packed[wb];
                        }
                    }
                }
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_d7_s, tile_prof::read_gtimer(),
                                  12u, pool_m_idx / BLOCK_M);
#endif
            }

            ptx::arrive_unaligned(d7_act_disp_t + kNumEpilogueThreads, kScatterDrainBarrierIdx);
        }
#elif DG_D7_QUEUE_SCATTER != 0
        if constexpr (kD7QueueScatterActive) {
            const bool is_scatter_sm =
                (kD7QueueActiveSms >= kNumSMs) or
                (((sm_idx * kD7QueueActiveSms) % kNumSMs) < kD7QueueActiveSms);
            if (is_scatter_sm) {
                const uint32_t epoch = ptx::ld_acq(buffer.get_d7_queue_epoch_ptr());
#if DG_MEGA_TILE_PROF
                const uint64_t prof_d7_s = tile_prof::read_gtimer();
#endif
                const uint32_t half_base = lane_idx & 16u;
                const uint32_t lane_in_half = lane_idx & 15u;
                while (true) {
                    uint32_t claim_old = 0u, claim_n = 0u;
                    if (lane_idx == 0u) {
                        uint32_t h = ptx::ld_acq(buffer.get_d7_queue_head_ptr());
                        while (true) {
                            const uint32_t t = ptx::ld_acq(buffer.get_d7_queue_tail_ptr());
                            if (h >= t) {
                                claim_n = 0u;
                                break;
                            }
                            const uint32_t n = cute::min(2u, t - h);
                            uint32_t prev;
                            asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;"
                                         : "=r"(prev)
                                         : "l"(buffer.get_d7_queue_head_ptr()), "r"(h), "r"(h + n)
                                         : "memory");
                            if (prev == h) {
                                claim_old = h;
                                claim_n = n;
                                break;
                            }
                            h = prev;
                        }
                    }
                    claim_old = __shfl_sync(0xffffffffu, claim_old, 0);
                    claim_n = __shfl_sync(0xffffffffu, claim_n, 0);
                    if (claim_n == 0u) {
                        if (ptx::ld_acq(buffer.get_d7_queue_producer_done_ptr()) >= kNumSMs)
                            break;
                        __nanosleep(64);
                        continue;
                    }
                    const uint32_t half = lane_idx >> 4;
                    if (half < claim_n) {
                        const uint32_t slot = claim_old + half;
                        uint64_t desc = 0ull;
                        if (lane_in_half == 0u) {
                            while (ptx::ld_acq(buffer.get_d7_queue_ready_ptr() + slot) != epoch)
                                __nanosleep(32);
                            desc = ptx::ld_acq_gpu(buffer.get_d7_queue_desc_ptr() + slot);
                        }
                        const uint32_t group_mask = half ? 0xffff0000u : 0x0000ffffu;
                        const uint32_t desc_lo = __shfl_sync(group_mask, static_cast<uint32_t>(desc), half_base);
                        const uint32_t desc_hi = __shfl_sync(group_mask, static_cast<uint32_t>(desc >> 32), half_base);
                        desc = (static_cast<uint64_t>(desc_hi) << 32) | desc_lo;
                        if ((desc & 0x8000000000000000ull) == 0ull) {
                            const uint32_t dst_rank = static_cast<uint32_t>((desc >> 32) & 0xfull);
                            const uint32_t dst_token = static_cast<uint32_t>(desc & 0xffffffffull);
                            const uint32_t dst_topk = static_cast<uint32_t>((desc >> 36) & 0x1full);
                            const uint32_t sub_n = static_cast<uint32_t>((desc >> 41) & 0xffull);
                            const uint32_t n_idx = sub_n * 128u;
                            const auto dst_token_buf = buffer.combine_token_buffer
                                .get_rank_buffer(dst_topk).get_data_buffer(dst_token);
                            const auto dst_ptr = math::advance_ptr<float4>(
                                dst_token_buf.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                lane_in_half * static_cast<uint32_t>(sizeof(float4)));
                            float4 packed;
                            asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                                         : "=r"(*reinterpret_cast<uint32_t*>(&packed.x)),
                                           "=r"(*reinterpret_cast<uint32_t*>(&packed.y)),
                                           "=r"(*reinterpret_cast<uint32_t*>(&packed.z)),
                                           "=r"(*reinterpret_cast<uint32_t*>(&packed.w))
                                         : "l"(buffer.get_d7_local_row_ptr(dst_rank, dst_token, n_idx) + lane_in_half));
                            *sym_buffer.map(dst_ptr, dst_rank) = packed;
                        }
                    }
                }
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_d7_s, tile_prof::read_gtimer(),
                                  29u, sm_idx);
#endif
                ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
                ptx::fence_acq_rel_sys();
                if (thread_idx == 0u)
                    ptx::red_add_rel(buffer.get_d7_queue_consumer_done_ptr(), 1u);
                ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
            } else {
                if (thread_idx == 0u) {
                    while (ptx::ld_acq(buffer.get_d7_queue_producer_done_ptr()) < kNumSMs or
                           ptx::ld_acq(buffer.get_d7_queue_consumer_done_ptr()) < kD7QueueActiveSms)
                        __nanosleep(512);
                }
                ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
            }
            ptx::arrive_unaligned(d7_act_disp_t + kNumEpilogueThreads, kScatterDrainBarrierIdx);
        }
#endif

        // ====================================================================
        // Workspace Cleanup (overlapped with Combine)
        // ====================================================================
        const uint32_t cleanup_dispatch_threads = get_active_dispatch_threads();
        ptx::sync_unaligned(cleanup_dispatch_threads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#if DG_MEGA_TILE_PROF
        const uint64_t prof_clean_s = tile_prof::read_gtimer();
#endif
        if constexpr (kD6GroupTiles > 0) {
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [&]() { ptx::sync_aligned(cleanup_dispatch_threads, kDispatchBarrierIdx); }
            );
        }

#if (DG_ISOLATED_D1_G > 0) && (DG_ISOLATED_D5_DEDUP != 2)
        if (sm_idx == 0) {
            if (warp_idx == 0 and cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t d = 0; d < kNumTaskCountDomains; ++ d) {
                    *workspace.get_l1_task_count_ptr(d) = 0;
                    *workspace.get_l2_task_count_ptr(d) = 0;
                    *workspace.get_shared_l1_task_count_ptr(d) = 0;
                    *workspace.get_shared_l2_task_count_ptr(d) = 0;
                }
                #if DG_ISOLATED_D4_READYQ
                *workspace.get_d4_ready_write_count_ptr() = 0;
                #endif
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += cleanup_dispatch_threads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const uint32_t num_recv_tokens = scheduler.get_num_tokens(i);
                const uint32_t num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const uint32_t num_groups = math::ceil_div(num_recv_tokens, kD1GroupTokens);
                const uint32_t num_bm_words = math::ceil_div(num_groups, 64u);
                const uint32_t expert_pool_block_offset_clean = ptx::exchange(expert_blk_offset[i / 32], i % 32);
                ptx::sync_aligned(cleanup_dispatch_threads, kDispatchBarrierIdx);

                if (warp_idx == 0) {
                    if (cute::elect_one_sync()) {
                        *workspace.get_d1_expert_write_cursor_ptr(i) = 0;
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    }
                    __syncwarp();
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t g = thread_idx; g < num_groups; g += cleanup_dispatch_threads)
                    *workspace.get_d1_group_meta_count_ptr(i, g) = 0;
                for (uint32_t w = thread_idx; w < num_bm_words; w += cleanup_dispatch_threads)
                    *workspace.get_d1_ready_bitmap_word_ptr(i, w) = 0;
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += cleanup_dispatch_threads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset_clean + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset_clean + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_count_ptr((expert_pool_block_offset_clean + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset_clean + j) % kNumRingBlocks) = 0;
                    #if DG_ISOLATED_D4_READYQ
                    if (expert_pool_block_offset_clean + j < kNumRingBlocks)
                        *workspace.get_d4_ready_queue_ptr(expert_pool_block_offset_clean + j) = 0;
                    #endif
                }
                __syncwarp();
            }
        }
#else
    #if DG_ISOLATED_D5_DEDUP == 2
        if (lane_idx == 0u and warp_fanout_stores > 0u)
            ptx::red_add(d5::fanout_sum_ptr(workspace), warp_fanout_stores);
        const uint32_t last_loc_keys = *d5::num_local_keys_ptr(workspace);
        const uint32_t last_rem_keys = *d5::num_remote_keys_ptr(workspace);
        #pragma unroll 1
        for (uint32_t i = disp_tid; i < last_loc_keys; i += kNumGlobalDispatchThreads) {
            const uint32_t kw = *d5::local_key_ptr(workspace, i);
            *d5::key_count_ptr(workspace, kw >> 24, kw & 0x00ffffffu) = 0u;
        }
        #pragma unroll 1
        for (uint32_t i = disp_tid; i < last_rem_keys; i += kNumGlobalDispatchThreads) {
            const uint32_t kw = *d5::remote_key_ptr(workspace, i);
            *d5::key_count_ptr(workspace, kw >> 24, kw & 0x00ffffffu) = 0u;
        }
        #pragma unroll 1
        for (uint32_t b = disp_tid; b < scheduler.num_total_m_blocks; b += kNumGlobalDispatchThreads) {
            *d5::tile_expected_ptr(workspace, b) = 0u;
            *d5::tile_ready_ptr(workspace, b) = 0u;
            *d5::tile_flags_ptr(workspace, b) = 0u;
            *d5::sched_m_block_ptr(workspace, b) = 0u;
        }
        const uint32_t num_bmap_words = math::ceil_div(scheduler.num_total_m_blocks, 32u);
        #pragma unroll 1
        for (uint32_t w = disp_tid; w < num_bmap_words; w += kNumGlobalDispatchThreads) {
            *d5::tier_bitmap_word_ptr(workspace, 0u, w) = 0u;
            *d5::tier_bitmap_word_ptr(workspace, 1u, w) = 0u;
            *d5::tier_bitmap_word_ptr(workspace, 2u, w) = 0u;
        }
    #endif

        if (sm_idx == 0) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += cleanup_dispatch_threads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t d = 0; d < kNumTaskCountDomains; ++ d) {
                    *workspace.get_l1_task_count_ptr(d) = 0;
                    *workspace.get_l2_task_count_ptr(d) = 0;
                    *workspace.get_shared_l1_task_count_ptr(d) = 0;
                    *workspace.get_shared_l2_task_count_ptr(d) = 0;
                }
                if constexpr (kD2SmidMasked) {
                    *workspace.get_d2_pull_warp_count_ptr() = 0;
                }
                #if DG_ISOLATED_D4_READYQ
                *workspace.get_d4_ready_write_count_ptr() = 0;
                #endif
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += cleanup_dispatch_threads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);
                ptx::sync_aligned(cleanup_dispatch_threads, kDispatchBarrierIdx);

                    if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    #if (DG_ISOLATED_D1_G > 0) && (DG_ISOLATED_D5_DEDUP == 2)
                    *workspace.get_d1_ready_bitmap_word_ptr(i, 0) = 0;
                    #endif
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += cleanup_dispatch_threads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += cleanup_dispatch_threads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    #if DG_ISOLATED_D4_READYQ
                    if (expert_pool_block_offset + j < kNumRingBlocks)
                        *workspace.get_d4_ready_queue_ptr(expert_pool_block_offset + j) = 0;
                    #endif
                }
                __syncwarp();
            }
        }
#endif

        comm::nvlink_barrier<kNumRanks, kNumSMs, 128,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(cleanup_dispatch_threads, kDispatchBarrierIdx); },
            true,
            false
        );
#if DG_MEGA_TILE_PROF
        tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                          prof_rec_idx, prof_t0_ns, prof_clean_s, tile_prof::read_gtimer(),
                          13u, 0u);
#endif

#if DG_ISOLATED_D5_DEDUP == 2
        if (blockIdx.x == 0u and threadIdx.x == 0u) {
            const layout::Workspace ws_end(
                sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
                kNumMaxTokensPerRank, kNumTopk, kNumRingTokens);
            const uint32_t total_memberships = *d5::total_members_ptr(ws_end);
            const uint32_t total_loc = *d5::total_loc_keys_ptr(ws_end) + *d5::num_local_keys_ptr(ws_end);
            const uint32_t total_rem = *d5::total_rem_keys_ptr(ws_end) + *d5::num_remote_keys_ptr(ws_end);
            const uint32_t unique_keys = total_loc + total_rem;
            const uint32_t fanout_stores = *d5::fanout_sum_ptr(ws_end);
            const uint32_t b0 = *d5::tier0_count_ptr(ws_end);
            const uint32_t b1 = *d5::tier1_count_ptr(ws_end);
            const uint32_t b2 = *d5::tier2_count_ptr(ws_end);
            DG_TRAP_ONLY_DEVICE_ASSERT(fanout_stores == total_memberships - unique_keys);
            uint32_t* telem = d5::telemetry_ptr(ws_end);
            telem[0] = total_memberships;
            telem[1] = unique_keys;
            telem[2] = total_rem;
            telem[3] = total_rem;
            telem[4] = fanout_stores;
            telem[5] = (b0 & 0x3ffu) | ((b1 & 0x3ffu) << 10) | ((b2 & 0x3ffu) << 20);

            *d5::num_local_keys_ptr(ws_end) = 0u;
            *d5::num_remote_keys_ptr(ws_end) = 0u;
            *d5::remote_cursor_ptr(ws_end) = 0u;
            *d5::plan_ready_ptr(ws_end) = 0u;
            *d5::tier0_count_ptr(ws_end) = 0u;
            *d5::tier1_count_ptr(ws_end) = 0u;
            *d5::tier2_count_ptr(ws_end) = 0u;
            *d5::total_loc_keys_ptr(ws_end) = 0u;
            *d5::total_rem_keys_ptr(ws_end) = 0u;
            *d5::total_members_ptr(ws_end) = 0u;
            *d5::fanout_sum_ptr(ws_end) = 0u;
        }
#endif
    } else if (warp_idx == kNumDispatchWarps) {
#if DG_P1_HEAD_HELPER_K == 0
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
#endif

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const uint32_t aligned_valid_m = task_info.get_umma_aligned_valid_m();
            uint32_t runtime_load_m = LOAD_BLOCK_M;
            const cute::TmaDescriptor* tensor_map_a_ptr =
                task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                                                            &tensor_map_shared_l2_acts;
#if DG_D3_HAS_EXTRA_TMAS
            if constexpr (DG_D3_DYN_TMA_A >= 2 and LOAD_BLOCK_M > 8u) {
                if (not task_info.is_shared() and aligned_valid_m <= 16u) {
                    runtime_load_m = 8u;
                    tensor_map_a_ptr = (task_info.block_phase == sched::BlockPhase::Linear1)
                        ? &tensor_map_l1_acts_m8 : &tensor_map_l2_acts_m8;
                } else if (not task_info.is_shared() and LOAD_BLOCK_M > 16u and aligned_valid_m <= 32u) {
                    runtime_load_m = 16u;
                    tensor_map_a_ptr = (task_info.block_phase == sched::BlockPhase::Linear1)
                        ? &tensor_map_l1_acts_m16 : &tensor_map_l2_acts_m16;
                }
            } else if constexpr (DG_D3_DYN_TMA_A == 1 and LOAD_BLOCK_M > 16u) {
                if (not task_info.is_shared() and aligned_valid_m <= 32u) {
                    runtime_load_m = 16u;
                    tensor_map_a_ptr = (task_info.block_phase == sched::BlockPhase::Linear1)
                        ? &tensor_map_l1_acts_m16 : &tensor_map_l2_acts_m16;
                }
            }
#endif
            const auto tensor_map_sfa_ptr =
                task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                                                            &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            if (task_info.block_phase == sched::BlockPhase::Linear1) {
#if DG_ISOLATED_D5_DEDUP == 2
                while ((ptx::ld_acq(d5::tile_ready_ptr(workspace, pool_block_idx)) & 0x80000000u) == 0u) {}
#else
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
#endif
            } else if (task_info.block_phase == sched::BlockPhase::Linear2) {
                const auto ptr = workspace.get_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (L2_SHAPE_K / (BLOCK_N / 2)) * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = SHARED_L2_SHAPE_K / (BLOCK_N / 2);
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }
#if DG_MEGA_TILE_PROF
            const uint32_t prof_tma_a_s = tile_prof::read_gtimer();
#endif

            const uint32_t expected_a_tx = runtime_load_m * BLOCK_K * sizeof(a_dtype_t) * 2u +
                                           static_cast<uint32_t>(sizeof(SharedStorage::smem_sfa[0])) * 2u;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);

                if (not is_leader_cta)
                    m_idx += aligned_valid_m / 2;

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, BLOCK_K / 128, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(expected_a_tx);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
#if DG_MEGA_TILE_PROF
            const uint32_t stg_a = (task_info.block_phase == sched::BlockPhase::Linear1 or
                                    task_info.block_phase == sched::BlockPhase::SharedLinear1) ? 5u : 8u;
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_tma_a_s, tile_prof::read_gtimer(),
                              stg_a, task_info.pool_block_idx);
#endif
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
#if DG_P1_HEAD_HELPER_K == 0
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
#endif

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
#if DG_MEGA_TILE_PROF
            uint32_t prof_tma_b_s = tile_prof::read_gtimer();
#endif
            const auto tensor_map_b_ptr =
                task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                                                            &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr =
                task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                                                            &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
#if DG_MEGA_TILE_PROF
                if (k_block_idx == kNumStages)
                    prof_tma_b_s = tile_prof::read_gtimer();
#endif

                uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ? k_block_idx * (BLOCK_K / 128) : task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                if (cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, BLOCK_K / 128, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * 2 + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    } else {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, BLOCK_K / 128, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                }
                __syncwarp();
            }
#if DG_MEGA_TILE_PROF
            const uint32_t stg_b = (task_info.block_phase == sched::BlockPhase::Linear1 or
                                    task_info.block_phase == sched::BlockPhase::SharedLinear1) ? 6u : 9u;
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_tma_b_s, tile_prof::read_gtimer(),
                              stg_b, task_info.pool_block_idx);
#endif
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
#if DG_P1_HEAD_HELPER_K == 0
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
#endif

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    shared_b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
            auto shared_b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[0]), 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(SharedStorage::smem_a[0]) / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;
            uint32_t shared_b_desc_lo = lane_idx < kNumStages ? shared_b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;

            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
#if DG_MEGA_TILE_PROF
                uint32_t prof_mma_s = 0u;
#endif
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;

                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();
#if DG_MEGA_TILE_PROF
                    if (k_block_idx == 0)
                        prof_mma_s = tile_prof::read_gtimer();
#endif

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(task_info.is_shared() ? shared_b_desc_lo : b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < kNumKSubtilesPerBlock; ++ umma_k_block_idx) {
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + umma_k_block_idx * kNumSFATmemColsPerSubtile + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + umma_k_block_idx * kNumSFBTmemColsPerSubtile + i * 4);
                            }

                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);

                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                        a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
                                #pragma unroll
                                for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                                    const uint32_t b_sub_offset =
                                        umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N + n_sub * UMMA_BLOCK_K * LAYOUT_AD_M;
                                    if (task_info.is_shared()) {
                                        b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                            cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                                                b_desc_base_lo, b_sub_offset * sizeof(shared_b_dtype_t), k * UMMA_K);
                                    } else {
                                        b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                            cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                                                b_desc_base_lo, b_sub_offset * sizeof(b_dtype_t), k * UMMA_K);
                                    }
                                    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                        b_desc, a_desc,
                                        accum_stage_idx * kNumAccumColsPerStage + n_sub * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0,
                                        runtime_instr_desc,
                                        kTmemStartColOfSFB + umma_k_block_idx * kNumSFBTmemColsPerSubtile + n_sub * 4,
                                        kTmemStartColOfSFA + umma_k_block_idx * kNumSFATmemColsPerSubtile);
                                }
                            }
                        }
                    }
                    __syncwarp();

                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
#if DG_MEGA_TILE_PROF
                const uint32_t stg_m = (task_info.block_phase == sched::BlockPhase::Linear1 or
                                        task_info.block_phase == sched::BlockPhase::SharedLinear1) ? 7u : 10u;
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_mma_s, tile_prof::read_gtimer(),
                                  stg_m, task_info.pool_block_idx);
#endif
            }

            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
#if DG_P1_HEAD_HELPER_K == 0
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
#endif

        if (is_leader_cta) {
#if DG_MEGA_TILE_PROF
            const uint32_t prof_sched_s = tile_prof::read_gtimer();
#endif
#if DG_ISOLATED_D4_LUT_ACTIVE
            scheduler.mainloop(num_tokens, sym_buffer.rank_idx, nullptr, nullptr, sm_locality_domains);
#else
            scheduler.mainloop(num_tokens, 0, nullptr, nullptr, sm_locality_domains);
#endif
#if DG_MEGA_TILE_PROF
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_sched_s, tile_prof::read_gtimer(),
                              4u, 0u);
#endif
        }
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
#if DG_MEGA_TILE_PROF
        prof_rec_idx = 0u;
#endif

        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        // BM=240's 120-row warpgroup slice is the first tile where a one-store-block
        // register prefetch of the TMEM accumulator pays off: it hides the TMEM read
        // latency behind SwiGLU / cast work while keeping the same 2-stage TMEM pool.
        // TMEM epilogue pipeline depth: 1.0 / 1.75 / 2.0 stages.  The auto
        // value is 2.0 below the 256-row tile and 1.75 at 256 rows, exactly the
        // verified producer configuration.
        constexpr uint32_t kTmemPipelineMode = WEAVE_TMEM_PIPELINE_MODE == 0
            ? (BLOCK_M >= 256u ? 175u : 200u)
            : static_cast<uint32_t>(WEAVE_TMEM_PIPELINE_MODE);
        DG_STATIC_ASSERT(kTmemPipelineMode == 100u or kTmemPipelineMode == 175u or
                         kTmemPipelineMode == 200u,
                         "WEAVE_TMEM_PIPELINE_MODE must be auto, 1.0, 1.75 or 2.0");
        DG_STATIC_ASSERT(kTmemPipelineMode != 175u or BLOCK_M == 256u,
                         "The 1.75-stage TMEM pipeline requires BLOCK_M=256");
        constexpr uint32_t kNumEpiPrefetchStages =
            (kTmemPipelineMode >= 175u and BLOCK_M >= 240u and
             kNumNSubtiles == 1u and DG_D3_DYN_EPI == 0) ? 1u : 0u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        constexpr uint32_t kMaxTotalAtoms = BLOCK_M / ATOM_M;
        constexpr uint32_t kMaxWgAtoms = (kMaxTotalAtoms + kNumEpilogueWarpgroups - 1) / kNumEpilogueWarpgroups;
        constexpr uint32_t kMaxPassesPerWg = (DG_D3_DYN_EPI == 2)
            ? ((kMaxWgAtoms + 1) / 2)
            : ((DG_D3_DYN_EPI == 1)
                ? ((math::constexpr_ceil_div(BLOCK_M, STORE_BLOCK_M) + kNumEpilogueWarpgroups - 1) / kNumEpilogueWarpgroups)
                : (WG_BLOCK_M / STORE_BLOCK_M));

        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M_L2 % ATOM_M == 0, "Invalid L2 store block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup/store L1 block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M_L2 == 0, "Invalid warpgroup/store L2 block M");

#if DG_P1_HEAD_HELPER_K > 0
        run_p1_head_pull.template operator()<kNumEpilogueRegisters, false>(p1_wait_for_metadata());
#else
        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#endif
        [[maybe_unused]] const uint16_t d6_cur_epoch = kD6GroupTiles > 0 ?
            static_cast<uint16_t>(*buffer.get_d6_epoch_ptr()) : static_cast<uint16_t>(0u);

        uint32_t current_iter_idx = 0;
        [[maybe_unused]] uint32_t tma_store_seq = 0;
#if DG_D7_HANDOFF_MODE != 0
        uint32_t epi_scatter_slot = 0u;
        uint32_t epi_scatter_phase = 0u;
        uint32_t epi_scatter_step = 0u;
        bool scatter_handoff_enabled = false;
#endif
        uint32_t d7_queue_epoch = 0u;
        uint32_t d7_queue_slot_next = 0u;
        uint32_t d7_queue_slot_end = 0u;
#if DG_D7_QUEUE_SCATTER != 0
        if constexpr (kD7QueueScatterActive)
            d7_queue_epoch = ptx::ld_acq(buffer.get_d7_queue_epoch_ptr());
#endif

        // Publish one completed local row.  Only the leader lane of a 16-lane
        // half-warp reserves a slot; the reservation is amortized over a batch
        // so 148 CTAs do not serialize on a single global atomic per row.
        const auto d7_queue_publish = [&](const uint32_t& dst_rank_idx,
                                          const uint32_t& dst_token_idx,
                                          const uint32_t& dst_topk_idx,
                                          const uint32_t& sub_n_idx) {
            if constexpr (!kD7QueueScatterActive) {
                return;
            } else {
                if ((lane_idx & 15u) != 0u)
                    return;
                if (d7_queue_slot_next >= d7_queue_slot_end) {
                    const uint32_t base = ptx::atomic_add(
                        buffer.get_d7_queue_tail_ptr(), kD7QueueReserveBatch);
                    d7_queue_slot_next = base;
                    d7_queue_slot_end = base + kD7QueueReserveBatch;
                }
                const uint32_t slot = d7_queue_slot_next ++;
                const uint64_t desc =
                    (static_cast<uint64_t>(dst_token_idx) & 0xffffffffull) |
                    ((static_cast<uint64_t>(dst_rank_idx) & 0xfull) << 32) |
                    ((static_cast<uint64_t>(dst_topk_idx) & 0x1full) << 36) |
                    ((static_cast<uint64_t>(sub_n_idx) & 0xffull) << 41);
                auto* const desc_ptr = buffer.get_d7_queue_desc_ptr() + slot;
                auto* const ready_ptr = buffer.get_d7_queue_ready_ptr() + slot;
                asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(desc_ptr), "l"(desc) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(ready_ptr), "r"(d7_queue_epoch) : "memory");
            }
        };

        // Reserved-but-unused slots are filled with dummy descriptors so the
        // monotone tail never leaves a permanent hole in the consumer stream.
        const auto d7_queue_flush = [&]() {
            if constexpr (!kD7QueueScatterActive) {
                return;
            } else {
                if ((lane_idx & 15u) != 0u)
                    return;
                constexpr uint64_t kDummyDescriptor = 0x8000000000000000ull;
                for (uint32_t slot = d7_queue_slot_next; slot < d7_queue_slot_end; ++ slot) {
                    auto* const desc_ptr = buffer.get_d7_queue_desc_ptr() + slot;
                    auto* const ready_ptr = buffer.get_d7_queue_ready_ptr() + slot;
                    asm volatile("st.global.cg.u64 [%0], %1;"
                                 :: "l"(desc_ptr), "l"(kDummyDescriptor) : "memory");
                    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                 :: "l"(ready_ptr), "r"(d7_queue_epoch) : "memory");
                }
                d7_queue_slot_next = 0u;
                d7_queue_slot_end = 0u;
            }
        };

        const auto d7_scatter_or_prereduce_row = [&](const bool& is_shared,
                                                     const uint32_t& dst_rank_idx,
                                                     const uint32_t& dst_token_idx,
                                                     const uint32_t& raw_topk_meta,
                                                     const uint32_t& n_idx,
                                                     const float4& packed) {
            const uint32_t dst_topk_idx = raw_topk_meta & 0xffu;
            const uint32_t lane_in_half = lane_idx & 15u;
            if constexpr (kD7QueueScatterActive) {
                const uint32_t sub_n_idx = n_idx / 128u;
                const uint32_t topk_bm = is_shared ? 0u : ((raw_topk_meta >> 8) & 0x3ffu);
                const uint32_t hit_count = is_shared ? 1u : static_cast<uint32_t>(__popc(topk_bm));
                const uint32_t leader_topk_idx = is_shared ? kNumTopk : static_cast<uint32_t>(__ffs(topk_bm) - 1);
                const uint32_t half_mask = __activemask();
                const uint32_t half_base = lane_idx & 16u;
                float4* const local_ptr =
                    buffer.get_d7_local_row_ptr(dst_rank_idx, dst_token_idx, n_idx) + lane_in_half;
                float4 out_val = packed;
                uint32_t next_cnt = 1u;
                if (hit_count > 1u) {
                    uint32_t* const state_ptr =
                        buffer.get_d7_subtile_state_ptr(dst_rank_idx, dst_token_idx, sub_n_idx);
                    uint32_t old_state = 0u;
                    if (lane_in_half == 0u) {
                        while (true) {
                            asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;"
                                         : "=r"(old_state)
                                         : "l"(state_ptr),
                                           "r"(old_state & 0x7fffffffu),
                                           "r"((old_state & 0x7fffffffu) | 0x80000000u)
                                         : "memory");
                            if ((old_state & 0x80000000u) == 0u)
                                break;
                        }
                    }
                    const uint32_t done_cnt = __shfl_sync(half_mask, old_state, half_base);
                    if (done_cnt > 0u) {
                        float4 prev_val;
                        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&prev_val.x)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&prev_val.y)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&prev_val.z)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&prev_val.w))
                                     : "l"(local_ptr));
                        const auto* p_bf16 = reinterpret_cast<const nv_bfloat162*>(&prev_val);
                        const auto* c_bf16 = reinterpret_cast<const nv_bfloat162*>(&packed);
                        auto* o_bf16 = reinterpret_cast<nv_bfloat162*>(&out_val);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l) {
                            o_bf16[l] = __float22bfloat162_rn(__fadd2_rn(
                                __bfloat1622float2(p_bf16[l]),
                                __bfloat1622float2(c_bf16[l])));
                        }
                    }
                    next_cnt = done_cnt + 1u;
                }
                // Every completion writes the final local row, including the
                // last contributor, before the queue descriptor is published.
                asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                             :: "l"(local_ptr),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_val.x)),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_val.y)),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_val.z)),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_val.w))
                             : "memory");
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                __syncwarp(half_mask);
                if (hit_count > 1u and lane_in_half == 0u) {
                    uint32_t* const state_ptr =
                        buffer.get_d7_subtile_state_ptr(dst_rank_idx, dst_token_idx, sub_n_idx);
                    const uint32_t new_state = (next_cnt < hit_count) ? next_cnt : 0u;
                    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                 :: "l"(state_ptr), "r"(new_state) : "memory");
                }
                if (next_cnt == hit_count) {
                    d7_queue_publish(dst_rank_idx, dst_token_idx, leader_topk_idx, sub_n_idx);
                }
            } else if constexpr (!kD7LocalDedupActive) {
                const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                       .get_data_buffer(dst_token_idx);
                const auto dst_ptr = math::advance_ptr<float4>(
                    dst_token.get_base_ptr(),
                    n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                    lane_in_half * static_cast<uint32_t>(sizeof(float4)));
                *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
            } else {
                const uint32_t topk_bm = (raw_topk_meta >> 8) & 0x3ffu;
                const uint32_t hit_count = __popc(topk_bm);
                if (is_shared or dst_rank_idx == sym_buffer.rank_idx or hit_count <= 1u) {
                    const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                           .get_data_buffer(dst_token_idx);
                    const auto dst_ptr = math::advance_ptr<float4>(
                        dst_token.get_base_ptr(),
                        n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                        lane_in_half * static_cast<uint32_t>(sizeof(float4)));
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                } else {
                    const uint32_t half_mask = __activemask();
                    const uint32_t half_base = lane_idx & 16u;
                    const uint32_t leader_topk_idx = static_cast<uint32_t>(__ffs(topk_bm) - 1);
                    const uint32_t sub_n_idx = n_idx / 128u;
                    uint32_t* const state_ptr = buffer.get_d7_subtile_state_ptr(dst_rank_idx, dst_token_idx, sub_n_idx);
                    float4* const local_ptr = buffer.get_d7_local_row_ptr(dst_rank_idx, dst_token_idx, n_idx) + lane_in_half;

                    uint32_t old_state = 0u;
                    if (lane_in_half == 0u) {
                        while (true) {
                            asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;"
                                         : "=r"(old_state)
                                         : "l"(state_ptr),
                                           "r"(old_state & 0x7fffffffu),
                                           "r"((old_state & 0x7fffffffu) | 0x80000000u)
                                         : "memory");
                            if ((old_state & 0x80000000u) == 0u)
                                break;
                        }
                    }
                    const uint32_t done_cnt = __shfl_sync(half_mask, old_state, half_base);
                    float4 out_val = packed;
                    if (done_cnt > 0u) {
                        float4 prev_val;
                        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&prev_val.x)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&prev_val.y)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&prev_val.z)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&prev_val.w))
                                     : "l"(local_ptr));
                        const auto* p_bf16 = reinterpret_cast<const nv_bfloat162*>(&prev_val);
                        const auto* c_bf16 = reinterpret_cast<const nv_bfloat162*>(&packed);
                        auto* o_bf16 = reinterpret_cast<nv_bfloat162*>(&out_val);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l) {
                            o_bf16[l] = __float22bfloat162_rn(__fadd2_rn(
                                __bfloat1622float2(p_bf16[l]),
                                __bfloat1622float2(c_bf16[l])));
                        }
                    }
                    const uint32_t next_cnt = done_cnt + 1u;
                    if (next_cnt < hit_count) {
                        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                     :: "l"(local_ptr),
                                        "r"(*reinterpret_cast<const uint32_t*>(&out_val.x)),
                                        "r"(*reinterpret_cast<const uint32_t*>(&out_val.y)),
                                        "r"(*reinterpret_cast<const uint32_t*>(&out_val.z)),
                                        "r"(*reinterpret_cast<const uint32_t*>(&out_val.w))
                                     : "memory");
                    }
                    __syncwarp(half_mask);
                    if (lane_in_half == 0u) {
                        const uint32_t new_state = (next_cnt < hit_count) ? next_cnt : 0u;
                        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(state_ptr), "r"(new_state) : "memory");
                    }
                    if (next_cnt == hit_count) {
                        const auto leader_tok = buffer.combine_token_buffer.get_rank_buffer(leader_topk_idx)
                                                .get_data_buffer(dst_token_idx);
                        const auto leader_ptr = math::advance_ptr<float4>(
                            leader_tok.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            lane_in_half * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(leader_ptr, dst_rank_idx) = out_val;
                    }
                }
            }
        };

        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();
#if DG_MEGA_TILE_PROF
            const uint64_t prof_epi_s = tile_prof::read_gtimer();
#endif

            scheduler.release_task_info();

            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t wg_start_m = epilogue_wg_idx * WG_BLOCK_M;
            const uint32_t num_valid_wg_rows = valid_m <= wg_start_m ? 0u : cute::min(valid_m - wg_start_m, WG_BLOCK_M);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;

            [[maybe_unused]] uint32_t wg_atom_begin = 0;
            [[maybe_unused]] uint32_t wg_atoms = 0;
            [[maybe_unused]] uint32_t num_passes = 0;
            if constexpr (DG_D3_DYN_EPI == 2) {
                DG_STATIC_ASSERT(STORE_BLOCK_M == 16u and kNumAtomsPerStore == 2u, "DG_D3_DYN_EPI=2 expects STORE_BLOCK_M=16");
                const bool allow_m8_tail = not task_info.is_shared();
                const uint32_t total_atoms = allow_m8_tail
                    ? math::ceil_div(valid_m, ATOM_M)
                    : (math::ceil_div(valid_m, STORE_BLOCK_M) * kNumAtomsPerStore);
                wg_atom_begin = (total_atoms * epilogue_wg_idx + kNumEpilogueWarpgroups - 1) / kNumEpilogueWarpgroups;
                const uint32_t wg_atom_end = (total_atoms * (epilogue_wg_idx + 1) + kNumEpilogueWarpgroups - 1) / kNumEpilogueWarpgroups;
                wg_atoms = wg_atom_end - wg_atom_begin;
                num_passes = (wg_atoms + 1u) / 2u;
            } else if constexpr (DG_D3_DYN_EPI == 1) {
                const uint32_t total_blocks = math::ceil_div(valid_m, STORE_BLOCK_M);
                const uint32_t blk_begin = (total_blocks * epilogue_wg_idx + kNumEpilogueWarpgroups - 1) / kNumEpilogueWarpgroups;
                const uint32_t blk_end = (total_blocks * (epilogue_wg_idx + 1) + kNumEpilogueWarpgroups - 1) / kNumEpilogueWarpgroups;
                num_passes = blk_end - blk_begin;
                wg_atom_begin = blk_begin * kNumAtomsPerStore;
                wg_atoms = num_passes * kNumAtomsPerStore;
            }

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

#if DG_D3_DYN_EPI > 0
                if (wg_atoms == 0u) {
                    ptx::tcgen05_before_thread_sync();
                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                } else {
                    #pragma unroll
                    for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                        const uint32_t sub_accum_base = accum_stage_idx * kNumAccumColsPerStage + n_sub * UMMA_N;
                        const uint32_t sub_n_block_idx = n_block_idx * kNumNSubtiles + n_sub;
                        float stored_cached_weight = 1.0f;

                        #pragma unroll
                        for (uint32_t s = 0; s < kMaxPassesPerWg; ++ s) {
                            if (s >= num_passes)
                                break;
                            const uint32_t atoms_in_pass = (DG_D3_DYN_EPI == 2)
                                ? cute::min(2u, wg_atoms - s * 2u)
                                : kNumAtomsPerStore;
                            const uint32_t store_m_idx = (wg_atom_begin + s * kNumAtomsPerStore) * ATOM_M;

                            float2 activation_values[kNumAtomsPerStore][2];
                            float2 amax_values[kNumAtomsPerStore];
                            #pragma unroll
                            for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                                if (DG_D3_DYN_EPI == 2 and i >= atoms_in_pass)
                                    break;
                                const uint32_t atom_m_idx = store_m_idx + i * ATOM_M;
                                const uint32_t weight_group_m_idx = atom_m_idx & ~31u;
                                if (not task_info.is_shared() and ((atom_m_idx % 32 == 0) or (s == 0 and i == 0)) and
                                    (BLOCK_M % 32 == 0 or weight_group_m_idx + lane_idx < BLOCK_M)) {
                                    stored_cached_weight = *buffer.l1_topk_weights_buffer
                                        .get_data_buffer(ring_m_idx + weight_group_m_idx + lane_idx)
                                        .template get_base_ptr<float>();
                                }

                                const float2 weights = {
                                    ptx::exchange(stored_cached_weight, atom_m_idx % 32 + (lane_idx % 4) * 2 + 0),
                                    ptx::exchange(stored_cached_weight, atom_m_idx % 32 + (lane_idx % 4) * 2 + 1)
                                };

                                uint2 raw_values[4];
                                uint32_t tmem_addr = sub_accum_base + atom_m_idx;
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                       raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                       raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                                cutlass::arch::fence_view_async_tmem_load();

                                if ((n_sub == kNumNSubtiles - 1) and (s * kNumAtomsPerStore + i == wg_atoms - 1)) {
                                    ptx::tcgen05_before_thread_sync();
                                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                }

                                auto fp32_values = reinterpret_cast<float2*>(raw_values);
                                #pragma unroll
                                for (uint32_t k = 0; k < 2; ++ k) {
                                    auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                                    auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);
                                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                        bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                        bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                        bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                                    }
                                    auto gate = __bfloat1622float2(bf16_gate);
                                    auto neg_gate_exp = make_float2(
                                        kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                        kFastMath ? __expf(-gate.y) : expf(-gate.y));
                                    const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                                    if constexpr (kFastMath) {
                                        gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                                    } else {
                                        gate = {gate.x / denom.x, gate.y / denom.y};
                                    }
                                    const auto up = __bfloat1622float2(bf16_up);
                                    activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                                }

                                float2 thread_local_amax = {0.f, 0.f};
                                #pragma unroll
                                for (uint32_t k = 0; k < 2; ++ k) {
                                    thread_local_amax.x = cute::max(thread_local_amax.x, cute::abs(activation_values[i][k].x));
                                    thread_local_amax.y = cute::max(thread_local_amax.y, cute::abs(activation_values[i][k].y));
                                }
                                amax_values[i].x = math::warp_reduce<4, true>(
                                    thread_local_amax.x, math::ReduceMax<float>());
                                amax_values[i].y = math::warp_reduce<4, true>(
                                    thread_local_amax.y, math::ReduceMax<float>());

                                if (lane_idx < 4)
                                    shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                                __syncwarp();
                            }

                            const uint32_t tma_stage_idx = (tma_store_seq ++) % kNumTMAStoreStages;
                            ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                                if (DG_D3_DYN_EPI == 2 and i >= atoms_in_pass)
                                    break;
                                const float2 wp_amax =
                                    shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                                amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                                amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                                float2 sf, sf_inv;
                                math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                                const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                                const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);
                                const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                                uint32_t row = lane_idx;
                                uint32_t col = warp_idx_in_wg;
                                const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                                    + i * ATOM_M * L1_OUT_BLOCK_N
                                    + row * L1_OUT_BLOCK_N
                                    + (col ^ (row / 2)) * kNumBankGroupBytes;
                                ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                                if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                                    const uint32_t k_idx = sub_n_block_idx * 2 + warp_idx_in_wg / 2;
                                    const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                                    const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                                    const auto sf_base_ptr = task_info.is_shared() ?
                                        buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                                    const uint32_t token_base_idx = store_m_idx + i * ATOM_M;
                                    __builtin_assume(token_base_idx < BLOCK_M);
                                    const auto sf_token_idx = block_idx * SF_BLOCK_M
                                        + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                                    const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                                    sf_base_ptr[sf_addr] =
                                        (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                                    sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                        (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                                }
                                __syncwarp();
                            }
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                            if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                                const uint32_t out_n_idx = sub_n_block_idx * L1_OUT_BLOCK_N;
                                const auto tensor_map_l1_output_ptr = task_info.is_shared()
                                    ? &tensor_map_shared_l1_output
#if DG_D3_HAS_EXTRA_TMAS
                                    : ((DG_D3_DYN_EPI == 2 and atoms_in_pass == 1u)
                                        ? &tensor_map_l1_output_m8
                                        : &tensor_map_l1_output);
#else
                                    : &tensor_map_l1_output;
#endif
                                cute::tma_store_fence();
                                cute::SM90_TMA_STORE_2D::copy(
                                    tensor_map_l1_output_ptr,
                                    shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                                    out_n_idx,
                                    m_idx + store_m_idx);
                                cute::tma_store_arrive();
                            }
                            __syncwarp();
                        }
                    }
                }
#else
                #pragma unroll
                for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                    const uint32_t sub_accum_base = accum_stage_idx * kNumAccumColsPerStage + n_sub * UMMA_N;
                    const uint32_t sub_n_block_idx = n_block_idx * kNumNSubtiles + n_sub;
                    float stored_cached_weight = 1.0f;
                    const uint32_t num_valid_store_blocks_l1 = math::ceil_div(num_valid_wg_rows, STORE_BLOCK_M);
                    uint2 prefetched_l1[kNumEpiPrefetchStages + 1][kNumAtomsPerStore][4];
                    const auto issue_l1_epi_block = [&](const uint32_t& sb) {
                        if (sb >= num_valid_store_blocks_l1)
                            return;
                        #pragma unroll
                        for (uint32_t pi = 0; pi < kNumAtomsPerStore; ++ pi) {
                            auto& rv = prefetched_l1[sb % (kNumEpiPrefetchStages + 1)][pi];
                            const uint32_t tmem_addr = sub_accum_base + epilogue_wg_idx * WG_BLOCK_M +
                                (sb * kNumAtomsPerStore + pi) * ATOM_M;
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                   rv[0].x, rv[0].y, rv[1].x, rv[1].y);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                   rv[2].x, rv[2].y, rv[3].x, rv[3].y);
                        }
                        if (sb + 1u == num_valid_store_blocks_l1) {
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }
                    };
                    if constexpr (kNumEpiPrefetchStages > 0) {
                        if (num_valid_store_blocks_l1 == 0u) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        } else {
                            issue_l1_epi_block(0u);
                        }
                    }

                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                        if (s >= num_valid_store_blocks_l1)
                            break;
                        if constexpr (kNumEpiPrefetchStages > 0) {
                            cutlass::arch::fence_view_async_tmem_load();
                            issue_l1_epi_block(s + 1u);
                        }

                        float2 activation_values[kNumAtomsPerStore][2];
                        float2 amax_values[kNumAtomsPerStore];
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                            const uint32_t j = s * kNumAtomsPerStore + i;

                            DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                            if (not task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                                (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                                stored_cached_weight = *buffer.l1_topk_weights_buffer
                                    .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                    .template get_base_ptr<float>();
                            }

                            const float2 weights = {
                                ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                                ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                            };

                            uint2 raw_values_local[4];
                            uint2* raw_values = raw_values_local;
                            if constexpr (kNumEpiPrefetchStages > 0) {
                                raw_values = prefetched_l1[s % (kNumEpiPrefetchStages + 1)][i];
                            } else {
                                uint32_t tmem_addr = sub_accum_base + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                       raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                       raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                                cutlass::arch::fence_view_async_tmem_load();

                                if (n_sub == kNumNSubtiles - 1 and j == WG_BLOCK_M / ATOM_M - 1) {
                                    ptx::tcgen05_before_thread_sync();
                                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                }
                            }

                            auto fp32_values = reinterpret_cast<float2*>(raw_values);
                            #pragma unroll
                            for (uint32_t k = 0; k < 2; ++ k) {
                                auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                                auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                    bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                    bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                    bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                                }

                                auto gate = __bfloat1622float2(bf16_gate);
                                auto neg_gate_exp = make_float2(
                                    kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                    kFastMath ? __expf(-gate.y) : expf(-gate.y));
                                const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                                if constexpr (kFastMath) {
                                    gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                                } else {
                                    gate = {gate.x / denom.x, gate.y / denom.y};
                                }
                                const auto up = __bfloat1622float2(bf16_up);
                                activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                            }

                            float2 thread_local_amax = {0.f, 0.f};
                            #pragma unroll
                            for (uint32_t k = 0; k < 2; ++ k) {
                                thread_local_amax.x = cute::max(thread_local_amax.x, cute::abs(activation_values[i][k].x));
                                thread_local_amax.y = cute::max(thread_local_amax.y, cute::abs(activation_values[i][k].y));
                            }

                            amax_values[i].x = math::warp_reduce<4, true>(
                                thread_local_amax.x, math::ReduceMax<float>());
                            amax_values[i].y = math::warp_reduce<4, true>(
                                thread_local_amax.y, math::ReduceMax<float>());

                            if (lane_idx < 4)
                                shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                            __syncwarp();
                        }

                        const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                        ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        #pragma unroll
                        for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                            const float2 wp_amax =
                                shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                            amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                            amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                            float2 sf, sf_inv;
                            math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                            const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                            const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);
                            const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                            uint32_t row = lane_idx;
                            uint32_t col = warp_idx_in_wg;
                            const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                                + i * ATOM_M * L1_OUT_BLOCK_N
                                + row * L1_OUT_BLOCK_N
                                + (col ^ (row / 2)) * kNumBankGroupBytes;
                            ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                            if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                                const uint32_t k_idx = sub_n_block_idx * 2 + warp_idx_in_wg / 2;
                                const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                                const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                                const auto sf_base_ptr = task_info.is_shared() ?
                                    buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                                const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                                __builtin_assume(token_base_idx < BLOCK_M);
                                const auto sf_token_idx = block_idx * SF_BLOCK_M
                                    + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                                const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                                sf_base_ptr[sf_addr] =
                                    (*reinterpret_cast<uint32_t*>(&sf.x) >> 23);
                                sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                    (*reinterpret_cast<uint32_t*>(&sf.y) >> 23);
                            }
                            __syncwarp();
                        }
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            uint32_t out_n_idx = sub_n_block_idx * L1_OUT_BLOCK_N;
                            const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                tensor_map_l1_output_ptr,
                                shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                                out_n_idx,
                                m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                    }
                }
#endif

                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), kNumNSubtiles);
                    } else {
                        ptx::red_add_rel(
                            workspace.get_l2_full_count_ptr(ring_block_idx), kNumNSubtiles);
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_epi_s, tile_prof::read_gtimer(),
                                  kD7QueueScatterActive ? 23u : 11u, pool_block_idx);
#endif
            } else {
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M_L2 % 8 == 0, "Invalid store M");

#if DG_D7_HANDOFF_MODE != 0
                #pragma unroll
                for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                    const uint32_t sub_accum_base = accum_stage_idx * kNumAccumColsPerStage + n_sub * UMMA_N;
                    const uint32_t n_idx = base_n_idx + n_sub * LAYOUT_AD_M;
                    const bool is_last_n_sub = (n_sub == kNumNSubtiles - 1);

                    const auto run_d7_l2_epilogue = [&]<uint32_t kChunkM, bool kWg0Only>() {
                        constexpr uint32_t kNumSubBlocks = WG_BLOCK_M / kChunkM;
                        constexpr uint32_t kAtoms = kChunkM / ATOM_M;
                        constexpr uint32_t kRowsPerEpiWarp = kChunkM / 8u;
                        bool tmem_released = false;

                        #pragma unroll
                        for (uint32_t s = 0; s < kNumSubBlocks; ++ s) {
                            const uint32_t s_row_offset = s * kChunkM;
                            if (s_row_offset >= valid_m) {
                                break;
                            }

                            const bool wg_has_rows = (not kWg0Only or epilogue_wg_idx == 0u) and
                                (epilogue_wg_idx * WG_BLOCK_M + s_row_offset < valid_m);
                            const bool is_last_valid_s_all =
                                (s + 1u == kNumSubBlocks) or ((s + 1u) * kChunkM >= valid_m);
                            const bool is_last_valid_s_wg =
                                (s + 1u == kNumSubBlocks) or
                                (epilogue_wg_idx * WG_BLOCK_M + s_row_offset + kChunkM >= valid_m);

                            uint32_t packed_bf16[kAtoms][4];
                            if (wg_has_rows) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAtoms; ++ i) {
                                    const uint32_t atom_m_base = epilogue_wg_idx * WG_BLOCK_M + s_row_offset + i * ATOM_M;
                                    if (atom_m_base < valid_m) {
                                        const uint32_t tmem_addr = sub_accum_base + atom_m_base;
                                        uint32_t values[ATOM_M];
                                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                                            tmem_addr, values[0], values[1], values[2], values[3]);
                                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                                            tmem_addr | 0x00100000, values[4], values[5], values[6], values[7]);
                                        cutlass::arch::fence_view_async_tmem_load();

                                        const bool is_last_valid_atom =
                                            (i + 1u == kAtoms) or (atom_m_base + ATOM_M >= valid_m);
                                        if (is_last_n_sub and is_last_valid_s_wg and is_last_valid_atom and not tmem_released) {
                                            ptx::tcgen05_before_thread_sync();
                                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                            tmem_released = true;
                                        }

                                        packed_bf16[i][0] = math::cast_into_bf16_and_pack(values[0], values[1]);
                                        packed_bf16[i][1] = math::cast_into_bf16_and_pack(values[2], values[3]);
                                        packed_bf16[i][2] = math::cast_into_bf16_and_pack(values[4], values[5]);
                                        packed_bf16[i][3] = math::cast_into_bf16_and_pack(values[6], values[7]);
                                    }
                                }
                            } else if (is_last_n_sub and not tmem_released) {
                                ptx::tcgen05_before_thread_sync();
                                shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                tmem_released = true;
                            }

                            bool offload_to_dispatch = false;
                            const uint32_t mode_ring_idx = (epi_scatter_step ++) & 3u;
                            if constexpr (DG_D7_HANDOFF_MODE != 2) {
                                if (epilogue_warp_idx == 0 and lane_idx == 0) {
                                    const uint32_t pull_done = scatter_handoff_enabled ?
                                        1u : ptx::ld_shared(&shared_storage.dispatch_pull_done);
                                    bool skip_final = false;
                                    if constexpr (DG_D7_HANDOFF_MODE == 1) {
                                        if (is_last_n_sub and is_last_valid_s_all and kNumEpilogueWarps >= kNumDispatchWarps) {
                                            if (scheduler.task_info_full_barriers[scheduler.sched_stage_idx].test_wait(scheduler.sched_phase)) {
                                                skip_final = (scheduler.task_infos[scheduler.sched_stage_idx].block_phase == sched::BlockPhase::None);
                                            }
                                        }
                                    }
                                    const uint32_t slot_ok = (pull_done != 0u and not skip_final and kD7UseDualDeadSlots and
                                        shared_storage.scatter_empty_barriers[epi_scatter_slot].test_wait(epi_scatter_phase ^ 1u)) ? 1u : 0u;
                                    ptx::st_shared(&shared_storage.epi_scatter_mode[mode_ring_idx], (pull_done << 1) | slot_ok);
                                }
                                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                                const uint32_t mode_bits = ptx::ld_shared(&shared_storage.epi_scatter_mode[mode_ring_idx]);
                                scatter_handoff_enabled = (mode_bits & 2u) != 0u;
                                offload_to_dispatch = (mode_bits & 1u) != 0u;
                            } else {
                                if (not scatter_handoff_enabled) {
                                    if (epilogue_warp_idx == 0 and lane_idx == 0) {
                                        ptx::st_shared(&shared_storage.epi_scatter_mode[mode_ring_idx], ptx::ld_shared(&shared_storage.dispatch_pull_done));
                                    }
                                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                                    scatter_handoff_enabled = (ptx::ld_shared(&shared_storage.epi_scatter_mode[mode_ring_idx]) != 0u);
                                }
                                if (scatter_handoff_enabled and kD7UseDualDeadSlots) {
                                    shared_storage.scatter_empty_barriers[epi_scatter_slot].wait(epi_scatter_phase ^ 1u);
                                    offload_to_dispatch = true;
                                }
                            }

                            if (offload_to_dispatch) {
                                if (wg_has_rows) {
                                    uint8_t* slot_wg_ptr = (d7_get_scatter_base_ptr() + epi_scatter_slot * kD7SlotBytes) +
                                        (kWg0Only ? 0u : epilogue_wg_idx * (kChunkM * 256u));
                                    const uint32_t row = lane_idx % 8u;
                                    const uint32_t col = (epilogue_warp_idx % 2u) * 4u + lane_idx / 8u;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kAtoms; ++ i) {
                                        const uint32_t atom_m_base = epilogue_wg_idx * WG_BLOCK_M + s_row_offset + i * ATOM_M;
                                        if (atom_m_base < valid_m) {
                                            uint8_t* smem_ptr = slot_wg_ptr +
                                                (warp_idx_in_wg / 2u) * kChunkM * kSwizzleCDMode +
                                                i * ATOM_M * kSwizzleCDMode +
                                                row * (kNumBankGroupBytes * 8u) +
                                                (col ^ row) * kNumBankGroupBytes;
                                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                                packed_bf16[i][0], packed_bf16[i][1],
                                                packed_bf16[i][2], packed_bf16[i][3],
                                                smem_ptr
                                            );
                                        }
                                    }
                                }
                                if (epilogue_warp_idx == 0 and lane_idx == 0) {
                                    const uint32_t packed_meta =
                                        (valid_m & 0xffffu) |
                                        ((s_row_offset & 0x1fffu) << 16) |
                                        (kWg0Only ? 0x20000000u : 0u) |
                                        (task_info.is_shared() ? 0x40000000u : 0u);
                                    ptx::st_shared(
                                        &shared_storage.scatter_jobs[epi_scatter_slot],
                                        pool_m_idx, n_idx, packed_meta, 0u);
                                }
                                shared_storage.scatter_full_barriers[epi_scatter_slot].arrive();
                                epi_scatter_slot ^= 1u;
                                epi_scatter_phase ^= (epi_scatter_slot == 0u);
                            } else {
                                if (wg_has_rows) {
                                    uint8_t* slot_wg_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]);
                                    const uint32_t row = lane_idx % 8u;
                                    const uint32_t col = (epilogue_warp_idx % 2u) * 4u + lane_idx / 8u;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kAtoms; ++ i) {
                                        const uint32_t atom_m_base = epilogue_wg_idx * WG_BLOCK_M + s_row_offset + i * ATOM_M;
                                        if (atom_m_base < valid_m) {
                                            uint8_t* smem_ptr = slot_wg_ptr +
                                                (warp_idx_in_wg / 2u) * kChunkM * kSwizzleCDMode +
                                                i * ATOM_M * kSwizzleCDMode +
                                                row * (kNumBankGroupBytes * 8u) +
                                                (col ^ row) * kNumBankGroupBytes;
                                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                                packed_bf16[i][0], packed_bf16[i][1],
                                                packed_bf16[i][2], packed_bf16[i][3],
                                                smem_ptr
                                            );
                                        }
                                    }
                                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                                    const uint32_t row_in_atom = (warp_idx_in_wg * 2u + lane_idx / 16u) % ATOM_M;
                                    const uint32_t bank_group_idx = lane_idx % 8u;
                                    #pragma unroll
                                    for (uint32_t j = 0; j < kRowsPerEpiWarp; ++ j) {
                                        const uint32_t row_in_store = j * 8u + warp_idx_in_wg * 2u + lane_idx / 16u;
                                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s_row_offset + row_in_store;
                                        if (m_idx_in_block >= valid_m)
                                            break;

                                        uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                                        if (task_info.is_shared()) {
                                            dst_rank_idx = sym_buffer.rank_idx;
                                            dst_token_idx = pool_m_idx + m_idx_in_block;
                                            dst_topk_idx = kNumTopk;
                                        } else {
                                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                                            dst_rank_idx = src_metadata.rank_idx;
                                            dst_token_idx = src_metadata.token_idx;
                                            dst_topk_idx = src_metadata.topk_idx;
                                        }

                                        const auto smem_ptr = slot_wg_ptr +
                                            (lane_idx % 16u / 8u) * kChunkM * kSwizzleCDMode +
                                            row_in_store * kSwizzleCDMode +
                                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                                        const auto packed = ptx::ld_shared(reinterpret_cast<const float4*>(smem_ptr));
                                        d7_scatter_or_prereduce_row(
                                            task_info.is_shared(), dst_rank_idx, dst_token_idx, dst_topk_idx, n_idx, packed);
                                    }
                                }
                                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                            }
                        }

                        if (is_last_n_sub and not tmem_released) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }
                    };

                    if constexpr (kD7AllowDynamicWg0Full32) {
                        if (valid_m <= WG_BLOCK_M) {
                            run_d7_l2_epilogue.template operator()<32u, true>();
                        } else {
                            run_d7_l2_epilogue.template operator()<kD7DualWgChunkM, false>();
                        }
                    } else {
                        run_d7_l2_epilogue.template operator()<kD7DualWgChunkM, false>();
                    }
                }
#elif DG_D3_DYN_EPI > 0
                constexpr uint32_t kMaxRowsPerWarp = STORE_BLOCK_M_L2 / 8;

                if (wg_atoms == 0u) {
                    ptx::tcgen05_before_thread_sync();
                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                } else {
                    #pragma unroll
                    for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                        const uint32_t sub_accum_base = accum_stage_idx * kNumAccumColsPerStage + n_sub * UMMA_N;
                        const uint32_t n_idx = base_n_idx + n_sub * LAYOUT_AD_M;

                        #pragma unroll
                        for (uint32_t s = 0; s < kMaxPassesPerWg; ++ s) {
                            if (s >= num_passes)
                                break;
                            const uint32_t atoms_in_pass = (DG_D3_DYN_EPI == 2)
                                ? cute::min(2u, wg_atoms - s * 2u)
                                : kNumAtomsPerStore;
                            const uint32_t runtime_store_m = atoms_in_pass * ATOM_M;
                            const uint32_t store_m_idx = (wg_atom_begin + s * kNumAtomsPerStore) * ATOM_M;

                            #pragma unroll
                            for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                                if (DG_D3_DYN_EPI == 2 and i >= atoms_in_pass)
                                    break;
                                uint32_t tmem_addr = sub_accum_base + store_m_idx + i * ATOM_M;
                                uint32_t values[ATOM_M];
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                       values[0], values[1], values[2], values[3]);
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                       values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();

                                if (i == 0 and (s > 0 or n_sub > 0))
                                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                                if ((n_sub == kNumNSubtiles - 1) and (s * kNumAtomsPerStore + i == wg_atoms - 1)) {
                                    ptx::tcgen05_before_thread_sync();
                                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                }

                                uint32_t row = lane_idx % 8;
                                uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                                const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                                    (warp_idx_in_wg / 2) * runtime_store_m * kSwizzleCDMode +
                                    i * ATOM_M * kSwizzleCDMode +
                                    row * (kNumBankGroupBytes * 8) +
                                    (col ^ row) * kNumBankGroupBytes;
                                ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                    math::cast_into_bf16_and_pack(values[0], values[1]),
                                    math::cast_into_bf16_and_pack(values[2], values[3]),
                                    math::cast_into_bf16_and_pack(values[4], values[5]),
                                    math::cast_into_bf16_and_pack(values[6], values[7]),
                                    smem_ptr
                                );
                            }

                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                            const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                            const uint32_t bank_group_idx = lane_idx % 8;

                            #pragma unroll
                            for (uint32_t j = 0; j < kMaxRowsPerWarp; ++ j) {
                                if (DG_D3_DYN_EPI == 2 and j >= atoms_in_pass)
                                    break;
                                const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                                const uint32_t m_idx_in_block = store_m_idx + row_in_store;
                                if (m_idx_in_block >= valid_m)
                                    break;

                                uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                                if (task_info.is_shared()) {
                                    dst_rank_idx = sym_buffer.rank_idx;
                                    dst_token_idx = pool_m_idx + m_idx_in_block;
                                    dst_topk_idx = kNumTopk;
                                } else {
                                    const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                                    dst_rank_idx = src_metadata.rank_idx;
                                    dst_token_idx = src_metadata.token_idx;
                                    dst_topk_idx = src_metadata.topk_idx;
                                }

                                const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                                    (lane_idx % 16 / 8) * runtime_store_m * kSwizzleCDMode +
                                    row_in_store * kSwizzleCDMode +
                                    (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                                const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));
                                d7_scatter_or_prereduce_row(
                                    task_info.is_shared(), dst_rank_idx, dst_token_idx, dst_topk_idx, n_idx, packed);
                            }
                        }
                    }
                }

                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
#else
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M_L2 / 8;

                #pragma unroll
                for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                    const uint32_t sub_accum_base = accum_stage_idx * kNumAccumColsPerStage + n_sub * UMMA_N;
                    const uint32_t n_idx = base_n_idx + n_sub * LAYOUT_AD_M;
                    constexpr uint32_t kNumAtomsPerStoreL2 = STORE_BLOCK_M_L2 / ATOM_M;
                    const uint32_t num_valid_store_blocks_l2 = math::ceil_div(num_valid_wg_rows, STORE_BLOCK_M_L2);
                    uint32_t prefetched_l2[kNumEpiPrefetchStages + 1][kNumAtomsPerStoreL2][ATOM_M];
                    const auto issue_l2_epi_block = [&](const uint32_t& sb) {
                        if (sb >= num_valid_store_blocks_l2)
                            return;
                        #pragma unroll
                        for (uint32_t pi = 0; pi < kNumAtomsPerStoreL2; ++ pi) {
                            auto& v = prefetched_l2[sb % (kNumEpiPrefetchStages + 1)][pi];
                            const uint32_t tmem_addr = sub_accum_base + epilogue_wg_idx * WG_BLOCK_M +
                                (sb * kNumAtomsPerStoreL2 + pi) * ATOM_M;
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                   v[0], v[1], v[2], v[3]);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                   v[4], v[5], v[6], v[7]);
                        }
                        if (sb + 1u == num_valid_store_blocks_l2) {
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }
                    };
                    if constexpr (kNumEpiPrefetchStages > 0) {
                        if (num_valid_store_blocks_l2 == 0u) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        } else {
                            issue_l2_epi_block(0u);
                        }
                    }

                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M_L2; ++ s) {
                        if (s >= num_valid_store_blocks_l2)
                            break;
                        if constexpr (kNumEpiPrefetchStages > 0) {
                            cutlass::arch::fence_view_async_tmem_load();
                            issue_l2_epi_block(s + 1u);
                        }

                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_M_L2 / ATOM_M; ++ i) {
                            uint32_t values_local[ATOM_M];
                            uint32_t* values = values_local;
                            if constexpr (kNumEpiPrefetchStages > 0) {
                                values = prefetched_l2[s % (kNumEpiPrefetchStages + 1)][i];
                            } else {
                                uint32_t tmem_addr = sub_accum_base + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + i * ATOM_M;
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                       values[0], values[1], values[2], values[3]);
                                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                       values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();

                                if (n_sub == kNumNSubtiles - 1 and s == WG_BLOCK_M / STORE_BLOCK_M_L2 - 1 and i == STORE_BLOCK_M_L2 / ATOM_M - 1) {
                                    ptx::tcgen05_before_thread_sync();
                                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                }
                            }

                            if (i == 0 and (s > 0 or n_sub > 0))
                                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                            uint32_t row = lane_idx % 8;
                            uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                            const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                                (warp_idx_in_wg / 2) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                                i * ATOM_M * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) +
                                (col ^ row) * kNumBankGroupBytes;
                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                math::cast_into_bf16_and_pack(values[0], values[1]),
                                math::cast_into_bf16_and_pack(values[2], values[3]),
                                math::cast_into_bf16_and_pack(values[4], values[5]),
                                math::cast_into_bf16_and_pack(values[6], values[7]),
                                smem_ptr
                            );
                        }

                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                        const uint32_t bank_group_idx = lane_idx % 8;

                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                            const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + row_in_store;
                            if (m_idx_in_block >= valid_m)
                                break;

                            uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                            if (task_info.is_shared()) {
                                dst_rank_idx = sym_buffer.rank_idx;
                                dst_token_idx = pool_m_idx + m_idx_in_block;
                                dst_topk_idx = kNumTopk;
                            } else {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                                dst_rank_idx = src_metadata.rank_idx;
                                dst_token_idx = src_metadata.token_idx;
                                dst_topk_idx = src_metadata.topk_idx;
                            }

                            const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                                (lane_idx % 16 / 8) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                                row_in_store * kSwizzleCDMode +
                                (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                            const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));
                            d7_scatter_or_prereduce_row(
                                task_info.is_shared(), dst_rank_idx, dst_token_idx, dst_topk_idx, n_idx, packed);
                        }
                    }
                }

                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
#endif

                if constexpr (kD6GroupTiles > 0) {
                    if (epilogue_warp_idx == 0) {
                        const uint32_t group_idx = n_block_idx / kD6GroupTiles;
                        uint32_t done = 1u;
                        if constexpr (kD6GroupTiles > 1) {
                            if (cute::elect_one_sync()) {
                                auto* const cnt_ptr = buffer.get_d6_local_group_cnt_ptr(
                                    task_info.is_shared(), pool_block_idx, group_idx);
                                const uint32_t old_cnt = ptx::atomic_add_acq_rel_gpu(cnt_ptr, 1u);
                                done = (((old_cnt + 1u) & (kD6GroupTiles - 1u)) == 0u) ? 1u : 0u;
                            }
                            done = ptx::exchange(done, 0);
                        }
                        if (done != 0u) {
                            ptx::fence_acq_rel_sys();
                            for (uint32_t m_row = lane_idx; m_row < valid_m; m_row += 32u) {
                                uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                                if (task_info.is_shared()) {
                                    dst_rank_idx = sym_buffer.rank_idx;
                                    dst_token_idx = pool_m_idx + m_row;
                                    dst_topk_idx = kNumTopk;
                                } else {
                                    const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_row);
                                    dst_rank_idx = src_metadata.rank_idx;
                                    dst_token_idx = src_metadata.token_idx;
                                    dst_topk_idx = src_metadata.topk_idx & 0xffu;
                                }
                                uint16_t* const ready_ptr =
                                    buffer.get_d6_ready_epoch_ptr(dst_token_idx, group_idx) + dst_topk_idx;
                                ptx::st_relaxed_sys(sym_buffer.map(ready_ptr, dst_rank_idx), d6_cur_epoch);
                            }
                        }
                    }
                }
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_epi_s, tile_prof::read_gtimer(),
                                  12u, pool_block_idx);
#endif
            }
        }

#if DG_D7_HANDOFF_MODE != 0
        shared_storage.scatter_empty_barriers[epi_scatter_slot].wait(epi_scatter_phase ^ 1u);
        if (epilogue_warp_idx == 0 and lane_idx == 0) {
            ptx::st_shared(&shared_storage.scatter_jobs[epi_scatter_slot], 0u, 0u, 0x80000000u, 0u);
        }
        shared_storage.scatter_full_barriers[epi_scatter_slot].arrive();
        ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kScatterDrainBarrierIdx);
#elif DG_D7_QUEUE_SCATTER != 0
        if constexpr (kD7QueueScatterActive) {
            d7_queue_flush();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            if (epilogue_warp_idx == 0 and lane_idx == 0) {
                ptx::red_add_rel(buffer.get_d7_queue_producer_done_ptr(), 1u);
            }
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kScatterDrainBarrierIdx);
        }
#endif

        // Deallocate tensor memory
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        if constexpr (kD6ReduceWarps == 16u and not kD2SmidMasked) {
            if constexpr (kD6GroupTiles == 0) {
#if DG_MEGA_TILE_PROF
                const uint64_t prof_pre_s = tile_prof::read_gtimer();
#endif
                comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                     kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                    workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                    [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
                );
                ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_pre_s, tile_prof::read_gtimer(),
                                  13u, 0u);
#endif
            } else {
                ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            }
        } else if constexpr (kD6GroupTiles == 0) {
#if DG_MEGA_TILE_PROF
            const uint64_t prof_pre_s = tile_prof::read_gtimer();
#endif
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );

            ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#if DG_MEGA_TILE_PROF
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_pre_s, tile_prof::read_gtimer(),
                              13u, 0u);
#endif

            constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
            constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);
            constexpr uint32_t kNumChunkSlots = 3;
            constexpr uint32_t kNumMaxRegistersForBuffer = (kD2Is20Warp or kNumEpilogueRegisters <= 160u) ? 64u : 128u;

            constexpr uint32_t kBaseChunks =
                (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and
                 kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1u : 2u;
            constexpr uint32_t kNumChunks =
                (kNumChunkSlots * kNumEpilogueWarps * (kNumHiddenBytes / kBaseChunks) <= kNumReusableSmemBytes and
                 (kHidden / kBaseChunks) <= 16 * kNumMaxRegistersForBuffer)
                    ? kBaseChunks : 4u;
            constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
            constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
            constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
            DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
            DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes, "Hidden is too large");
            DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
            DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
            DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
            DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

            DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

            const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
                return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
            });
            const auto combine_store_buffer = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);
            auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
                return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
            });

            uint32_t combine_phase = 0;
            uint32_t load_stage_idx = 0;
            for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
                 token_idx < num_tokens;
                 token_idx += kNumSMs * kNumEpilogueWarps) {
                const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                    static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                    (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                bool is_active_combine_slot = (stored_topk_slot_idx >= 0);
                if constexpr (kD7LocalDedupActive) {
                    const uint32_t exp_rank = (lane_idx < kNumTopk and stored_topk_slot_idx >= 0)
                        ? (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank)
                        : (0xffff0000u | lane_idx);
                    const uint32_t same_r_mask = __match_any_sync(0xffffffffu, exp_rank);
                    if (lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and exp_rank != sym_buffer.rank_idx) {
                        is_active_combine_slot = (lane_idx == static_cast<uint32_t>(__ffs(same_r_mask) - 1));
                    }
                }
                const uint32_t total_mask = __ballot_sync(0xffffffff, is_active_combine_slot);

                for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
#if DG_MEGA_TILE_PROF
                    const uint64_t prof_red_s = tile_prof::read_gtimer();
#endif
                    const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                    uint32_t mask = total_mask;
                    const auto move_mask_and_load = [&](const uint32_t& i) {
                        if (mask) {
                            const uint32_t slot_idx = __ffs(mask) - 1;
                            mask ^= 1 << slot_idx;
                            if (cute::elect_one_sync()) {
                                const auto src_ptr = math::advance_ptr<uint8_t>(
                                    buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                        .get_data_buffer(token_idx).get_base_ptr(),
                                    chunk_byte_offset);
                                ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                                ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                            }
                            __syncwarp();
                            return true;
                        }
                        return false;
                    };

                    bool do_reduce = move_mask_and_load(load_stage_idx);
                    float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                    while (do_reduce) {
                        do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                        combine_load_barriers[load_stage_idx]->wait(combine_phase);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                            const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                        }
                        combine_phase ^= load_stage_idx;
                        load_stage_idx ^= 1;
                    }

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
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
                            math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                            combine_store_buffer, kNumChunkBytes);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
#if DG_MEGA_TILE_PROF
                    tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                      prof_rec_idx, prof_t0_ns, prof_red_s, tile_prof::read_gtimer(),
                                      14u, token_idx * kNumChunks + chunk);
#endif
                }
            }
        } else {
            // Axis 6 (D6 V3): Fine combine with static token mapping & streaming readiness prefetch
            ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

            constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
            constexpr uint32_t kGroupBytes = kD6GroupTiles * BLOCK_N * sizeof(nv_bfloat16);
            constexpr uint32_t kNumGroups = kNumHiddenBytes / kGroupBytes;
            constexpr uint32_t kNumChunkSlots = 3;
            constexpr uint32_t kMaxSubChunkBytes = (kD2Is20Warp or kNumEpilogueRegisters <= 160u) ? 4096u : 8192u;
            constexpr uint32_t kBaseSubChunks = (kGroupBytes > kMaxSubChunkBytes) ? (kGroupBytes / kMaxSubChunkBytes) : 1u;
            constexpr uint32_t kNumSubChunks =
                (kNumChunkSlots * kNumEpilogueWarps * (kGroupBytes / kBaseSubChunks) <= kNumReusableSmemBytes)
                    ? kBaseSubChunks : (kBaseSubChunks * 2u);
            constexpr uint32_t kSubChunkBytes = kGroupBytes / kNumSubChunks;
            constexpr uint32_t kActiveCombineWarps =
                (kD6ReduceWarps == 0u or kD6ReduceWarps >= kNumEpilogueWarps) ? kNumEpilogueWarps : kD6ReduceWarps;

            using VecT = std::conditional_t<(kSubChunkBytes == 256u), uint2, uint4>;
            constexpr uint32_t kVecBytes = sizeof(VecT);
            constexpr uint32_t kElemsPerVec = kVecBytes / sizeof(nv_bfloat162);
            constexpr uint32_t kVecsPerLane = kSubChunkBytes / (32u * kVecBytes);

            DG_STATIC_ASSERT(kNumHiddenBytes % kGroupBytes == 0, "Hidden bytes must be divisible by D6 group bytes");
            DG_STATIC_ASSERT(kGroupBytes % kNumSubChunks == 0, "Group bytes must be divisible by sub-chunks");
            DG_STATIC_ASSERT(kSubChunkBytes % (32u * kVecBytes) == 0, "Invalid D6 vectorization");
            DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kSubChunkBytes <= kNumReusableSmemBytes,
                             "D6 sub-chunk bytes exceed reusable shared memory");
            DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u,
                             "Top-k + shared must fit in a single warp");

            if (epilogue_warp_idx < kActiveCombineWarps) {
                const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
                    return math::advance_ptr<VecT>(
                        smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kSubChunkBytes);
                });
                const auto combine_store_buffer = math::advance_ptr<VecT>(
                    smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kSubChunkBytes);
                auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
                    return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
                });

                uint32_t combine_phase = 0;
                uint32_t load_stage_idx = 0;
                for (uint32_t token_idx = sm_idx * kActiveCombineWarps + epilogue_warp_idx;
                     token_idx < num_tokens;
                     token_idx += kNumSMs * kActiveCombineWarps) {
                    const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                        static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                        (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                    const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

                    for (uint32_t group_idx = 0; group_idx < kNumGroups; ++ group_idx) {
                        const uint32_t group_byte_offset = group_idx * kGroupBytes;
                        const uint16_t* const ready_row_ptr = buffer.get_d6_ready_epoch_ptr(token_idx, group_idx);

                        for (uint32_t sub = 0; sub < kNumSubChunks; ++ sub) {
#if DG_MEGA_TILE_PROF
                            const uint64_t prof_red_s = tile_prof::read_gtimer();
#endif
                            const uint32_t sub_byte_offset = group_byte_offset + sub * kSubChunkBytes;
                            uint32_t mask = total_mask;
                            const auto move_mask_and_load = [&](const uint32_t& i) {
                                if (mask) {
                                    const uint32_t slot_idx = __ffs(mask) - 1;
                                    mask ^= 1u << slot_idx;
                                    if (cute::elect_one_sync()) {
                                        if (sub == 0) {
                                            const uint16_t* const ready_ptr = ready_row_ptr + slot_idx;
                                            if (ptx::ld_acq_sys(ready_ptr) != d6_cur_epoch) {
                                                const auto start_clock = clock64();
                                                do {
                                                    if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                                                        printf("DeepGEMM D6 FineCombine timeout: rank=%u sm=%u warp=%u tok=%u grp=%u slot=%u ep=%u\n",
                                                               sym_buffer.rank_idx, sm_idx, epilogue_warp_idx,
                                                               token_idx, group_idx, slot_idx, static_cast<uint32_t>(d6_cur_epoch));
                                                        DG_DEVICE_ASSERT(false and "D6 FineCombine readiness timeout");
                                                    }
                                                    asm volatile("nanosleep.u32 32;");
                                                } while (ptx::ld_acq_sys(ready_ptr) != d6_cur_epoch);
                                            }
                                        }
                                        const auto src_ptr = math::advance_ptr<uint8_t>(
                                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                                .get_data_buffer(token_idx).get_base_ptr(),
                                            sub_byte_offset);
                                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kSubChunkBytes);
                                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kSubChunkBytes);
                                    }
                                    __syncwarp();
                                    return true;
                                }
                                return false;
                            };

                            bool do_reduce = move_mask_and_load(load_stage_idx);
                            float2 reduced[kVecsPerLane * kElemsPerVec] = {};
                            while (do_reduce) {
                                do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                                #pragma unroll
                                for (uint32_t j = 0; j < kVecsPerLane; ++ j) {
                                    const VecT vec_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                                    const auto* bf16_values = reinterpret_cast<const nv_bfloat162*>(&vec_values);
                                    #pragma unroll
                                    for (uint32_t l = 0; l < kElemsPerVec; ++ l)
                                        ptx::accumulate(reduced[j * kElemsPerVec + l], bf16_values[l]);
                                }
                                combine_phase ^= load_stage_idx;
                                load_stage_idx ^= 1;
                            }

                            #pragma unroll
                            for (uint32_t j = 0; j < kVecsPerLane; ++ j) {
                                VecT casted;
                                auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                                #pragma unroll
                                for (uint32_t l = 0; l < kElemsPerVec; ++ l)
                                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kElemsPerVec + l]);

                                if (j == 0) {
                                    ptx::tma_store_wait<0>();
                                    __syncwarp();
                                }
                                if constexpr (kSubChunkBytes == 256u) {
                                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx, casted.x, casted.y);
                                } else {
                                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                                   casted.x, casted.y, casted.z, casted.w);
                                }
                            }
                            __syncwarp();

                            if (cute::elect_one_sync()) {
                                cute::tma_store_fence();
                                ptx::tma_store_1d(
                                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + sub_byte_offset),
                                    combine_store_buffer, kSubChunkBytes);
                                cute::tma_store_arrive();
                            }
                            __syncwarp();
#if DG_MEGA_TILE_PROF
                            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                              prof_rec_idx, prof_t0_ns, prof_red_s, tile_prof::read_gtimer(),
                                              14u, token_idx * (kNumGroups * kNumSubChunks) + group_idx * kNumSubChunks + sub);
#endif
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Axis 6 (D6 All-Warp Reduce): ALL Warps in CTA (W0..W15, 512 threads)
    // converge and execute 1024B Tile-Granularity Reduce together at the end!
    // ========================================================================
    if constexpr (kD6ReduceWarps == 16u and not kD2SmidMasked) {
        constexpr uint32_t kAllReduceWarps = kNumThreads / 32u;
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kSubChunkBytes = 1024u;
        constexpr uint32_t kNumSubChunks = kNumHiddenBytes / kSubChunkBytes;
        constexpr uint32_t kGroupBytes = (kD6GroupTiles > 0u) ? (kD6GroupTiles * BLOCK_N * sizeof(nv_bfloat16)) : kNumHiddenBytes;
        constexpr uint32_t kSubsPerGroup = (kGroupBytes >= kSubChunkBytes) ? (kGroupBytes / kSubChunkBytes) : 1u;
        constexpr uint32_t kVecsPerLane = kSubChunkBytes / (32u * sizeof(uint4));
        constexpr uint32_t kElemsPerVec = sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kSmemAllWarpBytes = kAllReduceWarps * 3u * kSubChunkBytes;
        DG_STATIC_ASSERT(kNumHiddenBytes % kSubChunkBytes == 0, "Hidden bytes must be divisible by 1024B");

        const uint16_t all_d6_epoch = (kD6GroupTiles > 0u) ?
            static_cast<uint16_t>(*buffer.get_d6_epoch_ptr()) : static_cast<uint16_t>(0u);

        ptx::sync_aligned(kNumThreads, kAllWarpReduceBarrierIdx);
        auto* all_combine_barriers = reinterpret_cast<Barrier*>(smem_buffer + kSmemAllWarpBytes);
        if (warp_idx == 0u and lane_idx < kAllReduceWarps * 2u) {
            all_combine_barriers[lane_idx].init(1);
        }
        cutlass::arch::fence_barrier_init();
        ptx::sync_aligned(kNumThreads, kAllWarpReduceBarrierIdx);

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (warp_idx + i * kAllReduceWarps) * kSubChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (warp_idx + kAllReduceWarps * 2u) * kSubChunkBytes);
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &all_combine_barriers[i + warp_idx * 2u];
        });

        uint32_t combine_phase = 0u;
        uint32_t load_stage_idx = 0u;
        const uint32_t total_reduce_tiles = num_tokens * kNumSubChunks;
        for (uint32_t tile_idx = sm_idx * kAllReduceWarps + warp_idx;
             tile_idx < total_reduce_tiles;
             tile_idx += kNumSMs * kAllReduceWarps) {
#if DG_MEGA_TILE_PROF
            uint32_t prof_red_s = tile_prof::read_gtimer();
#endif
            const uint32_t token_idx = tile_idx / kNumSubChunks;
            const uint32_t sub_idx = tile_idx % kNumSubChunks;
            const uint32_t sub_byte_offset = sub_idx * kSubChunkBytes;

            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            bool is_active_combine_slot = (stored_topk_slot_idx >= 0);
            if constexpr (kD7LocalDedupActive) {
                const uint32_t exp_rank = (lane_idx < kNumTopk and stored_topk_slot_idx >= 0)
                    ? (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank)
                    : (0xffff0000u | lane_idx);
                const uint32_t same_r_mask = __match_any_sync(0xffffffffu, exp_rank);
                if (lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and exp_rank != sym_buffer.rank_idx) {
                    is_active_combine_slot = (lane_idx == static_cast<uint32_t>(__ffs(same_r_mask) - 1));
                }
            }
            uint32_t mask = __ballot_sync(0xffffffff, is_active_combine_slot);
            const uint16_t* const ready_row_ptr = (kD6GroupTiles > 0u) ?
                buffer.get_d6_ready_epoch_ptr(token_idx, sub_idx / kSubsPerGroup) : nullptr;
            bool first_slot = true;

            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1u << slot_idx;
                    if (cute::elect_one_sync()) {
                        if constexpr (kD6GroupTiles > 0u) {
                            const uint16_t* const ready_ptr = ready_row_ptr + slot_idx;
                            if (ptx::ld_acq_sys(ready_ptr) != all_d6_epoch) {
                                const auto start_clock = clock64();
                                do {
                                    if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                                        DG_DEVICE_ASSERT(false and "D6 All-Warp Reduce readiness timeout");
                                    }
                                    asm volatile("nanosleep.u32 32;");
                                } while (ptx::ld_acq_sys(ready_ptr) != all_d6_epoch);
                            }
                        }
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            sub_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kSubChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kSubChunkBytes);
                    }
                    __syncwarp();
#if DG_MEGA_TILE_PROF
                    if (first_slot) {
                        prof_red_s = tile_prof::read_gtimer();
                    }
#endif
                    first_slot = false;
                    return true;
                }
                return false;
            };

            bool do_reduce = move_mask_and_load(load_stage_idx);
            float2 reduced[kVecsPerLane * kElemsPerVec] = {};
            while (do_reduce) {
                do_reduce = move_mask_and_load(load_stage_idx ^ 1u);
                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kVecsPerLane; ++ j) {
                    const uint4 vec_values = combine_load_buffer[load_stage_idx][j * 32u + lane_idx];
                    const auto* bf16_values = reinterpret_cast<const nv_bfloat162*>(&vec_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kElemsPerVec; ++ l)
                        ptx::accumulate(reduced[j * kElemsPerVec + l], bf16_values[l]);
                }
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1u;
            }

            #pragma unroll
            for (uint32_t j = 0; j < kVecsPerLane; ++ j) {
                uint4 casted;
                auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kElemsPerVec; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kElemsPerVec + l]);
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(combine_store_buffer + j * 32u + lane_idx,
                               casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();

            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + sub_byte_offset),
                    combine_store_buffer, kSubChunkBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
#if DG_MEGA_TILE_PROF
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_red_s, tile_prof::read_gtimer(),
                              14u, tile_idx);
#endif
        }
        ptx::tma_store_wait<0>();
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
