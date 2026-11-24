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
#ifndef WEAVE_REDUCE_WARPS
#define WEAVE_REDUCE_WARPS 0
#endif
#ifndef WEAVE_SM0_STREAM_SCHED
#define WEAVE_SM0_STREAM_SCHED 0
#endif
// CTA-0 runtime cost-model selection.  It costs a 64 B shared-memory policy
// block plus a per-launch cross-CTA publish/wait handshake, so it is compiled
// out unless SM0 streaming is explicitly requested; with it off the compiled
// SMEM plan matches the reference producer and the pipeline gets its 9th TMA
// stage back.
#ifndef WEAVE_RUNTIME_POLICY
#define WEAVE_RUNTIME_POLICY 0
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
#ifndef WEAVE_DISPATCH_ACTIVE_SMS
#define WEAVE_DISPATCH_ACTIVE_SMS 0
#endif
#ifndef WEAVE_SCATTER_OWNER
#define WEAVE_SCATTER_OWNER 0
#endif
#ifndef WEAVE_SCATTER_STAGING
#define WEAVE_SCATTER_STAGING 0
#endif
#ifndef WEAVE_COMBINE_SYNC_CHUNK
#define WEAVE_COMBINE_SYNC_CHUNK 0
#endif
#ifndef WEAVE_EPI_REG_STAGES
#define WEAVE_EPI_REG_STAGES 0
#endif
#ifndef WEAVE_SCHED_L2_RATIO
#define WEAVE_SCHED_L2_RATIO 1
#endif
#ifndef WEAVE_REG_BALANCE
#define WEAVE_REG_BALANCE 0
#endif
static_assert(WEAVE_SCATTER_OWNER <= 1, "WEAVE_SCATTER_OWNER: 0 = epilogue_only, 1 = comm_only");
static_assert(WEAVE_SCATTER_STAGING <= 2, "WEAVE_SCATTER_STAGING: 0 = reg_2stage, 1 = smem, 2 = hbm_dedup");
static_assert(WEAVE_SCATTER_STAGING == 0 or WEAVE_SCATTER_OWNER == 1,
              "smem / hbm_dedup staging is owned by communication warps (comm_only)");
static_assert(WEAVE_EPI_REG_STAGES <= 2, "WEAVE_EPI_REG_STAGES: 0 = auto, 1, 2");
static_assert(WEAVE_SCHED_L2_RATIO >= 1 and WEAVE_SCHED_L2_RATIO <= 4,
              "WEAVE_SCHED_L2_RATIO must be in [1, 4]");
static_assert(WEAVE_REG_BALANCE <= 1, "WEAVE_REG_BALANCE must be 0 or 1");

// Reset both per-domain queues even when the build does not use them; the cost
// is four stores once per launch and keeps the workspace byte layout stable.
static constexpr unsigned int kNumTaskCountDomains =
    (WEAVE_LOCALITY_DOMAINS > 1u ? static_cast<unsigned int>(WEAVE_LOCALITY_DOMAINS) : 1u);
static_assert(kNumTaskCountDomains == 1u or kNumTaskCountDomains == 2u,
              "SM103 supports one or two locality domains");

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
#if WEAVE_REG_PULL_BYTES != 0 && WEAVE_REG_PULL_BYTES != 1024 && \
    WEAVE_REG_PULL_BYTES != 2048 && WEAVE_REG_PULL_BYTES != 4096
#error "WEAVE_REG_PULL_BYTES must be 0 (TMA SMEM staging), 1024, 2048 or 4096"
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
#if WEAVE_SCATTER_WARPS != 0 && WEAVE_SCATTER_WARPS != 4 && WEAVE_SCATTER_WARPS != 8
#error "WEAVE_SCATTER_WARPS must be 0 (automatic), 4 or 8"
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

// Internal fused SharedLinear2 + INT7 codegen switch.  The host derives
// this from the existing public WEAVE_* preset (see
// csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp); it is never parsed from
// the environment and is not part of the 21-knob surface.
#ifndef WEAVE_FUSED_SHARED_INT7
#define WEAVE_FUSED_SHARED_INT7 0
#endif

// ============================================================================
// Unified 7-Axis Parameter Normalization (DG_TUNE_* <-> DG_ISOLATED_* / DG_D3_*)
// ============================================================================

// Head Dispatch (P1): DG_P1_HEAD_HELPER_K is derived from the public
// WEAVE_V_HEAD knob after every axis macro is normalized (see the block just
// before `namespace deep_gemm`).  DG_P1_HEAD_TOKENS optionally caps the head
// window in tokens (0 = only the ring-capacity / quota cap).
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

// Fused-shared INT7 pack/unpack.  One lane owns eight BF16 values (16 B); lanes 0..7
// and 8..15 form the two 64-column scale groups of a 128-column N block.
// Stored byte q decodes as ((128 + q) - 192) * scale = (q - 64) * scale.
CUTLASS_DEVICE void fused_shared_quantize_int7(const float4& packed,
                                         uint32_t& q0, uint32_t& q1, uint32_t& sc_u16) {
    const auto* const pb2 = reinterpret_cast<const nv_bfloat162*>(&packed);
    nv_bfloat162 m2 = __hmax2(__hmax2(__habs2(pb2[0]), __habs2(pb2[1])),
                              __hmax2(__habs2(pb2[2]), __habs2(pb2[3])));
    m2 = __hmax2(m2, __shfl_xor_sync(0xffffffffu, m2, 1));
    m2 = __hmax2(m2, __shfl_xor_sync(0xffffffffu, m2, 2));
    m2 = __hmax2(m2, __shfl_xor_sync(0xffffffffu, m2, 4));
    const uint32_t m2_u = reinterpret_cast<const uint32_t&>(m2);
    const uint32_t amax_u16 = cute::max(m2_u & 0xffffu, m2_u >> 16);
    const float amax = __uint_as_float(amax_u16 << 16);
    sc_u16 = (__float_as_uint(amax * (1.0f / 62.0f)) + 0xffffu) >> 16;
    const float scale_exact = __uint_as_float(sc_u16 << 16);
    float inv_s_raw;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_s_raw) : "f"(scale_exact));
    const float inv_s = (scale_exact > 0.0f) ? inv_s_raw : 0.0f;
    const nv_bfloat162 inv_s2 = __float2bfloat162_rn(inv_s);
    const nv_bfloat162 magic2 = __float2bfloat162_rn(192.0f);
    const nv_bfloat162 z01_h = __hfma2(pb2[0], inv_s2, magic2);
    const nv_bfloat162 z23_h = __hfma2(pb2[1], inv_s2, magic2);
    const nv_bfloat162 z45_h = __hfma2(pb2[2], inv_s2, magic2);
    const nv_bfloat162 z67_h = __hfma2(pb2[3], inv_s2, magic2);
    q0 = __byte_perm(reinterpret_cast<const uint32_t&>(z01_h),
                     reinterpret_cast<const uint32_t&>(z23_h), 0x6420u);
    q1 = __byte_perm(reinterpret_cast<const uint32_t&>(z45_h),
                     reinterpret_cast<const uint32_t&>(z67_h), 0x6420u);
}

CUTLASS_DEVICE void fused_shared_accumulate_int7(const uint32_t& q0, const uint32_t& q1,
                                           const uint32_t& sc_pair, const uint32_t& group,
                                           nv_bfloat162* acc) {
    const uint32_t sc2_u = __byte_perm(sc_pair, sc_pair, group ? 0x3232u : 0x1010u);
    const nv_bfloat162 sc2 = reinterpret_cast<const nv_bfloat162&>(sc2_u);
    const nv_bfloat162 neg_magic2 = __float2bfloat162_rn(-192.0f);
    const uint32_t b01 = __byte_perm(q0, 0x43434343u, 0x4140u);
    const uint32_t b23 = __byte_perm(q0, 0x43434343u, 0x4342u);
    const uint32_t b45 = __byte_perm(q1, 0x43434343u, 0x4140u);
    const uint32_t b67 = __byte_perm(q1, 0x43434343u, 0x4342u);
    acc[0] = __hfma2(__hadd2(reinterpret_cast<const nv_bfloat162&>(b01), neg_magic2), sc2, acc[0]);
    acc[1] = __hfma2(__hadd2(reinterpret_cast<const nv_bfloat162&>(b23), neg_magic2), sc2, acc[1]);
    acc[2] = __hfma2(__hadd2(reinterpret_cast<const nv_bfloat162&>(b45), neg_magic2), sc2, acc[2]);
    acc[3] = __hfma2(__hadd2(reinterpret_cast<const nv_bfloat162&>(b67), neg_magic2), sc2, acc[3]);
}

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

// ============================================================================
// Head Dispatch (WEAVE_V_HEAD): at layer start ALL warps of every CTA
// (W00..W15: 4 dispatch + 4 TMA/MMA/CLC + 8 epilogue) pull the first
// head_end received tokens together; afterwards only the dispatch warps keep
// pulling.  The head is cooperative: the 16 warps of a CTA copy one token
// together (each warp a 512-B slice, one uint4 per lane), DG_HEAD_TOK tokens
// per CTA per round.  WEAVE_V_HEAD = N is the head quota in tokens per SM, so
// the head window is min(N * num_sms, ring capacity, received tokens) and,
// with DG_HEAD_TOK = 1, N is the number of head rounds.  Token t of the window
// belongs to CTA (t / DG_HEAD_TOK) mod num_sms -- a static zero-CAS split, no
// shared cursor.  WEAVE_V_HEAD = 0 (auto) uses DG_HEAD16_AUTO_QUOTA = one
// round (148 tokens on SM103).
// The head path is compiled out (falls back to dispatch-only pulling) on
// pull variants it is not implemented for: fine-grained group readiness
// (WEAVE_READY_GROUP_SIZE), warp-split modes, per-expert BM ladder
// (WEAVE_ADAPT_BM), SM0 ready queue, pull dedup (WEAVE_DEDUP_MODE) and the
// SM0 runtime policy.  DG_TUNE_HEAD16_OFF=1 is an internal ablation switch
// (never a public knob) that compiles the head path out.
// ============================================================================
#ifndef DG_HEAD16_MAX_VEC
#define DG_HEAD16_MAX_VEC 8u
#endif
#ifndef WEAVE_V_HEAD
#define WEAVE_V_HEAD 0u
#endif
// Rows per epilogue warpgroup of one D7 SMEM handoff slot (see SharedStorage).
#ifndef DG_D7_RING_ROWS
#  if (DG_D7_HANDOFF_MODE == 3) && !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
#    define DG_D7_RING_ROWS 8u
#  else
#    define DG_D7_RING_ROWS 16u
#  endif
#endif
// Number of D7 SMEM handoff slots.  Strict comm_only (mode 3) with the
// register-direct pull affords a 4-deep ring (4 x 2 WG x 16 rows x 256 B =
// 32 KiB, exactly the freed pull bounce buffers), so the epilogue only
// blocks when the comm warps fall 4 sub-tiles behind; every other mode keeps
// the 2-slot ping-pong.
#ifndef DG_D7_RING_SLOTS
#  if (DG_D7_HANDOFF_MODE == 3) && (DG_ISOLATED_D2_REG_PULL_BYTES > 0)
#    define DG_D7_RING_SLOTS 4u
#  else
#    define DG_D7_RING_SLOTS 2u
#  endif
#endif
// Cooperative head granularity: tokens per CTA per head round.  With one
// token per round each of the 16 warps of a CTA moves 8192 B / 16 = 512 B of
// that token, i.e. exactly one 16-B uint4 per lane, and one round over the
// 148 SMs of SM103 covers the first 148 received tokens.
#ifndef DG_HEAD_TOK
#define DG_HEAD_TOK 1u
#endif
// Auto head quota in tokens per SM (= head rounds when DG_HEAD_TOK == 1):
// one round, i.e. the first kNumSMs received tokens (148 on SM103).
#ifndef DG_HEAD16_AUTO_QUOTA
#define DG_HEAD16_AUTO_QUOTA DG_HEAD_TOK
#endif
#ifndef DG_TUNE_HEAD16_OFF
#define DG_TUNE_HEAD16_OFF 0
#endif
// Non-blocking head helper (V3, default): the head is pulled by the dispatch
// warps W00..W03 plus the epilogue warps W08..W15, which are idle until the
// first L1 accumulator is ready anyway.  The TMA-A / TMA-B / UMMA / CLC warps
// (W04..W07) never join the head nor its CTA rendezvous (named barrier 14):
// they start CLC + the L1 weight TMA + UMMA immediately and TMA-A waits on the
// per-pool-block arrival counter (l1_full_count) like for any pulled token.
// DG_HEAD_NONBLOCKING = 0 restores the V2 all-16-warp blocking head (internal
// A/B diagnostic only).
#ifndef DG_HEAD_NONBLOCKING
#define DG_HEAD_NONBLOCKING 1
#endif
#if (DG_TUNE_HEAD16_OFF == 0) && (DG_ISOLATED_D1_G == 0) && (DG_ISOLATED_D2_MODE == 0) && \
    (DG_ISOLATED_D2_DISP_REGS == 0) && (DG_ISOLATED_D2_EPI_REGS == 0) && \
    (DG_D3_EXPERT_STEP_MODE == 0) && (DG_ISOLATED_D4_READYQ == 0) && \
    (DG_ISOLATED_D5_DEDUP == 0) && (WEAVE_RUNTIME_POLICY == 0)
// scheduler/weave_knobs.cuh defaults an un-emitted WEAVE_V_HEAD to the auto
// sentinel 0xffffffff.  V3: WEAVE_V_HEAD = 0 / unset turns the head OFF (only
// the dispatch warps W00..W03 pull, W04..W07 start CLC + TMA + UMMA at once),
// WEAVE_V_HEAD = N > 0 runs N non-blocking helper rounds (see below).
#  if ((WEAVE_V_HEAD) == 0) || ((WEAVE_V_HEAD) == 0xffffffffu)
#    define DG_HEAD16_ENABLED 0
#    define DG_P1_HEAD_HELPER_K 0u
#  else
#    define DG_HEAD16_ENABLED 1
#    define DG_P1_HEAD_HELPER_K (WEAVE_V_HEAD)
#  endif
#else
#  define DG_HEAD16_ENABLED 0
#  define DG_P1_HEAD_HELPER_K 0u
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
        // Reference order: all shared-expert descriptors are prefetched too, so the
        // descriptor cache is warm before the first (shared or routed) TMA issue.
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
    DG_STATIC_ASSERT(WEAVE_BLOCK_M_L1 == 0 or WEAVE_BLOCK_M_L1 <= BLOCK_M,
                     "WEAVE_BLOCK_M_L1 exceeds compiled BLOCK_M envelope");
    DG_STATIC_ASSERT(WEAVE_BLOCK_M_L2 == 0 or WEAVE_BLOCK_M_L2 <= BLOCK_M,
                     "WEAVE_BLOCK_M_L2 exceeds compiled BLOCK_M envelope");
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
    // TMEM epilogue pipeline depth: 1.0 (100), 1.75 overlapping (175, BM=256), 2.0 (200)
    constexpr uint32_t kTmemModeKernel = WEAVE_TMEM_PIPELINE_MODE == 0
        ? (BLOCK_M >= 256u ? 175u : 200u)
        : static_cast<uint32_t>(WEAVE_TMEM_PIPELINE_MODE);
    constexpr uint32_t kAccumOverlapCols =
        (kTmemModeKernel == 175u and BLOCK_M == 256u and kNumNSubtiles == 1u) ? 64u : 0u;
    constexpr uint32_t kAccumStageStride = kNumAccumColsPerStage - kAccumOverlapCols;
    constexpr uint32_t kMaxLegalAccumStages = kAccumOverlapCols > 0u
        ? 2u
        : ((512u - (kNumSFATmemCols + kNumSFBTmemCols)) / kNumAccumColsPerStage);
    constexpr uint32_t kRequestedAccumStages = (kTmemModeKernel == 100u) ? 1u
        : (kAccumOverlapCols > 0u ? 2u : static_cast<uint32_t>(DG_D3_ACCUM_STAGES));
    constexpr uint32_t kNumEpilogueStages =
        (kRequestedAccumStages <= kMaxLegalAccumStages) ? kRequestedAccumStages :
        (kMaxLegalAccumStages >= 1u ? kMaxLegalAccumStages : 1u);
    constexpr uint32_t kNumTMAStoreStages = DG_D3_TMA_STORE_STAGES;
    constexpr uint32_t kNumScheduleStages = DG_D3_SCHED_STAGES;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t L1_OUT_BLOCK_N = 64; // 64B swizzle subtile
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M / 2; // float2

    // Fused SharedLinear2 + INT7 combine.  Routed GEMM2 epilogues
    // write per-128-column INT7 stripes plus in-band per-64 BF16 scale pairs;
    // the tail SharedLinear2 epilogue folds every routed slot into `y` itself.
    static constexpr bool kFusedSharedInt7 =
        (WEAVE_FUSED_SHARED_INT7 != 0) and kHasShared and (kNumSharedExperts == 1u) and
        (kHidden % 256u == 0u) and (BLOCK_N == 128u) and (kNumTopk <= 8u) and
        (STORE_BLOCK_M_L2 == 8u) and (WG_BLOCK_M_RAW % STORE_BLOCK_M_L2 == 0u) and
        (BLOCK_M % STORE_BLOCK_M_L2 == 0u) and (L2_SHAPE_N % BLOCK_N == 0u);
    // Reserved per-warp dispatch slot: the 64K pull granularity stays 2,560 B,
    // the larger 5,120 B slot keeps the fused tail within the same SMEM budget.
    static constexpr uint32_t kSendSlotBytes = kFusedSharedInt7
        ? ((kNumBytesPerPull > 5120u) ? kNumBytesPerPull : 5120u)
        : kNumBytesPerPull;
    static constexpr uint32_t kFsInt7RowBytes = kNumTopk * BLOCK_N;
    static constexpr uint32_t kFsInt7DataBytes = STORE_BLOCK_M_L2 * kFsInt7RowBytes;
    static constexpr uint32_t kFsInt7ScaleRowBytes = 32u;
    static constexpr uint32_t kFsInt7SbBytes = kFsInt7DataBytes + STORE_BLOCK_M_L2 * kFsInt7ScaleRowBytes;
    static constexpr uint32_t kFsInt7NumSb = math::constexpr_ceil_div(kNumMaxTokensPerRank, STORE_BLOCK_M_L2);
    static constexpr uint64_t kFsInt7NBlkStride = static_cast<uint64_t>(kFsInt7NumSb) * kFsInt7SbBytes;
    static constexpr uint32_t kFsNumNBlocks = L2_SHAPE_N / BLOCK_N;
    static constexpr uint64_t kFsInt7TotalBytes = static_cast<uint64_t>(kFsNumNBlocks) * kFsInt7NBlkStride;
    static constexpr uint64_t kFsCombineBufBytes =
        static_cast<uint64_t>(kNumTopk + (kHasShared ? 1u : 0u)) *
        kNumMaxTokensPerRank * kHidden * static_cast<uint64_t>(sizeof(nv_bfloat16));
    static constexpr uint32_t kFsInt7PrefetchBytes = 96u * 1024u;
    static constexpr uint32_t kFsBarrierSleepNs =
        (kNumMaxTokensPerRank >= 65536u) ? 16u : 8u;
    static constexpr uint32_t kFsRingSleepNs =
        (kNumMaxTokensPerRank >= 65536u) ? 128u : 512u;
    DG_STATIC_ASSERT(not kFusedSharedInt7 or kFsInt7TotalBytes <= kFsCombineBufBytes,
                     "Fused-shared INT7 stripes exceed the combine-token buffer");
    DG_STATIC_ASSERT(not kFusedSharedInt7 or (kNumTopk + 1u <= 32u),
                     "Fused-shared top-k + shared must fit in one warp");
    DG_STATIC_ASSERT(not kFusedSharedInt7 or kFsInt7RowBytes % 16u == 0u,
                     "Fused-shared INT7 row stripes must stay 16B aligned");

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
            alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumTmaDispatchWarps][kSendSlotBytes];
    #endif
        };
#else
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
    #if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumTmaDispatchWarps][kSendSlotBytes];
    #elif (DG_D7_HANDOFF_MODE != 0) && (DG_TUNE_D7_SMEM_ALIAS != 0)
        alignas(kSharedMemoryAlignment) uint8_t d7_scatter_smem[8192];
    #endif
#endif
#if (DG_D7_HANDOFF_MODE != 0) && (DG_TUNE_D7_SMEM_ALIAS == 0)
        // Strict comm_only (mode 3) with the TMA pull keeps 8-row slots (8 KiB
        // instead of 16 KiB) so the dedicated ring never costs the GEMM a
        // pipeline stage.  The register-direct pull (WEAVE_REG_PULL_BYTES > 0)
        // has no 4 x 8 KiB SMEM pull bounce buffers, so it affords the wide
        // 16-row slots: half as many handoffs per FC2 tile.
        alignas(kSharedMemoryAlignment) uint8_t d7_dedicated_scatter_smem[
            DG_D7_RING_SLOTS * kNumEpilogueWarpgroups * DG_D7_RING_ROWS * 256u];
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
        Barrier overlap_empty_barriers[2];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
#if DG_D7_HANDOFF_MODE != 0
        D7ScatterJobInfo scatter_jobs[DG_D7_RING_SLOTS];
        Barrier scatter_full_barriers[DG_D7_RING_SLOTS];
        Barrier scatter_empty_barriers[DG_D7_RING_SLOTS];
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
#if WEAVE_RUNTIME_POLICY
        weave::PolicyDecision weave_policy;
        uint32_t weave_policy_ready;
        uint32_t weave_epoch_base;
#endif
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
#ifdef WEAVE_HOST_SMEM_BYTES
    // The host planner (get_pipeline_config_for_mega_moe) sizes the dynamic SMEM
    // launch from the same resolved knobs; it must cover every used byte.
    DG_STATIC_ASSERT(offsetof(SharedStorage, tmem_ptr_in_smem) + sizeof(uint32_t) <= WEAVE_HOST_SMEM_BYTES,
                     "Host SMEM planner under-allocates the kernel SharedStorage");
#endif
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
#if WEAVE_RUNTIME_POLICY
    if (thread_idx == 0u) {
        auto* const control = workspace.get_weave_policy_control_ptr();
        shared_storage.weave_epoch_base =
            *reinterpret_cast<volatile uint32_t*>(&control->published_epoch);
        shared_storage.weave_policy_ready = 0u;
    }
#endif

#if !(DG_ISOLATED_D2_REG_PULL_BYTES > 0)
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumTmaDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));
#endif

    // Tensor memory size (1.75-stage overlapping accumulators share kAccumOverlapCols columns)
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages == 1u
        ? kNumAccumColsPerStage
        : kNumAccumColsPerStage + (kNumEpilogueStages - 1u) * kAccumStageStride;
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
    static constexpr uint32_t kRawDedicatedReduceWarps = static_cast<uint32_t>(WEAVE_REDUCE_WARPS);
    static constexpr bool kSpatialStreamReduce =
        (kRawDedicatedReduceWarps > 0u) and (DG_ISOLATED_D6_G > 0) and
        not kD2SmidMasked and (DG_D7_QUEUE_SCATTER == 0);
    // Scatter-gap Reduce (WEAVE_REDUCE_WARPS > 0 under strict comm_only, D7
    // handoff mode 3): no warp is taken away from the Scatter role.  All four
    // comm warps W00..W03 keep draining the SMEM handoff ring; whenever they
    // find no full slot after their pull has finished (the epilogue is still
    // computing), each warp reduces at most kGapTilesPerGap ready 1-KiB
    // combine tiles of this SM's private tile set (non-blocking readiness
    // check, register-only accumulation), then returns to the ring.
    // DG_REDUCE_GAP=0 restores the dedicated-warp Middle Reduce.
#ifndef DG_REDUCE_GAP
#define DG_REDUCE_GAP 1
#endif
    static constexpr bool kGapReduce =
        kSpatialStreamReduce and (DG_D7_HANDOFF_MODE == 3) and (DG_REDUCE_GAP != 0);
    static constexpr uint32_t kGapTilesPerGap = kRawDedicatedReduceWarps >= 2u ? 2u : 1u;
    static constexpr uint32_t kDedicatedReduceWarps = (!kSpatialStreamReduce or kGapReduce) ? 0u :
        ((DG_D7_HANDOFF_MODE != 0)
            ? (kRawDedicatedReduceWarps >= kNumDispatchWarps ? (kNumDispatchWarps / 2u) : kRawDedicatedReduceWarps)
            : (kRawDedicatedReduceWarps >= kNumDispatchWarps ? kNumDispatchWarps : kRawDedicatedReduceWarps));
    static constexpr uint32_t kFirstDedicatedReduceWarp = kNumDispatchWarps - kDedicatedReduceWarps;
    // Spatial Middle + tail Reduce control words (CTA-private, zero global
    // atomics): [0] = claim cursor over this SM's private tile set
    // P(sm) = { sm + k * kNumSMs }, [1] = tail-go flag.  The spatial path never
    // uses the epilogue combine mbarriers, so both words alias the first one,
    // which is then never initialised as an mbarrier.
    [[maybe_unused]] uint32_t* const spatial_reduce_ctrl =
        reinterpret_cast<uint32_t*>(shared_storage.combine_barriers);
    // All-16-warp tail Combine on the RW=0 path (WEAVE_COMBINE_REDUCE_WARPS=16,
    // the default whenever WEAVE_REDUCE_WARPS=0): W00..W15 drain the tail
    // through the same CTA-private cursor (spatial_reduce_ctrl[0]); the
    // epilogue-only combine mbarriers are never used on this path.
    static constexpr bool kAll16Tail =
        (not kSpatialStreamReduce) and (DG_ISOLATED_D6_REDUCE_WARPS == 16) and not kD2SmidMasked;
#ifndef DG_ALL16_DEFER_CLEAN_WAIT
#define DG_ALL16_DEFER_CLEAN_WAIT 1
#endif
#ifndef DG_ALL16_SUB_BYTES
#define DG_ALL16_SUB_BYTES 0
#endif
    // The cleanup barrier is the launch's last NVLink barrier only when no
    // dedup-2 end barrier follows it.
    static constexpr bool kAll16DeferCleanWait =
        kAll16Tail and (DG_ALL16_DEFER_CLEAN_WAIT != 0) and (DG_ISOLATED_D5_DEDUP != 2);
    DG_STATIC_ASSERT(sizeof(SharedStorage::combine_barriers) >= 2u * sizeof(uint32_t),
                     "Spatial Reduce control words must fit in the combine barrier slots");

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
            for (uint32_t i = 0; i < 2; ++ i) {
                // One overlap-owning warpgroup (128 threads) per CTA arrives at the
                // leader CTA's barrier, for both CTAs of the 2-CTA cluster.
                shared_storage.overlap_empty_barriers[i].init(2u * 128u);
            }
            if constexpr (kSpatialStreamReduce or kAll16Tail) {
                // Cursor and tail-go flag start at 0 every launch (fresh SMEM).
                // The CTA fence orders the stores before the cluster arrive.
                ptx::st_shared(spatial_reduce_ctrl + 0, 0u);
                ptx::st_shared(spatial_reduce_ctrl + 1, 0u);
                __threadfence_block();
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                    shared_storage.combine_barriers[i].init(1);
            }
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
            const uint32_t d7_init_disp_t =
                kSpatialStreamReduce ? (kFirstDedicatedReduceWarp * 32u) : get_active_dispatch_threads();
            #pragma unroll
            for (uint32_t i = 0; i < DG_D7_RING_SLOTS; ++ i) {
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
    // Per-phase / per-expert GEMM sub-tiling (WEAVE_BLOCK_M_L1/L2, WEAVE_ADAPT_BM):
    // a task covers rows [row_offset, row_offset + valid_m) of its BLOCK_M pool block,
    // and readiness counters are credited in 32-row units (kCreditsPerPoolBlock per block).
    using scheduler_type = decltype(scheduler);
    static constexpr bool kSubTileActive = scheduler_type::kSubTileActive;
    static constexpr bool kSubTileL2 = scheduler_type::kSubTileL2;
    static constexpr uint32_t kCreditsPerPoolBlock = scheduler_type::kCreditsPerPoolBlock;

    // Fine-grained L1 -> L2 readiness (reference `sched::L2KBlockDependency`):
    // every L1 N block of a pool block flips one bit of its ring slot's 32-bit
    // readiness word, and the L2 TMA-A producer waits per K block only for the
    // `kNumL1BlockNsPerL2KBlock` L1 N blocks that produce that K slice, instead
    // of waiting for all L1 N blocks before issuing the first L2 K block.
    // Generations alternate the expected word between 0 and the full mask, so no
    // per-generation reset is needed (the end-of-kernel cleanup zeroes it).
    static constexpr uint32_t kNumL1BlockNsForDep = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr bool kL2KBlockMaskDep =
        (BLOCK_N / LAYOUT_AD_M) == 1 and not kSubTileActive and kCreditsPerPoolBlock == 1 and
        kNumL1BlockNsForDep <= 32 and BLOCK_K % (BLOCK_N / 2) == 0 and
        L2_SHAPE_K == kNumL1BlockNsForDep * (BLOCK_N / 2);
    static constexpr uint32_t kL2KBlockFullMask =
        kNumL1BlockNsForDep >= 32 ? 0xffffffffu : ((1u << kNumL1BlockNsForDep) - 1u);

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
    // CTA-wide all-warp Reduce rendezvous.  On the spatial path it is the
    // tail-start barrier of every warp except the dedicated Middle Reduce warps:
    // W04 TMA-A / W05 TMA-B / W06 UMMA / W07 CLC, the epilogue warps and any
    // dispatch warp without a Middle role.  W04/W05 drain the scheduler (which
    // only publishes after Opening) and the epilogue warps arrive only once the
    // GEMM is complete, so this one barrier both frees the reusable SMEM and
    // gates W06/W07 of a non-leader CTA (no GEMM work) behind the Opening
    // d6_epoch bump.  The Middle warps join through the tail-go flag instead,
    // so an in-flight Middle tile never holds the tail back.
    constexpr uint32_t kAllWarpReduceBarrierIdx = 7;
    DG_STATIC_ASSERT(kAllWarpReduceBarrierIdx >= kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups,
                     "Invalid all-warp Reduce barrier index");
    [[maybe_unused]] constexpr uint32_t kTailStartThreads = kNumThreads - kDedicatedReduceWarps * 32u;
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
        (STORE_BLOCK_M % 32u == 0 and DG_D7_RING_SLOTS * kNumEpilogueWarpgroups * 32u * 256u <= kD7DeadDispatchBytes) ? 32u : 0u;
    constexpr uint32_t kD7FitChunkM16 =
        (STORE_BLOCK_M % 16u == 0 and DG_D7_RING_SLOTS * kNumEpilogueWarpgroups * 16u * 256u <= kD7DeadDispatchBytes) ? 16u : 0u;
    constexpr uint32_t kD7FitChunkM8 =
        (STORE_BLOCK_M % 8u == 0 and DG_D7_RING_SLOTS * kNumEpilogueWarpgroups * 8u * 256u <= kD7DeadDispatchBytes) ? 8u : 0u;
    constexpr bool kD7UseDualDeadSlots = (kD7FitChunkM32 > 0u or kD7FitChunkM16 > 0u or kD7FitChunkM8 > 0u);
    constexpr uint32_t kD7DualWgChunkM =
        kD7FitChunkM32 ? 32u : (kD7FitChunkM16 ? 16u : 8u);
    constexpr bool kD7AllowDynamicWg0Full32 =
        (kNumEpilogueWarpgroups == 2u and STORE_BLOCK_M == 32u and kD7DualWgChunkM == 16u and
         DG_D7_RING_SLOTS * 32u * 256u <= kD7DeadDispatchBytes);
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
#if DG_D7_HANDOFF_MODE == 3
    // Strict comm_only: the ring must not alias dispatch SMEM (it is drained
    // while the pull is still running) and every sub-tile must fit a slot.
    DG_STATIC_ASSERT(DG_TUNE_D7_SMEM_ALIAS == 0, "strict comm_only handoff needs dedicated scatter SMEM");
    DG_STATIC_ASSERT(kD7UseDualDeadSlots, "strict comm_only handoff needs a fitting slot size");
    DG_STATIC_ASSERT(not kD2SmidMasked, "strict comm_only handoff needs uniform dispatch warps");
#endif
#endif
#endif

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;
    constexpr uint32_t kFusedSharedInt7BarrierTag = 4;

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
    // Queue scatter is an epilogue-local-dedup path with a single dispatch-side
    // NVLink writer pool.  It is only legal for 128-column epilogue subtiles.
    static constexpr bool kD7QueueScatterActive =
        (DG_D7_QUEUE_SCATTER != 0) and (LAYOUT_AD_M == 128u);
    // HBM-queue dedup (D7 mode 3) composes with the fine-grained D6 combine: the
    // queue consumers publish one D6 ready flag per delivered (token, 128-column
    // subtile, leader slot) after their NVLink stores, the epilogue publishes
    // nothing, and the combine waits on the subtile flags of the leader slots.
    static constexpr bool kD7QueueD6Active = kD7QueueScatterActive and (kD6GroupTiles > 0u);
    static constexpr bool kD7LocalDedupActive =
        ((DG_D7_LOCAL_DEDUP != 0) or (WEAVE_LOCAL_SCATTER_DEDUP != 0)) and
        ((kD6GroupTiles == 0u) or kD7QueueScatterActive);
    DG_STATIC_ASSERT(WEAVE_LOCAL_SCATTER_DEDUP == 0 or DG_D7_QUEUE_SCATTER != 0 or WEAVE_SCATTER_OWNER == 0,
                     "Comm-owned WEAVE_LOCAL_SCATTER_DEDUP requires the HBM scatter-queue path");
    // Upstream per-peer combine readiness (default RW=0 SMEM combine only): one
    // local grid sync, SM 0 tags each peer's `combine_ready_grid_idx[rank]` with
    // that peer's grid index, and the dispatch warps are released into the
    // Workspace Cleanup right away; the cross-rank wait moves into the combine
    // loop, where each warp polls only the ranks owning its token's experts.
    // The cleanup -> final NVLink barrier chain therefore overlaps the peer
    // wait instead of starting after it.  D7 executors (handoff / local dedup /
    // queue) keep the split-phase barrier: their remote rows are not all
    // epilogue-issued before the epilogue grid sync.
    static constexpr bool kPerPeerCombineReady =
        (not kSpatialStreamReduce) and (kD6GroupTiles == 0u) and
        (DG_D7_HANDOFF_MODE == 0) and (DG_D7_QUEUE_SCATTER == 0) and (not kD7LocalDedupActive) and
        // the D1-only dispatch Opening has no grid-index push
        not ((DG_ISOLATED_D1_G > 0) and (DG_ISOLATED_D5_DEDUP != 2)) and
        (kNumRanks <= layout::Workspace::kNumMaxCombineReadyRanks);
    // Epilogue-local dedup without the queue (D7 mode 2) publishes merged rows only
    // from the last contributor, which the per-pool-block D6 publisher cannot track.
    DG_STATIC_ASSERT(kD6GroupTiles == 0u or kD7QueueScatterActive or
                     ((DG_D7_LOCAL_DEDUP == 0) and (WEAVE_LOCAL_SCATTER_DEDUP == 0)),
                     "Epilogue-local scatter dedup (D7 mode 2) cannot be combined with WEAVE_COMBINE_SYNC_CHUNK");
    // Sub-tiled GEMM2 tasks would interleave the per-pool-block D6 N-tile counters.
    DG_STATIC_ASSERT(not kSubTileL2 or kD6GroupTiles <= 1u,
                     "WEAVE_COMBINE_SYNC_CHUNK > 16 cannot be combined with sub-tiled GEMM2 tasks");
    // Width (in combine-row bytes) covered by one D6 ready flag.
    static constexpr uint32_t kD6FlagBytes = kD7QueueD6Active
        ? (128u * static_cast<uint32_t>(sizeof(nv_bfloat16)))
        : (kD6SafeDiv * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)));
    // Strict binary ownership: epilogue_only never hands rows to the
    // communication warps, and comm_only never leaves remote stores in the
    // epilogue (SMEM handoff or HBM dedup queue must be selected).
    static constexpr bool kScatterCommOwned = (WEAVE_SCATTER_OWNER == 1);
    static constexpr bool kD7CommExecutor =
        (DG_TUNE_D7_SCATTER_MODE == 1) or (DG_TUNE_D7_SCATTER_MODE == 3);
    DG_STATIC_ASSERT(kScatterCommOwned == kD7CommExecutor,
                     "WEAVE_SCATTER_OWNER must match the executor: epilogue_only = modes 0/2, "
                     "comm_only = SMEM handoff (1) or HBM dedup queue (3)");
    DG_STATIC_ASSERT(WEAVE_SCATTER_STAGING != 2 or DG_D7_QUEUE_SCATTER != 0,
                     "WEAVE_SCATTER_STAGING=hbm_dedup selects the HBM scatter-queue path");
    static constexpr uint32_t kD7QueueActiveSms =
        (DG_D7_ACTIVE_SMS <= 0 or static_cast<uint32_t>(DG_D7_ACTIVE_SMS) >= kNumSMs)
            ? kNumSMs : static_cast<uint32_t>(DG_D7_ACTIVE_SMS);
    static constexpr uint32_t kD7QueueReserveBatch = 64u;
    // A group only has to tile the N-tile count evenly; it need not be a power
    // of two.  kD6GroupTiles == L2_SHAPE_N / BLOCK_N is the full-row group: one
    // readiness flag (and one sys fence) per pool M-block instead of per column
    // group.
    DG_STATIC_ASSERT(kD6GroupTiles == 0u or
                     ((L2_SHAPE_N / BLOCK_N) % kD6SafeDiv == 0u),
                     "Invalid D6 group tile count");

    // Zero-spill register allocation across all 7 axes (strictly <= 64,512 total per SM)
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    static constexpr bool kD2AnyRegPull = kD2AllRegPull or kD2SplitHybridPull;
    static constexpr bool kD2Is20Warp = (kNumDispatchWarps == 8u and kNumEpilogueWarps == 8u);
    // The host JIT always mirrors its heuristic store block into
    // WEAVE_STORE_BLOCK_M (== T_STORE_BLOCK_M), so a store-M "axis" only counts
    // when it actually overrides the template value.  Treating the mirror as an
    // active axis used to move the default build off the upstream 48/40/208
    // dispatch/non-epilogue/epilogue register split (to 56/40/200).
    static constexpr bool kStoreMAxisActive =
        (DG_TUNE_D3_STORE_M > 0) and (static_cast<uint32_t>(DG_TUNE_D3_STORE_M) != T_STORE_BLOCK_M);
    static constexpr bool kAnyAxisActive =
        (DG_ISOLATED_D1_G > 0) or kD2Active or kD2Enable8D or kSpatialStreamReduce or
        (DG_TUNE_D3_BM > 0) or (DG_TUNE_D3_BN > 0) or (DG_TUNE_D3_BK > 0) or
        kStoreMAxisActive or (DG_D3_DYN_EPI > 0) or (DG_D3_DYN_TMA_A > 0) or
        (DG_D3_OVERLAP_EXPERT_SMEM > 0) or DG_ISOLATED_D4_LUT_ACTIVE or
        (DG_ISOLATED_D5_DEDUP > 0) or (DG_ISOLATED_D6_G > 0) or
        (DG_D7_HANDOFF_MODE != 0) or (DG_D7_LOCAL_DEDUP != 0);

    static constexpr bool kSchedNeedsMoreRegs = DG_ISOLATED_D4_LUT_ACTIVE or (DG_D3_EXPERT_STEP_MODE > 0) or kSubTileActive;
    // The CTA-wide tail Reduce (spatial path or the legacy 16-warp SMEM
    // combine) also executes on the MMA non-epilogue warps, whose 40-register
    // slice otherwise spills its TMA bookkeeping.  One extra 8-register
    // granule keeps it spill-free on the non-20-warp layouts; kMaxRemainingEpi
    // still keeps the 64512-register budget satisfied (the epilogue target has
    // slack there).  The tight 20-warp layout stays untouched.
    static constexpr bool kCtaWideTailReduce =
        kSpatialStreamReduce or (kD6ReduceWarps == 16u and not kD2SmidMasked);
    static constexpr uint32_t kTailReduceRegBump =
        (kCtaWideTailReduce and not kD2Is20Warp) ? 8u : 0u;
#if DG_MEGA_TILE_PROF
    static constexpr uint32_t kAutoNonEpiRegs = kD2Is20Warp
        ? (kUseMoreEpilogueRegisters ? 48u : 56u)
        : ((kUseMoreEpilogueRegisters ? 56u : 88u) + kTailReduceRegBump);
    static constexpr uint32_t kAutoDispRegs = kD2Is20Warp
        ? (kUseMoreEpilogueRegisters ? (kSpatialStreamReduce ? 72u : 64u) : 72u)
        : (kUseMoreEpilogueRegisters ? (kSpatialStreamReduce ? 72u : 64u) : 88u);
#else
    static constexpr uint32_t kAutoNonEpiRegs = !kAnyAxisActive
        ? ((kUseMoreEpilogueRegisters ? 40u : 88u) + kTailReduceRegBump)
        : (kD2Is20Warp
            ? (kUseMoreEpilogueRegisters ? (kSchedNeedsMoreRegs ? 48u : 40u) : (kD2AnyRegPull ? 40u : 56u))
            : ((kUseMoreEpilogueRegisters ? (kSchedNeedsMoreRegs ? 56u : 40u) : 88u) + kTailReduceRegBump));

    static constexpr uint32_t kAutoDispRegs = !kAnyAxisActive
        ? (kUseMoreEpilogueRegisters ? 48u : 96u)
        : (kD2Is20Warp
            ? (!kUseMoreEpilogueRegisters ? ((kD2AnyRegPull or kSpatialStreamReduce) ? 72u : 64u) :
               (DG_ISOLATED_D4_LUT_ACTIVE ? 64u : ((kD2AnyRegPull or kSpatialStreamReduce) ? 72u : 56u)))
            : (!kUseMoreEpilogueRegisters ? 88u :
               (kNumEpilogueWarps == 4u ? ((kD2AnyRegPull or kSpatialStreamReduce or DG_ISOLATED_D1_G > 0 or DG_ISOLATED_D5_DEDUP > 0 or DG_ISOLATED_D7_SCATTER != 0) ? 80u : 64u) :
                ((kD2AnyRegPull or kSpatialStreamReduce or DG_ISOLATED_D1_G > 0 or DG_ISOLATED_D5_DEDUP > 0) ? 72u : 56u))));
#endif

    static constexpr uint32_t kMaxRemainingEpiRegs =
        ((64512u - kAutoDispRegs * kNumDispatchThreads - kAutoNonEpiRegs * kNumNonEpilogueThreads) / kNumEpilogueThreads) & ~7u;
    static constexpr uint32_t kTargetEpiRegs = !kAnyAxisActive
        ? (kUseMoreEpilogueRegisters ? 208u : 160u)
        : (kD2Is20Warp ? 168u : (kUseMoreEpilogueRegisters ? 208u : 160u));
    static constexpr uint32_t kAutoEpiRegs = (kTargetEpiRegs <= kMaxRemainingEpiRegs) ? kTargetEpiRegs : kMaxRemainingEpiRegs;

#if DG_P1_HEAD_HELPER_K > 0
    DG_STATIC_ASSERT(not kD2SmidMasked, "Head Dispatch is compiled out for warp-split modes");
#endif
    constexpr uint32_t kRegBalDisp = (WEAVE_REG_BALANCE != 0 and not kUseMoreEpilogueRegisters) ? 64u : kAutoDispRegs;
    constexpr uint32_t kRegBalNonEpi = (WEAVE_REG_BALANCE != 0 and not kUseMoreEpilogueRegisters) ? 56u : kAutoNonEpiRegs;
    constexpr uint32_t kRegBalEpi = (WEAVE_REG_BALANCE != 0 and not kUseMoreEpilogueRegisters) ? 192u : kAutoEpiRegs;
    constexpr uint32_t kNumDispatchRegisters = (DG_ISOLATED_D2_DISP_REGS > 0u) ? DG_ISOLATED_D2_DISP_REGS : kRegBalDisp;
    constexpr uint32_t kNumNonEpilogueRegisters = kRegBalNonEpi;
    constexpr uint32_t kNumEpilogueRegisters = (DG_ISOLATED_D2_EPI_REGS > 0u) ? DG_ISOLATED_D2_EPI_REGS : kRegBalEpi;
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

#ifndef DG_L2_PREFETCH_EXPERTS
#define DG_L2_PREFETCH_EXPERTS 0
#endif
#ifndef DG_L2_PREFETCH_DIST
#define DG_L2_PREFETCH_DIST 0
#endif
#if DG_L2_PREFETCH_EXPERTS > 0
    // Opening HBM warm-up: while the Opening / Head Dispatch is NVLink-bound
    // and HBM idles, W05 (TMA-B) of every CTA issues L2 tensor prefetches of
    // the L1 weights + SF of the first DG_L2_PREFETCH_EXPERTS local experts
    // (the scheduler hands out L1 tiles in pool-block order, so the first
    // wave lands on the first experts).  Tile t (expert-major) is prefetched
    // by CTA t % kNumSMs: a static split, no synchronization, no SMEM.
    if (warp_idx == kNumDispatchWarps + 1u and lane_idx == 0u) {
        constexpr uint32_t kPfNB = L1_SHAPE_N / BLOCK_N;
        constexpr uint32_t kPfKB = L1_SHAPE_K / BLOCK_K;
        constexpr uint32_t kPfExperts = static_cast<uint32_t>(DG_L2_PREFETCH_EXPERTS) < kNumExpertsPerRank ?
            static_cast<uint32_t>(DG_L2_PREFETCH_EXPERTS) : kNumExpertsPerRank;
        constexpr uint32_t kPfTiles = kPfExperts * kPfNB * kPfKB;
        constexpr uint32_t kPfShapeSfbK = math::constexpr_ceil_div(L1_SHAPE_K, kGranK * 4u);
        #pragma unroll 1
        for (uint32_t t = sm_idx; t < kPfTiles; t += kNumSMs) {
            const uint32_t e = t / (kPfNB * kPfKB);
            const uint32_t r = t % (kPfNB * kPfKB);
            const uint32_t nb = r % kPfNB, kb = r / kPfNB;
            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                         :: "l"(reinterpret_cast<uint64_t>(&tensor_map_l1_weights)),
                            "r"(kb * BLOCK_K), "r"(e * L1_SHAPE_N + nb * BLOCK_N) : "memory");
            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                         :: "l"(reinterpret_cast<uint64_t>(&tensor_map_l1_weights_sf)),
                            "r"(nb * BLOCK_N), "r"(e * kPfShapeSfbK + kb * (BLOCK_K / 128u)) : "memory");
        }
    }
#endif

#if DG_P1_HEAD_HELPER_K > 0
    // Every CTA derives the same head window from the received-token
    // histogram, which is complete once the dispatch warps leave the
    // post-dispatch NVLink barrier.  The CTA rendezvous below (named barrier
    // 14, all kNumThreads) hands that ordering to the TMA/MMA and epilogue
    // warps; the gpu-scope fence then makes their histogram loads observe it.
    constexpr uint32_t kNumHeadThreads = DG_HEAD_NONBLOCKING ? (kNumThreads - kNumNonEpilogueThreads) : kNumThreads;
    constexpr uint32_t kNumHeadWarpsPerCTA = kNumHeadThreads / 32u;
    // Head warp index: W00..W03 -> 0..3, W08..W15 -> 4..11 (non-blocking), or
    // the plain warp index for the all-16-warp blocking head.
    const uint32_t head_warp_idx = (DG_HEAD_NONBLOCKING and warp_idx >= kNumDispatchWarps)
        ? warp_idx - kNumMMANonEpilogueWarps : warp_idx;

    const auto p1_count_head_tokens = [&]() {
        // WEAVE_V_HEAD (or the auto quota) = head tokens per SM.
        constexpr uint64_t kHelperCapacity = static_cast<uint64_t>(DG_P1_HEAD_HELPER_K) * kNumSMs;
        constexpr uint32_t kRequested = DG_P1_HEAD_TOKENS > 0u ? DG_P1_HEAD_TOKENS :
            static_cast<uint32_t>(kHelperCapacity < 0xffffffffull ? kHelperCapacity : 0xffffffffull);
        uint32_t tokens = 0u, used_blocks = 0u;
        for (uint32_t base = 0u; base < kNumExpertsPerRank and
             used_blocks < kNumRingBlocks and tokens < kRequested; base += 32u) {
            const uint32_t expert = base + lane_idx;
            const uint32_t count = expert < kNumExpertsPerRank ?
                ptx::ld_acq(reinterpret_cast<const uint32_t*>(
                    workspace.get_expert_recv_count_sum_ptr(expert))) : 0u;  // low word = count
            const uint32_t blocks = math::ceil_div(count, BLOCK_M);
            const uint32_t end = math::warp_inclusive_sum(blocks, lane_idx);
            const uint32_t start = used_blocks + end - blocks;
            // The head never waits on ring reuse (that depends on GEMM, which
            // the helping warps have yet to start): stop at the first window.
            const uint32_t available = start < kNumRingBlocks ? (kNumRingBlocks - start) * BLOCK_M : 0u;
            tokens += __reduce_add_sync(0xffffffffu, cute::min(count, available));
            used_blocks += __shfl_sync(0xffffffffu, end, 31);
        }
        return cute::min(tokens, kRequested);
    };

    // All warps rendezvous after dispatch finishes publishing cross-rank metadata.
    const auto p1_wait_for_metadata = [&]() {
        asm volatile("barrier.sync 14, %0;" :: "n"(kNumHeadThreads) : "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        return p1_count_head_tokens();
    };

    const auto run_p1_head_pull = [&]<uint32_t kRegisters, bool kDispatch>(const uint32_t head_end) {
        // Cooperative head: every warp of this CTA (W00..W15) copies its own
        // 1/kNumHeadWarpsPerCTA slice of the SAME kHeadTok tokens per round,
        // so a token costs one NVLink round trip per warp (each lane moves at
        // most kSliceVecPerLane uint4 per token) instead of a whole-row copy
        // through one warp's registers.  Token t of the head window belongs to
        // CTA (t / kHeadTok) % kNumSMs (static, zero CAS): round r of CTA s
        // covers tokens [(r * kNumSMs + s) * kHeadTok, ... + kHeadTok).
        constexpr uint32_t kHeadTok = static_cast<uint32_t>(DG_HEAD_TOK);
        DG_STATIC_ASSERT(kHeadTok >= 1u and kHeadTok <= 4u, "DG_HEAD_TOK must be in [1, 4]");
        constexpr uint32_t kRowVec = kHidden / static_cast<uint32_t>(sizeof(uint4));
        constexpr uint32_t kSliceVec = math::constexpr_ceil_div(kRowVec, kNumHeadWarpsPerCTA);
        constexpr uint32_t kSliceVecPerLane = math::constexpr_ceil_div(kSliceVec, 32u);
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        constexpr uint32_t kNumSFUint32 = kHidden / 128u;
        constexpr uint32_t kSFWarp = kNumHeadWarpsPerCTA - 1u;      // SF copy
        constexpr uint32_t kMetaWarp = kNumHeadWarpsPerCTA - 2u;    // weight + source metadata
        DG_STATIC_ASSERT(kRegisters >= 40u, "Invalid head register budget");
        DG_STATIC_ASSERT(kHidden % 128u == 0u, "Invalid P1 activation width");
        DG_STATIC_ASSERT(kNumHeadWarpsPerCTA >= 2u, "Cooperative head needs >= 2 warps");
        (void) kDispatch;

        uint32_t expert_idx = 0u, expert_start = 0u, expert_end = 0u, pool_block_base = 0u;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        const uint32_t slice_begin = head_warp_idx * kSliceVec;
        constexpr uint32_t kRoundStride = kNumSMs * kHeadTok;
        #pragma unroll 1
        for (uint32_t base = sm_idx * kHeadTok; base < head_end; base += kRoundStride) {
#if DG_MEGA_TILE_PROF
            const uint64_t prof_pull_s = tile_prof::read_gtimer();
#endif
            const uint4* src_row[kHeadTok];
            uint4* dst_row[kHeadTok];
            uint32_t pool_token[kHeadTok], full_delta[kHeadTok], src_rank_v[kHeadTok], src_tt[kHeadTok];
            uint32_t src_token_v[kHeadTok], src_topk_v[kHeadTok], sf_ring_tok[kHeadTok];
            #pragma unroll
            for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                const uint32_t token = base + j;
                pool_token[j] = 0xffffffffu;
                full_delta[j] = src_rank_v[j] = src_tt[j] = src_token_v[j] = src_topk_v[j] = sf_ring_tok[j] = 0u;
                src_row[j] = nullptr;
                dst_row[j] = nullptr;
                if (token >= head_end)
                    continue;
                if (token >= expert_end) {
                    do {
                        pool_block_base += math::ceil_div(expert_end - expert_start, BLOCK_M);
                        expert_start = expert_end;
                        uint32_t count = 0u;
                        if (lane_idx == 0u)
                            count = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(expert_idx));
                        expert_end += __shfl_sync(0xffffffffu, count, 0);
                        ++ expert_idx;
                    } while (token >= expert_end);
                    #pragma unroll
                    for (uint32_t i = 0u; i < kNumRanksPerLane; ++ i) {
                        const uint32_t rank = i * 32u + lane_idx;
                        stored_rank_count[i] = rank < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(rank, expert_idx - 1u)) : 0u;
                    }
                }
                const uint32_t token_idx_in_expert = token - expert_start;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t src_rank, token_idx_in_rank;
                find_round_robin_pull_source(remaining, token_idx_in_expert, src_rank, token_idx_in_rank);
                const uint32_t src_token_topk_raw = *workspace.get_src_token_topk_idx_ptr(
                    expert_idx - 1u, src_rank, token_idx_in_rank);
                const uint32_t src_token_topk_idx = src_token_topk_raw & 0x0003ffffu;
                const uint32_t d7_topk_bm = kD7LocalDedupActive ? ((src_token_topk_raw >> 18) & 0x3ffu) : 0u;
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                pool_token[j] = pool_block_base * BLOCK_M + token_idx_in_expert;
                full_delta[j] = token == expert_end - 1u ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u;
                src_rank_v[j] = src_rank;
                src_tt[j] = src_token_topk_idx;
                src_token_v[j] = src_token_idx;
                src_topk_v[j] = (src_token_topk_idx % kNumTopk) | (d7_topk_bm << 8);
                sf_ring_tok[j] = (pool_token[j] / BLOCK_M) * SF_BLOCK_M +
                                 transform_sf_token_idx(token_idx_in_expert % BLOCK_M);
                src_row[j] = sym_buffer.map(
                    buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint4>(), src_rank);
                dst_row[j] = buffer.l1_token_buffer.get_data_buffer(pool_token[j]).get_base_ptr<uint4>();
            }

            // All loads of the round are in flight before any store (one round trip).
            uint4 values[kHeadTok][kSliceVecPerLane];
            #pragma unroll
            for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                #pragma unroll
                for (uint32_t v = 0u; v < kSliceVecPerLane; ++ v) {
                    const uint32_t off = v * 32u + lane_idx;
                    const uint32_t idx = slice_begin + off;
                    if (pool_token[j] != 0xffffffffu and off < kSliceVec and idx < kRowVec) {
                        auto& value = values[j][v];
                        asm volatile("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                                     : "l"(src_row[j] + idx));
                    }
                }
            }
            float weight_v[kHeadTok];
            #pragma unroll
            for (uint32_t j = 0u; j < kHeadTok; ++ j)
                weight_v[j] = 0.0f;
            if (head_warp_idx == kMetaWarp) {
                #pragma unroll
                for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                    if (pool_token[j] != 0xffffffffu)
                        weight_v[j] = *sym_buffer.map(
                            buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_tt[j], src_rank_v[j]);
                }
            }
            if (head_warp_idx == kSFWarp) {
                auto* local_sf = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
                #pragma unroll
                for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                    if (pool_token[j] == 0xffffffffu)
                        continue;
                    const auto* remote_sf = sym_buffer.map(
                        buffer.input_sf_buffer.get_data_buffer(src_token_v[j]).get_base_ptr<uint32_t>(), src_rank_v[j]);
                    #pragma unroll
                    for (uint32_t i = 0u; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                        const uint32_t k = i * 32u + lane_idx;
                        if (k < kNumSFUint32)
                            local_sf[k * kNumSFRingTokens + sf_ring_tok[j]] = remote_sf[k];
                    }
                }
            }
            #pragma unroll
            for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                #pragma unroll
                for (uint32_t v = 0u; v < kSliceVecPerLane; ++ v) {
                    const uint32_t off = v * 32u + lane_idx;
                    const uint32_t idx = slice_begin + off;
                    if (pool_token[j] != 0xffffffffu and off < kSliceVec and idx < kRowVec) {
                        const auto& value = values[j][v];
                        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                     :: "l"(dst_row[j] + idx),
                                        "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
                    }
                }
            }
            if (head_warp_idx == kMetaWarp and lane_idx == 0u) {
                #pragma unroll
                for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                    if (pool_token[j] == 0xffffffffu)
                        continue;
                    *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token[j]).get_base_ptr<float>() = weight_v[j];
                    *workspace.get_token_src_metadata_ptr(pool_token[j]) = {src_rank_v[j], src_token_v[j], src_topk_v[j]};
                }
            }

            // Every warp's slice stores precede the single block-ready release:
            // per-thread gpu fence, CTA rendezvous (named barrier 14, all
            // kNumThreads), then one release fetch-add per token (padded on the
            // last token of an expert) by W00.
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            asm volatile("barrier.sync 14, %0;" :: "n"(kNumHeadThreads) : "memory");
            if (warp_idx == 0u and lane_idx == 0u) {
                #pragma unroll
                for (uint32_t j = 0u; j < kHeadTok; ++ j) {
                    if (pool_token[j] != 0xffffffffu)
                        ptx::red_add_rel(workspace.get_l1_full_count_ptr(pool_token[j] / BLOCK_M), full_delta[j]);
                }
            }
#if DG_MEGA_TILE_PROF
            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                              prof_rec_idx, prof_t0_ns, prof_pull_s, tile_prof::read_gtimer(),
                              30u, pool_token[0] / BLOCK_M);
#endif
        }
    };

#if !DG_HEAD_NONBLOCKING
    if (warp_idx >= kNumDispatchWarps and warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        run_p1_head_pull.template operator()<kNumNonEpilogueRegisters, false>(p1_wait_for_metadata());
    }
#endif
#endif

    // Zero-SMEM in-register scoreboard fine-grained streaming Reduce executed by
    // the dedicated Middle Reduce warps (WEAVE_REDUCE_WARPS) while the GEMM
    // still owns every stage of shared memory, followed by the CTA-wide TMA tail
    // Reduce (run_spatial_tail_combine below).
    //
    // Work split (per-SM private, zero global atomics, no CAS): the combine
    // tiles t in [0, num_tokens * 14) -- sub-major, token = t % num_tokens,
    // sub = t / num_tokens -- are statically partitioned over the SMs of this
    // rank.  SM s owns P(s) = { s + k * kNumSMs : 0 <= k < n(s) }.  Inside the
    // CTA the Middle warps and every tail warp claim k from one shared-memory
    // cursor (spatial_reduce_ctrl[0], atom.shared.add): each k < n(s) is
    // returned to exactly one warp, which always processes it (a claim is never
    // burnt), and no SM ever touches another SM's tiles, so the split is free
    // of duplicates, gaps and cross-SM contention by construction.  The Middle
    // stops claiming at cap(s) = n(s) - n(s) / 2 (the same half split as the
    // former global cursor) or as soon as the tail-go flag is up.
    // True when every D6 ready flag covering combine-row bytes
    // [byte_begin, byte_begin + num_bytes) of (token, slot) carries `epoch`.
    // A 1 KiB reduce tile can span several D6 groups (G * BLOCK_N * 2 < 1 KiB)
    // or several queue subtiles, so all covering flags are checked.
    [[maybe_unused]] const auto d6_flags_ready = [&](const uint32_t& token_idx, const uint32_t& slot_idx,
                                                     const uint32_t& byte_begin, const uint32_t& num_bytes,
                                                     const uint16_t& epoch) -> bool {
        const uint32_t first_flag = byte_begin / kD6FlagBytes;
        const uint32_t last_flag = (byte_begin + num_bytes - 1u) / kD6FlagBytes;
        bool ready = true;
        for (uint32_t f = first_flag; f <= last_flag; ++ f)
            ready = ready and (ptx::ld_acq_sys(buffer.get_d6_ready_epoch_ptr(token_idx, f) + slot_idx) == epoch);
        return ready;
    };

    [[maybe_unused]] const auto get_num_private_reduce_tiles = [&](const uint32_t& total_tiles) -> uint32_t {
        return total_tiles > sm_idx ? (total_tiles - sm_idx - 1u) / kNumSMs + 1u : 0u;
    };

    [[maybe_unused]] const auto run_spatial_streaming_reduce = [&]<uint32_t kInflight>(const uint16_t cur_epoch) {
        if constexpr (kSpatialStreamReduce) {
            constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
            constexpr uint32_t kSubBytes = 1024u; // 2 x uint4 per lane (512 bf16)
            constexpr uint32_t kNumSubs = kNumHiddenBytes / kSubBytes;
            DG_STATIC_ASSERT(kNumHiddenBytes % kSubBytes == 0u,
                             "Hidden bytes must be divisible by 1024B for WEAVE_REDUCE_WARPS");
            DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u,
                             "Top-k + shared must fit in a single warp");
            // Middle claim cap: the register path stops claiming at half of this
            // SM's private tiles, so the 16-warp TMA tail always keeps the other
            // half (a fully greedy Middle measured as a net loss in the ABBA runs).
            const uint32_t num_private_tiles = get_num_private_reduce_tiles(num_tokens * kNumSubs);
            const uint32_t middle_claim_cap = num_private_tiles - num_private_tiles / 2u;
            if (middle_claim_cap == 0u)
                return;

            uint32_t slot_tile[kInflight];
            uint32_t slot_pending[kInflight];
            float2 slot_acc[kInflight][8];
            uint32_t next_exhausted = 0u;
            uint32_t active_slots = 0u;

            const auto claim_into_slot = [&](const uint32_t k) {
                slot_tile[k] = 0xffffffffu;
                slot_pending[k] = 0u;
                if (next_exhausted != 0u)
                    return;
                // Lane 0 claims the next private index from the CTA cursor,
                // unless the tail has started: the rest then belongs to the tail.
                uint32_t claimed = 0xffffffffu;
                if (lane_idx == 0u and ptx::ld_acq_shared(spatial_reduce_ctrl + 1) == 0u)
                    claimed = ptx::atomic_add_shared(spatial_reduce_ctrl + 0, 1u);
                claimed = __shfl_sync(0xffffffffu, claimed, 0);
                if (claimed >= num_private_tiles) {
                    next_exhausted = 1u;
                    return;
                }
                // A returned index is always processed, even one past the cap
                // when several Middle warps race for the last Middle indices.
                if (claimed + 1u >= middle_claim_cap)
                    next_exhausted = 1u;
                const uint32_t t = sm_idx + claimed * kNumSMs;
                slot_tile[k] = t;
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++e)
                    slot_acc[k][e] = {0.0f, 0.0f};
                const uint32_t tok_idx = t % num_tokens;
                const int topk_val = lane_idx < kNumTopk
                    ? static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + tok_idx * kNumTopk + lane_idx))
                    : (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                const uint32_t pmask = __ballot_sync(0xffffffffu, topk_val >= 0);
                if (pmask == 0u) {
                    const uint32_t sub_idx = t / num_tokens;
                    auto* dst_u4 = reinterpret_cast<uint4*>(
                        math::advance_ptr<uint8_t>(y, static_cast<uint64_t>(tok_idx) * kNumHiddenBytes + sub_idx * kSubBytes));
                    #pragma unroll
                    for (uint32_t v = 0u; v < 2u; ++v)
                        __stcs(dst_u4 + v * 32u + lane_idx, make_uint4(0u, 0u, 0u, 0u));
                    slot_tile[k] = 0xffffffffu;
                    slot_pending[k] = 0u;
                } else {
                    slot_pending[k] = pmask;
                    ++active_slots;
                }
            };

            #pragma unroll
            for (uint32_t k = 0u; k < kInflight; ++k) {
                slot_tile[k] = 0xffffffffu;
                slot_pending[k] = 0u;
                while (slot_tile[k] == 0xffffffffu and next_exhausted == 0u) {
                    claim_into_slot(k);
                }
            }

            const auto timeout_start = clock64();
            while (active_slots > 0u) {
                bool made_progress = false;
                #pragma unroll
                for (uint32_t k = 0u; k < kInflight; ++k) {
                    uint32_t pmask = slot_pending[k];
                    if (pmask != 0u) {
                        const uint32_t t = slot_tile[k];
                        const uint32_t tok_idx = t % num_tokens;
                        const uint32_t sub_idx = t / num_tokens;
                        const bool lane_ready = ((pmask & (1u << lane_idx)) != 0u) and
                            d6_flags_ready(tok_idx, lane_idx, sub_idx * kSubBytes, kSubBytes, cur_epoch);
                        uint32_t rmask = __ballot_sync(0xffffffffu, lane_ready);
                        // Bit-exact slot order across polls: only the ready prefix
                        // below the lowest pending-but-not-ready slot may be added.
                        // FP32 addition is not associative, so a slot that became
                        // ready early must still wait for every lower slot; with
                        // this rule each tile sums its slots strictly in ascending
                        // order from +0, exactly like the reference combine.
                        const uint32_t waiting_mask = pmask & ~rmask;
                        if (waiting_mask != 0u)
                            rmask &= (waiting_mask & (0u - waiting_mask)) - 1u;
                        if (rmask != 0u) {
                            // bar.warp.sync orders the observing lanes' acquire flag
                            // loads before every lane's source loads below.
                            __syncwarp();
                            made_progress = true;
#if DG_MEGA_TILE_PROF
                            const uint32_t prof_red_s = tile_prof::read_gtimer();
#endif
                            const uint32_t sub_byte_offset = sub_idx * kSubBytes;
                            // Vectorized source pairing: two ready sources issue four
                            // 128-bit loads before the first fused add consumes them,
                            // doubling the per-lane bytes in flight without any SMEM.
                            // s0 < s1 are consecutive in the ready prefix and s0 is
                            // always accumulated first, so the per-element FP32 sum
                            // stays in ascending slot order.
                            while (rmask != 0u) {
                                const uint32_t s0 = static_cast<uint32_t>(__ffs(rmask) - 1);
                                rmask ^= (1u << s0);
                                pmask ^= (1u << s0);
                                uint32_t s1 = 0xffffffffu;
                                if (rmask != 0u) {
                                    s1 = static_cast<uint32_t>(__ffs(rmask) - 1);
                                    rmask ^= (1u << s1);
                                    pmask ^= (1u << s1);
                                }
                                const auto* src0_u4 = reinterpret_cast<const uint4*>(math::advance_ptr<uint8_t>(
                                    buffer.combine_token_buffer.get_rank_buffer(s0)
                                          .get_data_buffer(tok_idx).get_base_ptr(),
                                    sub_byte_offset));
                                // __ldcs lowers to the same ld.global.cs.v4.u32 but is not
                                // volatile, so ptxas may batch all 16 slot loads of a tile
                                // instead of serialising each one behind its own FADDs.
                                const uint4 v00 = __ldcs(src0_u4 + lane_idx);
                                const uint4 v01 = __ldcs(src0_u4 + 32u + lane_idx);
                                uint4 v10 = make_uint4(0u, 0u, 0u, 0u);
                                uint4 v11 = v10;
                                if (s1 != 0xffffffffu) {
                                    const auto* src1_u4 = reinterpret_cast<const uint4*>(math::advance_ptr<uint8_t>(
                                        buffer.combine_token_buffer.get_rank_buffer(s1)
                                              .get_data_buffer(tok_idx).get_base_ptr(),
                                        sub_byte_offset));
                                    v10 = __ldcs(src1_u4 + lane_idx);
                                    v11 = __ldcs(src1_u4 + 32u + lane_idx);
                                }
                                {
                                    const auto* bf16_vals = reinterpret_cast<const nv_bfloat162*>(&v00);
                                    #pragma unroll
                                    for (uint32_t l = 0u; l < 4u; ++l)
                                        ptx::accumulate(slot_acc[k][l], bf16_vals[l]);
                                }
                                {
                                    const auto* bf16_vals = reinterpret_cast<const nv_bfloat162*>(&v01);
                                    #pragma unroll
                                    for (uint32_t l = 0u; l < 4u; ++l)
                                        ptx::accumulate(slot_acc[k][4u + l], bf16_vals[l]);
                                }
                                if (s1 != 0xffffffffu) {
                                    const auto* bf16_vals = reinterpret_cast<const nv_bfloat162*>(&v10);
                                    #pragma unroll
                                    for (uint32_t l = 0u; l < 4u; ++l)
                                        ptx::accumulate(slot_acc[k][l], bf16_vals[l]);
                                    const auto* bf16_vals2 = reinterpret_cast<const nv_bfloat162*>(&v11);
                                    #pragma unroll
                                    for (uint32_t l = 0u; l < 4u; ++l)
                                        ptx::accumulate(slot_acc[k][4u + l], bf16_vals2[l]);
                                }
                            }
                            slot_pending[k] = pmask;
                            if (pmask == 0u) {
                                auto* dst_u4 = reinterpret_cast<uint4*>(math::advance_ptr<uint8_t>(
                                    y, static_cast<uint64_t>(tok_idx) * kNumHiddenBytes + sub_byte_offset));
                                #pragma unroll
                                for (uint32_t v = 0u; v < 2u; ++v) {
                                    uint4 casted;
                                    auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                                    #pragma unroll
                                    for (uint32_t l = 0u; l < 4u; ++l)
                                        casted_bf16[l] = __float22bfloat162_rn(slot_acc[k][v * 4u + l]);
                                    __stcs(dst_u4 + v * 32u + lane_idx, casted);
                                }
                            }
#if DG_MEGA_TILE_PROF
                            tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                              prof_rec_idx, prof_t0_ns, prof_red_s, tile_prof::read_gtimer(),
                                              14u, t);
#endif
                            if (pmask == 0u) {
                                --active_slots;
                                slot_tile[k] = 0xffffffffu;
                                while (slot_tile[k] == 0xffffffffu and next_exhausted == 0u) {
                                    claim_into_slot(k);
                                }
                            }
                        }
                    }
                }
                if (not made_progress) {
                    if (clock64() - timeout_start >= comm::kNumTimeoutCycles) {
                        DG_DEVICE_ASSERT(false and "WEAVE_REDUCE_WARPS spatial streaming reduce timeout");
                    }
                    // Middle warps poll for the whole GEMM, so they back off hard
                    // to keep .sys L2-bypass traffic off the critical path; tail
                    // warps only wait for the last epilogue stores.
                    if constexpr (kInflight > 1u) {
                        asm volatile("nanosleep.u32 256;");
                    } else {
                        asm volatile("nanosleep.u32 32;");
                    }
                }
            }
        }
    };

    // Axis 6 (D6 All-Warp Reduce, spatial tail): once the tail has started
    // (spatial_tail_start below) every warp W00..W15 drains the rest of this
    // SM's private tiles through the reference double-buffered TMA -> SMEM path
    // (1 KiB sub-chunks; one load buffer pair plus two mbarriers and one store
    // buffer per warp), claiming k from the same CTA cursor the Middle register
    // path used.  The GEMM and the epilogue are complete at the tail start, so
    // the reusable shared memory is free and TMA runs at full throughput.
    // Sources are summed in ascending combine-slot order with FP32
    // accumulation and one final bf16 rounding, exactly like the reference
    // combine, so the 10-launch bitwise protocol is preserved.
    [[maybe_unused]] const auto run_spatial_tail_combine = [&]() {
        constexpr uint32_t kAllReduceWarps = kNumThreads / 32u;
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kSubChunkBytes = 1024u;
        constexpr uint32_t kNumSubChunks = kNumHiddenBytes / kSubChunkBytes;
        constexpr uint32_t kVecsPerLane = kSubChunkBytes / (32u * sizeof(uint4));
        constexpr uint32_t kElemsPerVec = sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kSmemAllWarpBytes = kAllReduceWarps * 3u * kSubChunkBytes;
        DG_STATIC_ASSERT(kNumHiddenBytes % kSubChunkBytes == 0u, "Hidden bytes must be divisible by 1024B");
        DG_STATIC_ASSERT(kSmemAllWarpBytes + kAllReduceWarps * 2u * sizeof(Barrier) <= kNumReusableSmemBytes,
                         "All-warp tail Reduce does not fit in the reusable shared memory");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u,
                         "Top-k + shared must fit in a single warp");

        // Read the epoch only after the CTA rendezvous, and with an acquire
        // load: W04..W07 never touch this counter in their own role, so a weak
        // load could be hoisted above the Opening bump.
        const uint16_t all_d6_epoch = static_cast<uint16_t>(ptx::ld_acq(buffer.get_d6_epoch_ptr()));
        // Each warp owns its two load mbarriers (only its own TMA loads complete
        // on them) and initialises them itself: no CTA-wide sync inside the
        // tail, so a Middle warp that joins late needs no rendezvous either.
        auto* all_combine_barriers = reinterpret_cast<Barrier*>(smem_buffer + kSmemAllWarpBytes);
        if (lane_idx < 2u)
            all_combine_barriers[warp_idx * 2u + lane_idx].init(1);
        cutlass::arch::fence_barrier_init();
        __syncwarp();

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
        const uint32_t num_private_tiles = get_num_private_reduce_tiles(num_tokens * kNumSubChunks);
        while (true) {
            uint32_t claimed = 0u;
            if (lane_idx == 0u)
                claimed = ptx::atomic_add_shared(spatial_reduce_ctrl + 0, 1u);
            claimed = __shfl_sync(0xffffffffu, claimed, 0);
            if (claimed >= num_private_tiles)
                break;
            const uint32_t tile_idx = sm_idx + claimed * kNumSMs;
#if DG_MEGA_TILE_PROF
            uint32_t prof_red_s = tile_prof::read_gtimer();
#endif
            // Same sub-major (token, sub) decode as the Middle register path.
            const uint32_t token_idx = tile_idx % num_tokens;
            const uint32_t sub_idx = tile_idx / num_tokens;
            const uint32_t sub_byte_offset = sub_idx * kSubChunkBytes;

            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            bool is_active_combine_slot = (stored_topk_slot_idx >= 0);
            if constexpr (kD7LocalDedupActive) {
                // Queue dedup delivers one merged row per (token, expert rank)
                // into the leader slot.
                const uint32_t exp_rank = (lane_idx < kNumTopk and stored_topk_slot_idx >= 0)
                    ? (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank)
                    : (0xffff0000u | lane_idx);
                const uint32_t same_r_mask = __match_any_sync(0xffffffffu, exp_rank);
                if (lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and
                    (kD7QueueScatterActive or exp_rank != sym_buffer.rank_idx)) {
                    is_active_combine_slot = (lane_idx == static_cast<uint32_t>(__ffs(same_r_mask) - 1));
                }
            }
            uint32_t mask = __ballot_sync(0xffffffff, is_active_combine_slot);

            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1u << slot_idx;
                    if (cute::elect_one_sync()) {
                        if (not d6_flags_ready(token_idx, slot_idx, sub_byte_offset, kSubChunkBytes, all_d6_epoch)) {
                            const auto start_clock = clock64();
                            do {
                                if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                                    DG_DEVICE_ASSERT(false and "D6 All-Warp Reduce readiness timeout");
                                }
                                asm volatile("nanosleep.u32 32;");
                            } while (not d6_flags_ready(token_idx, slot_idx, sub_byte_offset, kSubChunkBytes, all_d6_epoch));
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
    };

    // Spatial tail start: the barrier participants (every warp except the
    // dedicated Middle warps) meet on kAllWarpReduceBarrierIdx, then W04 lane 0
    // raises the tail-go flag with a CTA-scope release.
    [[maybe_unused]] const auto spatial_tail_start = [&]() {
        ptx::sync_aligned(kTailStartThreads, kAllWarpReduceBarrierIdx);
        if (thread_idx == kNumDispatchThreads)
            ptx::st_rel_shared(spatial_reduce_ctrl + 1, 1u);
    };
    // A dedicated Middle warp acquires the tail-go flag (shared-memory poll, no
    // global traffic) before it touches the reusable shared memory.
    [[maybe_unused]] const auto spatial_wait_tail_go = [&]() {
        if (lane_idx == 0u) {
            const auto start_clock = clock64();
            while (ptx::ld_acq_shared(spatial_reduce_ctrl + 1) == 0u) {
                if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                    DG_DEVICE_ASSERT(false and "Spatial Reduce tail-go timeout");
                }
                __nanosleep(64);
            }
        }
        __syncwarp();
    };

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

        // Scatter-gap Reduce state (kGapReduce): the one private combine tile
        // this warp has claimed but not reduced yet.  A claim is never burnt:
        // a claimed tile is reduced either in a later gap or, blocking, right
        // after the drain (d7_gap_reduce_finish), before the tail start.
        [[maybe_unused]] uint32_t gap_tile = 0xffffffffu;
        [[maybe_unused]] uint32_t gap_claims_done = 0u;
        [[maybe_unused]] uint16_t gap_d6_epoch = 0u;
        [[maybe_unused]] bool gap_epoch_loaded = false;
        // Exponential backoff after a not-ready readiness probe: each probe
        // costs up to ~10 sys-scope acquire loads per lane, so a held tile is
        // re-probed only every 2^k-th gap (k <= 6) instead of on every spin.
        [[maybe_unused]] uint32_t gap_skip = 0u;
        [[maybe_unused]] uint32_t gap_backoff_log = 0u;
        // Reduce gap_tile if every contributing combine slot is ready (or,
        // with blocking, once it is); returns true once the tile is done.
        [[maybe_unused]] const auto d7_gap_reduce_tile = [&](const bool blocking) -> bool {
            if constexpr (kGapReduce) {
                constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
                constexpr uint32_t kSubBytes = 1024u;
                DG_STATIC_ASSERT(kNumHiddenBytes % kSubBytes == 0u, "Hidden bytes must be divisible by 1024B");
                DG_STATIC_ASSERT(not kD7LocalDedupActive, "Scatter-gap Reduce assumes one row per top-k slot");
                const uint32_t t = gap_tile;
                const uint32_t tok_idx = t % num_tokens;
                const uint32_t sub_idx = t / num_tokens;
                const uint32_t sub_byte_offset = sub_idx * kSubBytes;
                const int topk_val = lane_idx < kNumTopk
                    ? static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + tok_idx * kNumTopk + lane_idx))
                    : (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                uint32_t pmask = __ballot_sync(0xffffffffu, topk_val >= 0);
                // Non-blocking: every contributing slot must already be ready.
                const auto all_ready = [&]() {
                    const bool lane_ready = ((pmask & (1u << lane_idx)) == 0u) or
                        d6_flags_ready(tok_idx, lane_idx, sub_byte_offset, kSubBytes, gap_d6_epoch);
                    return __all_sync(0xffffffffu, lane_ready);
                };
                if (not all_ready()) {
                    if (not blocking)
                        return false;
                    const auto start_clock = clock64();
                    do {
                        if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                            DG_DEVICE_ASSERT(false and "Scatter-gap Reduce readiness timeout");
                        }
                        asm volatile("nanosleep.u32 64;");
                    } while (not all_ready());
                }
                // bar.warp.sync orders the observing lanes' acquire flag loads
                // before every lane's source loads below.
                __syncwarp();
#if DG_MEGA_TILE_PROF
                const uint32_t prof_red_s = tile_prof::read_gtimer();
#endif
                // Ascending slot order from +0 with FP32 accumulation and one
                // final bf16 rounding: bit-identical to the tail combine.
                float2 acc[8];
                #pragma unroll
                for (uint32_t e = 0u; e < 8u; ++ e)
                    acc[e] = {0.0f, 0.0f};
                while (pmask != 0u) {
                    const uint32_t s0 = static_cast<uint32_t>(__ffs(pmask) - 1);
                    pmask ^= (1u << s0);
                    const auto* src_u4 = reinterpret_cast<const uint4*>(math::advance_ptr<uint8_t>(
                        buffer.combine_token_buffer.get_rank_buffer(s0).get_data_buffer(tok_idx).get_base_ptr(),
                        sub_byte_offset));
                    const uint4 v0 = __ldcs(src_u4 + lane_idx);
                    const uint4 v1 = __ldcs(src_u4 + 32u + lane_idx);
                    const auto* b0 = reinterpret_cast<const nv_bfloat162*>(&v0);
                    const auto* b1 = reinterpret_cast<const nv_bfloat162*>(&v1);
                    #pragma unroll
                    for (uint32_t l = 0u; l < 4u; ++ l)
                        ptx::accumulate(acc[l], b0[l]);
                    #pragma unroll
                    for (uint32_t l = 0u; l < 4u; ++ l)
                        ptx::accumulate(acc[4u + l], b1[l]);
                }
                auto* dst_u4 = reinterpret_cast<uint4*>(math::advance_ptr<uint8_t>(
                    y, static_cast<uint64_t>(tok_idx) * kNumHiddenBytes + sub_byte_offset));
                #pragma unroll
                for (uint32_t v = 0u; v < 2u; ++ v) {
                    uint4 casted;
                    auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0u; l < 4u; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(acc[v * 4u + l]);
                    __stcs(dst_u4 + v * 32u + lane_idx, casted);
                }
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_red_s, tile_prof::read_gtimer(),
                                  14u, t);
#endif
                gap_tile = 0xffffffffu;
                return true;
            } else {
                (void) blocking;
                return true;
            }
        };
        // One gap: claim from the CTA-private cursor (only up to the Middle
        // cap = half of this SM's private tiles; the rest belongs to the
        // 16-warp tail) and reduce up to kGapTilesPerGap ready tiles.
        [[maybe_unused]] const auto d7_gap_reduce_try = [&]() {
            if constexpr (kGapReduce) {
                if (not gap_epoch_loaded) {
                    gap_d6_epoch = static_cast<uint16_t>(ptx::ld_acq(buffer.get_d6_epoch_ptr()));
                    gap_epoch_loaded = true;
                }
                constexpr uint32_t kNumSubs = kHidden * sizeof(nv_bfloat16) / 1024u;
                #pragma unroll 1
                for (uint32_t g = 0u; g < kGapTilesPerGap; ++ g) {
                    if (gap_tile == 0xffffffffu) {
                        if (gap_claims_done != 0u)
                            return;
                        const uint32_t num_private_tiles = get_num_private_reduce_tiles(num_tokens * kNumSubs);
                        const uint32_t middle_claim_cap = num_private_tiles - num_private_tiles / 2u;
                        uint32_t claimed = 0xffffffffu;
                        if (lane_idx == 0u and ptx::ld_acq_shared(spatial_reduce_ctrl + 1) == 0u and
                            ptx::ld_shared(spatial_reduce_ctrl + 0) < middle_claim_cap)
                            claimed = ptx::atomic_add_shared(spatial_reduce_ctrl + 0, 1u);
                        claimed = __shfl_sync(0xffffffffu, claimed, 0);
                        if (claimed >= num_private_tiles) {
                            gap_claims_done = 1u;
                            return;
                        }
                        // A returned index is always reduced, even one past the
                        // cap when several comm warps race for the last ones.
                        if (claimed + 1u >= middle_claim_cap)
                            gap_claims_done = 1u;
                        gap_tile = sm_idx + claimed * kNumSMs;
                    }
                    if (gap_skip > 0u) {
                        -- gap_skip;
                        asm volatile("nanosleep.u32 128;");
                        return;
                    }
                    if (not d7_gap_reduce_tile(false)) {
                        gap_backoff_log = gap_backoff_log < 6u ? gap_backoff_log + 1u : 6u;
                        gap_skip = 1u << gap_backoff_log;
                        return;
                    }
                    gap_backoff_log = 0u;
                }
            }
        };
        [[maybe_unused]] const auto d7_gap_reduce_finish = [&]() {
            if constexpr (kGapReduce) {
                if (gap_tile != 0xffffffffu)
                    d7_gap_reduce_tile(true);
            }
        };

#if DG_D7_HANDOFF_MODE != 0
        // D7 SMEM handoff consumer: this warp's row share of every handed-off
        // FC2 sub-tile (no cross-warp barrier: each thread arrives on the slot's
        // empty barrier).  poll_only returns at the first non-full slot (strict
        // comm_only drains between pulls); both forms return on the terminator
        // without consuming it.
        uint32_t disp_slot = 0u;
        uint32_t disp_phase = 0u;
        [[maybe_unused]] const auto d7_drain_handoff = [&](const bool poll_only) {
            const uint32_t d7_scatter_t =
                kSpatialStreamReduce ? (kFirstDedicatedReduceWarp * 32u) : get_active_dispatch_threads();
                constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
                constexpr uint32_t kNumBankGroupBytes = 16u;
                const uint32_t rows_per_disp_wave = (d7_scatter_t / 32u) * 2u;
                constexpr uint32_t kBatchWaves = 1u;
                const uint32_t warp_row_offset = warp_idx * 2u + (lane_idx >> 4);
                const uint32_t half_base = (lane_idx & 16u);
                const uint32_t lane_in_half = (lane_idx & 15u);
                const uint32_t bank_group_idx = (lane_idx & 7u);

                while (true) {
                    if (poll_only) {
                        // Warp-uniform probe; the blocking wait below then
                        // gives every lane its own acquire of the slot.
                        const uint32_t ready = (lane_idx == 0u and
                            shared_storage.scatter_full_barriers[disp_slot].test_wait(disp_phase)) ? 1u : 0u;
                        if (__shfl_sync(0xffffffffu, ready, 0) == 0u)
                            return;
                    }
                    if constexpr (kGapReduce) {
                        // Post-pull drain: while no handed-off sub-tile is
                        // waiting, spend the gap on ready combine tiles.
                        if (not poll_only) {
                            while (true) {
                                const uint32_t ready = (lane_idx == 0u and
                                    shared_storage.scatter_full_barriers[disp_slot].test_wait(disp_phase)) ? 1u : 0u;
                                if (__shfl_sync(0xffffffffu, ready, 0) != 0u)
                                    break;
                                d7_gap_reduce_try();
                            }
                        }
                    }
                    shared_storage.scatter_full_barriers[disp_slot].wait(disp_phase);
                    const uint4 job_raw = ptx::ld_shared(
                        reinterpret_cast<const uint4*>(&shared_storage.scatter_jobs[disp_slot]));
                    const uint32_t packed_meta = job_raw.z;
                    if (packed_meta & 0x80000000u) {
                        return;
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

                        if constexpr (kD6GroupTiles == 0u) {
                            if (b + 1u == active_batches) {
                                shared_storage.scatter_empty_barriers[disp_slot].arrive();
                                disp_slot = (disp_slot + 1u == DG_D7_RING_SLOTS) ? 0u : (disp_slot + 1u);
                                disp_phase ^= (disp_slot == 0u);
                            }
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

                        if constexpr (kD6GroupTiles > 0u) {
                            if (b + 1u == active_batches) {
                                ptx::fence_acq_rel_sys();
                                shared_storage.scatter_empty_barriers[disp_slot].arrive();
                                disp_slot = (disp_slot + 1u == DG_D7_RING_SLOTS) ? 0u : (disp_slot + 1u);
                                disp_phase ^= (disp_slot == 0u);
                            }
                        }
                    }
#if DG_MEGA_TILE_PROF
                    // Comm-warp executor of the SMEM handoff: the only NVLink Scatter.
                    tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                      prof_rec_idx, prof_t0_ns, prof_d7_s, tile_prof::read_gtimer(),
                                      29u, pool_m_idx / BLOCK_M);
#endif
                }

        };
#endif
#if DG_D7_HANDOFF_MODE == 3
        const auto d7_strict_poll = [&]() {
            if (warp_idx < kFirstDedicatedReduceWarp)
                d7_drain_handoff(true);
        };
#else
        const auto d7_strict_poll = [&]() {};
#endif
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
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs * kCreditsPerPoolBlock;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
#if DG_D7_HANDOFF_MODE == 3
                // Ring reuse waits on GEMM progress, which may wait on a
                // handed-off FC2 sub-tile: keep draining while spinning.
                while (__any_sync(0xffffffffu, ptx::ld_acq(empty_ptr) < l1_empty_count_target))
                    d7_strict_poll();
#else
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
#endif
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

                // Load the weight first (every lane, same address), so that its
                // remote NVLink latency overlaps with the SF copy below -- the
                // upstream order.  Loading it from the elected lane only after the
                // SF loop's __syncwarp() serialised one extra remote read round
                // trip into every pulled token (it cannot be hoisted across the
                // warp barrier), which scales with tokens per dispatch warp.
                const auto weight = *sym_buffer.map(
                    buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);

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
        // The grouped-readiness opening has no runtime policy: the compiled
        // Middle Reduce pin is the decision (used by the post-pull Reduce).
        [[maybe_unused]] constexpr uint32_t weave_active_reduce_warps = kDedicatedReduceWarps;
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
                    // The spatial Reduce claims tiles from a CTA-private SMEM
                    // cursor, so only the epoch needs a per-launch update.
                    const uint32_t prev_ep = *buffer.get_d6_epoch_ptr();
                    ptx::st_rel(buffer.get_d6_epoch_ptr(), (prev_ep >= 65535u) ? 1u : (prev_ep + 1u));
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
        // With the SM 0 streaming scheduler, SM 0 leaves the pull queue and SMs
        // 1..N-1 own the dense range [0, (N-1) * W) of warp starts.
        const uint32_t d1_pull_start = WEAVE_SM0_STREAM_SCHED
            ? (sm_idx == 0u ? 0xffffffffu : (sm_idx - 1u) * kNumDispatchWarps + warp_idx)
            : sm_idx * kNumDispatchWarps + warp_idx;
        for (uint32_t token_idx = d1_pull_start;
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
                    // The spatial Reduce claims tiles from a CTA-private SMEM
                    // cursor, so only the epoch needs a per-launch update.
                    const uint32_t prev_ep = *buffer.get_d6_epoch_ptr();
                    ptx::st_rel(buffer.get_d6_epoch_ptr(), (prev_ep >= 65535u) ? 1u : (prev_ep + 1u));
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
            if constexpr (kPerPeerCombineReady) {
                // Push this launch's grid index (+1: never equal to a zeroed
                // workspace) to every peer; the NVLink barrier below orders it
                // before any peer's combine-readiness tag.
                if (thread_idx < kNumRanks)
                    *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) =
                        ptx::get_grid_idx() + 1;
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
#if WEAVE_RUNTIME_POLICY
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
        // Publish the CTA-0 decision to the whole dispatch group.  Without the
        // runtime policy there is nothing to publish: the NVLink barrier above
        // already ends with a grid sync (which includes this block barrier).
        ptx::sync_aligned(get_active_dispatch_threads(), kDispatchBarrierIdx);
#endif

#if DG_P1_HEAD_HELPER_K > 0
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
#if WEAVE_RUNTIME_POLICY
        const bool weave_policy_ok = shared_storage.weave_policy_ready != 0u;
        const uint32_t weave_comm_warps_per_sm = weave_policy_ok
            ? weave::r_comm_units(shared_storage.weave_policy.selected_r_mask)
            : kNumDispatchWarps;
        // Tier-3 online policy: active_v_head equals the published decision's
        // v_head for every pinned / balanced case and only shrinks under the
        // calibrated hot-expert skew rule (WEAVE_ADAPT_BM=1 + WEAVE_V_HEAD=0).
        const uint32_t selected_v_head = weave_policy_ok
            ? weave::load_active_v_head(workspace.get_weave_policy_control_ptr()) : 0u;
        // KEY runtime gate: the dedicated Middle Reduce warps stream only when
        // CTA-0's live Eq.(1)-(3) verdict keeps them (Middle HBM headroom + a
        // tail worth amortizing).  Gated off, they leave the reduce work to the
        // epilogue tail drain and never contend for Middle HBM.  An invalid
        // policy keeps the compiled pin, so explicit builds stay deterministic.
        const uint32_t weave_active_reduce_warps = weave_policy_ok
            ? weave::load_active_reduce_warps(workspace.get_weave_policy_control_ptr())
            : kDedicatedReduceWarps;
        (void) shared_storage.weave_policy.selected_scatter_mode;
#else
        // Runtime policy compiled out (WEAVE_SM0_STREAM_SCHED=0): the compiled
        // pins are the decision, exactly as for an invalid policy above.
        [[maybe_unused]] constexpr bool weave_policy_ok = false;
        constexpr uint32_t weave_comm_warps_per_sm = kNumDispatchWarps;
        constexpr uint32_t selected_v_head = 0u;
        constexpr uint32_t weave_active_reduce_warps = kDedicatedReduceWarps;
#endif
        const uint32_t active_pull_warps =
            weave_comm_warps_per_sm == 0u ? 1u :
            (weave_comm_warps_per_sm < kNumDispatchWarps
                 ? weave_comm_warps_per_sm : kNumDispatchWarps);
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

        // SM 0 streaming scheduler & independent partial-SM dispatch throttle:
        // when WEAVE_DISPATCH_ACTIVE_SMS = D > 0, only D SMs pull remote tokens
        // while remaining SMs proceed immediately to GEMM / cleanup barriers.
        constexpr bool kSm0StreamSchedActive = WEAVE_SM0_STREAM_SCHED != 0;
        constexpr uint32_t kMaxPullSms = kSm0StreamSchedActive ? (kNumSMs - 1u) : kNumSMs;
        constexpr uint32_t kActiveDispatchSMs =
            (WEAVE_DISPATCH_ACTIVE_SMS == 0 or static_cast<uint32_t>(WEAVE_DISPATCH_ACTIVE_SMS) > kMaxPullSms)
                ? kMaxPullSms : static_cast<uint32_t>(WEAVE_DISPATCH_ACTIVE_SMS);
        const uint32_t pull_sm_idx = kSm0StreamSchedActive
            ? (sm_idx == 0u ? kNumSMs : sm_idx - 1u) : sm_idx;
        const uint32_t pull_num_sms = kActiveDispatchSMs;
        uint32_t initial_pull_warp_idx = ((kSm0StreamSchedActive and sm_idx == 0u) or pull_sm_idx >= kActiveDispatchSMs)
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

        constexpr uint32_t kHeadPullChunk = WEAVE_HEAD_PULL_CHUNK;
        const uint32_t effective_head_pull_end = kHeadPullChunk == 0u
            ? head_pull_end : cute::min(head_pull_end, kHeadPullChunk);
        const bool split_middle_pull = (not kD2SmidMasked) and (active_pull_warps < kNumDispatchWarps);
        const uint32_t head_end_limit = split_middle_pull ? effective_head_pull_end : 0xffffffffu;
        const uint32_t pull_stride = get_pull_warp_stride();

        const auto run_pull_token = [&](const uint32_t token_idx) -> bool {
            d7_strict_poll();
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            if (current_expert_idx >= kNumExpertsPerRank)
                return false;

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
            return true;
        };

        for (uint32_t token_idx = initial_pull_warp_idx; token_idx < head_end_limit; token_idx += pull_stride) {
            if (not run_pull_token(token_idx))
                break;
        }
        if (split_middle_pull and initial_pull_warp_idx != 0xffffffffu and warp_idx < active_pull_warps) {
            uint32_t mid_base = effective_head_pull_end;
            #if DG_P1_HEAD_HELPER_K > 0
            mid_base = cute::max(mid_base, p1_head_end);
            #endif
            const uint32_t mid_stride = pull_num_sms * active_pull_warps;
            for (uint32_t token_idx = mid_base + pull_sm_idx * active_pull_warps + warp_idx;
                 ; token_idx += mid_stride) {
                if (not run_pull_token(token_idx))
                    break;
            }
        }
    #endif
#endif

#if (DG_D7_HANDOFF_MODE != 0) || (DG_D7_QUEUE_SCATTER != 0)
        // ====================================================================
        // Axis 7 (D7): publish 'opening pull finished' and wake the scatter role.
        // ====================================================================
        const uint32_t d7_act_disp_t = get_active_dispatch_threads();
        // Strict comm_only (mode 3) never reads the flag; a rendezvous here
        // would park a finished warp that still owes its handoff row share.
        if constexpr (DG_D7_HANDOFF_MODE != 3) {
            ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
            if (thread_idx == 0) {
                ptx::st_shared(&shared_storage.dispatch_pull_done, 1u);
            }
            ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
        }
#endif

#if DG_D7_HANDOFF_MODE != 0
        {
            const uint32_t d7_scatter_t =
                kSpatialStreamReduce ? (kFirstDedicatedReduceWarp * 32u) : d7_act_disp_t;
            if (warp_idx < kFirstDedicatedReduceWarp) {
                d7_drain_handoff(false);

                ptx::arrive_unaligned(d7_scatter_t + kNumEpilogueThreads, kScatterDrainBarrierIdx);
                // Every handed-off sub-tile is scattered: finish the one tile
                // this warp may still hold before joining the tail start.
                d7_gap_reduce_finish();
                // Non-dedicated Scatter warps fall straight through to the CTA-wide
                // tail Reduce; only the dedicated warps drain Middle tiles.
            } else if constexpr (kSpatialStreamReduce) {
                if (weave_active_reduce_warps > 0u) {
                    const uint16_t disp_d6_epoch = static_cast<uint16_t>(ptx::ld_acq(buffer.get_d6_epoch_ptr()));
                    run_spatial_streaming_reduce.template operator()<2u>(disp_d6_epoch);
                }
            }
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
                // D6 flags of delivered rows, one pending entry per lane of the half-warp
                // (lane `q_pend_cnt` records the next row).  Flushed with one
                // sys-scope fence + relaxed sys epoch stores (same release
                // pattern as the epilogue-side d6_publish_flags).
                [[maybe_unused]] const uint32_t q_d6_epoch = kD7QueueD6Active ?
                    static_cast<uint32_t>(*buffer.get_d6_epoch_ptr()) : 0u;
                [[maybe_unused]] uint32_t q_pend_cnt = 0u, q_pend_tok = 0u, q_pend_meta = 0xffffffffu;
                [[maybe_unused]] const auto q_flush_d6_flags = [&]() {
                    __syncwarp();
                    ptx::fence_acq_rel_sys();
                    if (q_pend_meta != 0xffffffffu) {
                        const uint32_t f_rank = q_pend_meta & 0xffu;
                        const uint32_t f_sub_n = (q_pend_meta >> 8) & 0xffu;
                        const uint32_t f_slot = q_pend_meta >> 16;
                        ptx::st_relaxed_sys(sym_buffer.map(
                            buffer.get_d6_ready_epoch_ptr(q_pend_tok, f_sub_n) + f_slot, f_rank), q_d6_epoch);
                        q_pend_meta = 0xffffffffu;
                    }
                    q_pend_cnt = 0u;
                    __syncwarp();
                };
                // Zero-CAS static split (non warp-split modes): consumer warp u of
                // the kQUnits scatter warps owns slot pairs {2p, 2p+1} for
                // p = u, u + kQUnits, ...  Each half-warp polls only its own
                // slot's ready word; a slot at or past the final tail (known
                // once every producer CTA has flushed) ends the warp's stream.
                // The former single global CAS head serialized ~600 warps on
                // one L2 line (2 slots per CAS: 40-170 ms on Qwen3.8-Max).
                constexpr bool kQStatic = not kD2SmidMasked;
                constexpr uint32_t kQWarpsPerSm = kNumDispatchThreads / 32u;
                constexpr uint32_t kQSms = kD7QueueActiveSms >= kNumSMs ? kNumSMs : kD7QueueActiveSms;
                constexpr uint32_t kQUnits = kQSms * kQWarpsPerSm;
                const uint32_t q_ord = kD7QueueActiveSms >= kNumSMs ? sm_idx :
                    (sm_idx * kD7QueueActiveSms) / kNumSMs;
                uint32_t q_pair = q_ord * kQWarpsPerSm + warp_idx;
                uint32_t q_final_tail = 0xffffffffu;
                while (true) {
                    uint32_t claim_old = 0u, claim_n = 0u;
                    if constexpr (kQStatic) {
                        claim_old = q_pair * 2u;
                        if (claim_old >= q_final_tail) {
                            if constexpr (kD7QueueD6Active) {
                                if (__any_sync(0xffffffffu, q_pend_cnt > 0u))
                                    q_flush_d6_flags();
                            }
                            break;
                        }
                        claim_n = 2u;
                        q_pair += kQUnits;
                        if constexpr (kD7QueueD6Active) {
                            // Publish pending D6 flags before a (possibly long) wait.
                            uint32_t not_ready = 0u;
                            if (lane_in_half == 0u)
                                not_ready = ptx::ld_acq(buffer.get_d7_queue_ready_ptr() + claim_old + (lane_idx >> 4)) != epoch;
                            if (__any_sync(0xffffffffu, not_ready != 0u) and __any_sync(0xffffffffu, q_pend_cnt > 0u))
                                q_flush_d6_flags();
                        }
                    } else if (lane_idx == 0u) {
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
                    if constexpr (!kQStatic) {
                        claim_old = __shfl_sync(0xffffffffu, claim_old, 0);
                        claim_n = __shfl_sync(0xffffffffu, claim_n, 0);
                    }
                    if (!kQStatic and claim_n == 0u) {
                        if constexpr (kD7QueueD6Active) {
                            if (__any_sync(0xffffffffu, q_pend_cnt > 0u))
                                q_flush_d6_flags();
                        }
                        // All producers done is necessary but not sufficient: a producer
                        // may have reserved and published slots after our tail read, so
                        // re-read head/tail after observing producer_done and exit only
                        // when the queue is truly drained.
                        uint32_t drained = 0u;
                        if (lane_idx == 0u and
                            ptx::ld_acq(buffer.get_d7_queue_producer_done_ptr()) >= kNumSMs) {
                            const uint32_t t2 = ptx::ld_acq(buffer.get_d7_queue_tail_ptr());
                            const uint32_t h2 = ptx::ld_acq(buffer.get_d7_queue_head_ptr());
                            drained = (h2 >= t2) ? 1u : 0u;
                        }
                        if (__shfl_sync(0xffffffffu, drained, 0) != 0u)
                            break;
                        __nanosleep(64);
                        continue;
                    }
                    const uint32_t half = lane_idx >> 4;
                    uint32_t q_tail_seen = 0xffffffffu;
                    if (half < claim_n) {
                        const uint32_t slot = claim_old + half;
                        uint64_t desc = 0ull;
                        if (lane_in_half == 0u) {
                            bool past_tail = false;
                            uint32_t spins = 0u;
                            while (ptx::ld_acq(buffer.get_d7_queue_ready_ptr() + slot) != epoch) {
                                if (kQStatic and ((++ spins) & 7u) == 0u and
                                    ptx::ld_acq(buffer.get_d7_queue_producer_done_ptr()) >= kNumSMs) {
                                    // Every producer flushed: the tail is final and every
                                    // slot below it is (or is about to be) published.
                                    q_tail_seen = ptx::ld_acq(buffer.get_d7_queue_tail_ptr());
                                    if (slot >= q_tail_seen) {
                                        past_tail = true;
                                        break;
                                    }
                                }
                                __nanosleep(32);
                            }
                            desc = past_tail ? 0x8000000000000000ull :
                                ptx::ld_acq_gpu(buffer.get_d7_queue_desc_ptr() + slot);
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
                            auto* const q_local_row = buffer.get_d7_local_row_ptr(dst_rank, dst_token, n_idx) + lane_in_half;
                            asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                                         : "=r"(*reinterpret_cast<uint32_t*>(&packed.x)),
                                           "=r"(*reinterpret_cast<uint32_t*>(&packed.y)),
                                           "=r"(*reinterpret_cast<uint32_t*>(&packed.z)),
                                           "=r"(*reinterpret_cast<uint32_t*>(&packed.w))
                                         : "l"(q_local_row));
                            *sym_buffer.map(dst_ptr, dst_rank) = packed;
                            // Re-zero the consumed row so the next launch's
                            // wait-free red.add pre-Reduce starts from a clean
                            // accumulator (kernel boundaries order the two).
                            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                         :: "l"(q_local_row), "r"(0u), "r"(0u), "r"(0u), "r"(0u) : "memory");
                            if constexpr (kD7QueueD6Active) {
                                if (lane_in_half == q_pend_cnt) {
                                    q_pend_tok = dst_token;
                                    q_pend_meta = dst_rank | (sub_n << 8) | (dst_topk << 16);
                                }
                                ++ q_pend_cnt;
                            }
                        }
                    }
                    if constexpr (kQStatic)
                        q_final_tail = cute::min(q_final_tail, __reduce_min_sync(0xffffffffu, q_tail_seen));
                    if constexpr (kD7QueueD6Active) {
                        if (__any_sync(0xffffffffu, q_pend_cnt >= 16u))
                            q_flush_d6_flags();
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
                        __nanosleep(kFusedSharedInt7 ? kFsRingSleepNs : 512u);
                }
                ptx::sync_aligned(d7_act_disp_t, kDispatchBarrierIdx);
            }
            ptx::arrive_unaligned(d7_act_disp_t + kNumEpilogueThreads, kScatterDrainBarrierIdx);
        }
#else
        // D7 mode 0 has no Scatter role, so the dispatch warps are idle after
        // Opening Pull: only the dedicated warps drain Middle tiles here, and
        // every dispatch warp still joins the CTA-wide tail below.
        if constexpr (kSpatialStreamReduce) {
            if (warp_idx >= kFirstDedicatedReduceWarp and weave_active_reduce_warps > 0u) {
                const uint16_t disp_d6_epoch = static_cast<uint16_t>(ptx::ld_acq(buffer.get_d6_epoch_ptr()));
                run_spatial_streaming_reduce.template operator()<2u>(disp_d6_epoch);
            }
        }
#endif

        if constexpr (kSpatialStreamReduce) {
            // Axis 6 (D6 All-Warp Reduce, spatial path): a dispatch warp with no
            // Middle role (WEAVE_REDUCE_WARPS < kNumDispatchWarps) joins the
            // tail-start barrier with W04..W15, so it only starts once the GEMM
            // is complete instead of claiming tiles and polling flags from t=0.
            // A dedicated Middle warp has drained its share of this SM's
            // private tiles above and joins through the tail-go flag, without
            // holding any barrier participant back.  Every warp W00..W15 then
            // claims the remaining private tiles from the same CTA cursor.  The
            // barrier-1 cleanup sync below (dispatch + epilogue warps) is what
            // makes Workspace Cleanup safe.
            if (warp_idx < kFirstDedicatedReduceWarp) {
                spatial_tail_start();
            } else {
                spatial_wait_tail_go();
            }
            run_spatial_tail_combine();
        }

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
                // Static-index select: a runtime `expert_blk_offset[i / 32]` forces the
                // per-lane register array into local memory (one LDL on this path).
                uint32_t expert_blk_offset_word = expert_blk_offset[0];
                #pragma unroll
                for (uint32_t w = 1; w < kNumExpertsPerLane; ++ w)
                    expert_blk_offset_word = (i / 32 == w) ? expert_blk_offset[w] : expert_blk_offset_word;
                const uint32_t expert_pool_block_offset_clean = ptx::exchange(expert_blk_offset_word, i % 32);
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

        // All-16 tail: SM 0's W00 lane 0 only signals here and waits for the
        // peers after the tail Reduce (comm::nvlink_barrier_deferred_wait), so
        // W00 of SM 0 joins the tail instead of spinning on the slowest rank.
        comm::nvlink_barrier<kNumRanks, kNumSMs, 128,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(cleanup_dispatch_threads, kDispatchBarrierIdx); },
            true,
            false,
            kAll16DeferCleanWait
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
#if DG_P1_HEAD_HELPER_K == 0 || DG_HEAD_NONBLOCKING
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
                if constexpr (not kL2KBlockMaskDep) {
                    const auto ptr = workspace.get_l2_full_count_ptr(block_idx);
                    const auto num_expected_blocks =
                        (L2_SHAPE_K / (BLOCK_N / 2)) * kCreditsPerPoolBlock * (pool_block_idx / kNumRingBlocks + 1);
                    while (ptx::ld_acq(ptr) != num_expected_blocks);
                }
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

            // Per-K-block L2 readiness (only consulted for routed Linear2 tasks)
            const bool wait_l2_k_blocks = kL2KBlockMaskDep and task_info.block_phase == sched::BlockPhase::Linear2;
            const uint32_t* l2_ready_ptr = workspace.get_l2_full_count_ptr(ring_block_idx);
            const uint32_t l2_expected_mask = ((pool_block_idx / kNumRingBlocks) & 1u) ? 0u : kL2KBlockFullMask;
            uint32_t l2_pending_mask = kL2KBlockFullMask;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if constexpr (kL2KBlockMaskDep) {
                    if (wait_l2_k_blocks) {
                        const uint32_t k_block_mask =
                            ((1u << kNumL1BlockNsPerL2KBlock) - 1u) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
                        while (l2_pending_mask & k_block_mask)
                            l2_pending_mask = ptx::ld_acq(l2_ready_ptr) ^ l2_expected_mask;
                    }
                }
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                uint32_t m_idx = block_idx * BLOCK_M + (kSubTileActive ? task_info.get_m_row_offset() : 0u);
                uint32_t k_idx = k_block_idx * BLOCK_K;
                // The SF chunk always covers the whole pool block; the MMA selects the
                // sub-tile's rows through the SFA TMEM column offset.
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
#if DG_P1_HEAD_HELPER_K == 0 || DG_HEAD_NONBLOCKING
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
#if DG_L2_PREFETCH_DIST > 0
            // Rolling L2 weight prefetch: the SMEM pipeline bounds the in-flight
            // weight bytes per SM (kNumStages tiles), so HBM runs well below its
            // peak.  Ahead of each K block of expert e this warp asks L2 for the
            // same (N block, K block) weight + SF tile of expert e + D: a
            // fire-and-forget bulk prefetch, no SMEM, no barrier.  Each CTA
            // prefetches exactly the tiles it will load itself.
            const bool pf_active = not task_info.is_shared() and
                task_info.local_expert_idx + DG_L2_PREFETCH_DIST < kNumExpertsPerRank;
            const uint32_t pf_n_idx = (task_info.local_expert_idx + DG_L2_PREFETCH_DIST) * shape_n + n_block_idx * BLOCK_N;
            const uint32_t pf_sfb_k_base = (task_info.local_expert_idx + DG_L2_PREFETCH_DIST) * shape_sfb_k;
#endif

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
#if DG_L2_PREFETCH_DIST > 0
                if (pf_active and cute::elect_one_sync()) {
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                                 :: "l"(reinterpret_cast<uint64_t>(tensor_map_b_ptr)),
                                    "r"(k_block_idx * BLOCK_K), "r"(pf_n_idx) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                                 :: "l"(reinterpret_cast<uint64_t>(tensor_map_sfb_ptr)),
                                    "r"(n_block_idx * BLOCK_N), "r"(pf_sfb_k_base + k_block_idx * (BLOCK_K / 128u)) : "memory");
                }
                __syncwarp();
#endif
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
#if DG_P1_HEAD_HELPER_K == 0 || DG_HEAD_NONBLOCKING
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
            // Scalar stage-0 descriptor bases (official form): the per-stage base is
            // `lo + stage_idx * stage_bytes / 16`, a uniform add instead of a per-k-block SHFL.
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;
            const uint32_t shared_b_desc_lo = shared_b_desc.lo;

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
                // UTCCP places SF row r of the pool block at TMEM column r / 32.
                const uint32_t sfa_row_col = kSubTileActive ? (task_info.get_m_row_offset() / 32u) : 0u;

                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                if constexpr (kAccumOverlapCols > 0u) {
                    if (current_iter_idx > 1u) {
                        const uint32_t prev_stage_idx = (current_iter_idx - 2u) % kNumEpilogueStages;
                        const uint32_t prev_stage_phase = ((current_iter_idx - 2u) / kNumEpilogueStages) & 1u;
                        shared_storage.overlap_empty_barriers[prev_stage_idx].wait(prev_stage_phase);
                    }
                }
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

                    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                    const uint32_t b_desc_base_lo = (task_info.is_shared() ? shared_b_desc_lo : b_desc_lo) +
                                                    stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
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
                                        accum_stage_idx * kAccumStageStride + n_sub * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0,
                                        runtime_instr_desc,
                                        kTmemStartColOfSFB + umma_k_block_idx * kNumSFBTmemColsPerSubtile + n_sub * 4,
                                        kTmemStartColOfSFA + umma_k_block_idx * kNumSFATmemColsPerSubtile + sfa_row_col);
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
#if DG_P1_HEAD_HELPER_K == 0 || DG_HEAD_NONBLOCKING
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
        constexpr bool kEpiRegPrefetchLegal =
            (kNumNSubtiles == 1u and DG_D3_DYN_EPI == 0);
        constexpr uint32_t kNumEpiPrefetchStages = (WEAVE_EPI_REG_STAGES == 0)
            ? ((kTmemPipelineMode >= 175u and (BLOCK_M == 80u or BLOCK_M == 96u or BLOCK_M >= 240u) and kEpiRegPrefetchLegal) ? 1u : 0u)
            : static_cast<uint32_t>(WEAVE_EPI_REG_STAGES) - 1u;
        DG_STATIC_ASSERT(kNumEpiPrefetchStages == 0u or kEpiRegPrefetchLegal,
                         "WEAVE_EPI_REG_STAGES=2 needs one N subtile and a static epilogue");
        // Without register prefetch, still load a whole store block from TMEM
        // before the single tcgen05.wait::ld and release the stage right after
        // the last valid block's loads: the upstream epilogue's behaviour for
        // every BLOCK_M != 240.  Only N-subtiled or dynamic epilogues keep the
        // per-atom wait, since their release point depends on the subtile index.
        constexpr bool kBatchedEpiTmemLoads = (kNumEpiPrefetchStages == 0u) and kEpiRegPrefetchLegal;
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
        // Acquire load: a weak load here can be hoisted above the Opening epoch
        // bump (bar.sync carries no memory clobber), which would leave the tail
        // Reduce waiting on epoch 0 while the D6 flags already carry epoch 1.
        [[maybe_unused]] const uint16_t d6_cur_epoch = kD6GroupTiles > 0 ?
            static_cast<uint16_t>(ptx::ld_acq(buffer.get_d6_epoch_ptr())) : static_cast<uint16_t>(0u);

        uint32_t current_iter_idx = 0;
        [[maybe_unused]] uint32_t tma_store_seq = 0;
#if DG_D7_HANDOFF_MODE != 0
        uint32_t epi_scatter_slot = 0u;
        uint32_t epi_scatter_phase = 0u;
        uint32_t epi_scatter_step = 0u;
        bool scatter_handoff_enabled = false;
        [[maybe_unused]] uint32_t d7_epi_remote_subtiles = 0u;   // hybrid fallbacks (stage 12)
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
                const uint32_t half_mask = (lane_idx < 16u) ? 0x0000ffffu : 0xffff0000u;
                const uint32_t half_base = lane_idx & 16u;
                float4* const local_ptr =
                    buffer.get_d7_local_row_ptr(dst_rank_idx, dst_token_idx, n_idx) + lane_in_half;
                uint32_t* const local_words = reinterpret_cast<uint32_t*>(local_ptr);
                const auto* packed_words = reinterpret_cast<const uint32_t*>(&packed);
                uint32_t next_cnt = 1u;
                if (hit_count > 1u) {
                    // Wait-free local pre-Reduce: every contributor adds its
                    // packed bf16x2 words straight into the zero-initialized
                    // local row with hardware reductions, fences, then bumps the
                    // arrival counter with one acq_rel fetch-add.  The last
                    // contributor alone publishes the queue descriptor; the
                    // consumer reads the merged row and re-zeroes it for the
                    // next launch.  No CAS, no lock bit, no spin: the 102 ms
                    // SIMT livelock of the lock-based protocol cannot happen
                    // because no warp ever waits on the subtile state.
                    uint32_t* const state_ptr =
                        buffer.get_d7_subtile_state_ptr(dst_rank_idx, dst_token_idx, sub_n_idx);
                    #pragma unroll
                    for (uint32_t w = 0; w < 4; ++ w)
                        asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                                     :: "l"(local_words + w), "r"(packed_words[w]) : "memory");
                    asm volatile("fence.acq_rel.gpu;" ::: "memory");
                    __syncwarp(half_mask);
                    uint32_t old_cnt = 0u;
                    if (lane_in_half == 0u)
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                     : "=r"(old_cnt) : "l"(state_ptr), "r"(1u) : "memory");
                    next_cnt = __shfl_sync(half_mask, old_cnt, half_base) + 1u;
                    if (next_cnt == hit_count and lane_in_half == 0u) {
                        // Reset the generation counter; kernel boundaries order
                        // the reset against the next launch's contributors.
                        asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                     :: "l"(state_ptr), "r"(0u) : "memory");
                    }
                } else {
                    // Single-hit rows store straight into the zeroed local row
                    // (0 + v is exact in bf16): no reduction traffic, no state.
                    // The fence + warp sync order every lane's row store before
                    // lane 0 publishes the queue descriptor.
                    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                 :: "l"(local_ptr),
                                    "r"(packed_words[0]), "r"(packed_words[1]),
                                    "r"(packed_words[2]), "r"(packed_words[3]) : "memory");
                    asm volatile("fence.acq_rel.gpu;" ::: "memory");
                    __syncwarp(half_mask);
                }
                if (next_cnt == hit_count) {
                    // Published only after this contributor's fence + the acq_rel
                    // counter chain, so the consumer sees the fully merged row.
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
                    const uint32_t half_mask = (lane_idx < 16u) ? 0x0000ffffu : 0xffff0000u;
                    const uint32_t half_base = lane_idx & 16u;
                    const uint32_t leader_topk_idx = static_cast<uint32_t>(__ffs(topk_bm) - 1);
                    const uint32_t sub_n_idx = n_idx / 128u;
                    uint32_t* const state_ptr = buffer.get_d7_subtile_state_ptr(dst_rank_idx, dst_token_idx, sub_n_idx);
                    float4* const local_ptr = buffer.get_d7_local_row_ptr(dst_rank_idx, dst_token_idx, n_idx) + lane_in_half;
                    uint32_t* const local_words = reinterpret_cast<uint32_t*>(local_ptr);
                    const auto* packed_words = reinterpret_cast<const uint32_t*>(&packed);

                    // Wait-free local pre-Reduce (see the queue path above): every
                    // contributor red.adds its packed bf16x2 words into the
                    // zero-initialized local row, fences, then bumps the arrival
                    // counter with one acq_rel fetch-add.  The last contributor
                    // alone reads the merged row, forwards it once to the leader
                    // combine slot, re-zeroes the row and resets the state word
                    // for the next launch.  No CAS, no lock bit, no spin.
                    #pragma unroll
                    for (uint32_t w = 0; w < 4; ++ w)
                        asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                                     :: "l"(local_words + w), "r"(packed_words[w]) : "memory");
                    asm volatile("fence.acq_rel.gpu;" ::: "memory");
                    __syncwarp(half_mask);
                    uint32_t old_cnt = 0u;
                    if (lane_in_half == 0u)
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                     : "=r"(old_cnt) : "l"(state_ptr), "r"(1u) : "memory");
                    const uint32_t next_cnt = __shfl_sync(half_mask, old_cnt, half_base) + 1u;
                    if (next_cnt == hit_count) {
                        // The acq_rel counter chain orders this read after every
                        // contributor's red.adds, so the row is fully merged.
                        float4 merged;
                        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&merged.x)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&merged.y)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&merged.z)),
                                       "=r"(*reinterpret_cast<uint32_t*>(&merged.w))
                                     : "l"(local_ptr));
                        const auto leader_tok = buffer.combine_token_buffer.get_rank_buffer(leader_topk_idx)
                                                .get_data_buffer(dst_token_idx);
                        const auto leader_ptr = math::advance_ptr<float4>(
                            leader_tok.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            lane_in_half * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(leader_ptr, dst_rank_idx) = merged;
                        // Re-zero the row so the next launch's red.adds start
                        // from a clean accumulator, then reset the state word.
                        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                     :: "l"(local_ptr), "r"(0u), "r"(0u), "r"(0u), "r"(0u) : "memory");
                        __syncwarp(half_mask);
                        if (lane_in_half == 0u) {
                            asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                         :: "l"(state_ptr), "r"(0u) : "memory");
                        }
                    }
                }
            }
        };

        bool fused_shared_peers_ready = false;
        // D6 aggregated publish state.  For multi-tile groups each tile's
        // writer performs one gpu-scope acq_rel RMW on the group counter
        // (`atomic_add_acq_rel_gpu`): the release half publishes this tile's
        // row stores, and the acquire half of the last arriver's RMW observes
        // every predecessor tile's release through the counter's RMW chain, so
        // all rows of the group happen-before the last arriver's publication.
        // The returned count is consumed one task later so the L2 round trip
        // overlaps the next tile's TMEM/STSM work.  The publication itself
        // (`d6_publish_flags`) is a sys-scope fence (`fence.acq_rel.sys`)
        // followed by relaxed sys-scope flag stores; the fence is cumulative
        // over everything that happens-before it, which forms a legal
        // fence-release pattern for the peer rank's `ld.acquire.sys` reader.
        [[maybe_unused]] uint32_t d6_pend_old_cnt = 0u;
        [[maybe_unused]] uint32_t d6_pend_m_base = 0u;
        [[maybe_unused]] uint32_t d6_pend_group = 0u;
        [[maybe_unused]] uint32_t d6_pend_valid_m = 0u;
        [[maybe_unused]] uint32_t d6_pend_is_shared = 0u;
        [[maybe_unused]] uint32_t d6_pend_active = 0u;
        [[maybe_unused]] const auto d6_publish_flags = [&](const uint32_t& m_base, const uint32_t& group_idx,
                                                            const uint32_t& valid_m_rows, const uint32_t& is_shared) {
#if DG_D7_HANDOFF_MODE != 0
            // Conservative drain check: the pending group's slots were reused
            // (hence drained) at least once by the time this runs; the current
            // slot pair wait additionally orders the comm warps' remote stores
            // into this thread's happens-before for the publication below.
            // Slot j >= current was last filled in the previous ring round
            // (empty phase = phase ^ 1), slot j < current in this round.
            #pragma unroll
            for (uint32_t j = 0u; j < DG_D7_RING_SLOTS; ++ j)
                shared_storage.scatter_empty_barriers[j].wait(
                    j >= epi_scatter_slot ? (epi_scatter_phase ^ 1u) : epi_scatter_phase);
#endif
            // Sys-scope release fence before the relaxed sys flag stores.  The
            // readers are `ld.acquire.sys` loads on other ranks (other GPUs),
            // so a gpu-scope fence would not order the remote row stores
            // before the flags for them; `fence.acq_rel.sys` is cumulative and
            // covers this warp's rows plus the predecessor tiles' rows
            // observed through the acq_rel group-counter chain.  Cost: about
            // one sys fence per group publication, only on kD6GroupTiles > 0
            // paths.
            ptx::fence_acq_rel_sys();
            for (uint32_t m_row = lane_idx; m_row < valid_m_rows; m_row += 32u) {
                uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                if (is_shared != 0u) {
                    dst_rank_idx = sym_buffer.rank_idx;
                    dst_token_idx = m_base + m_row;
                    dst_topk_idx = kNumTopk;
                } else {
                    const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_base + m_row);
                    dst_rank_idx = src_metadata.rank_idx;
                    dst_token_idx = src_metadata.token_idx;
                    dst_topk_idx = src_metadata.topk_idx & 0xffu;
                }
                uint32_t* const ready_ptr =
                    buffer.get_d6_ready_epoch_ptr(dst_token_idx, group_idx) + dst_topk_idx;
                ptx::st_relaxed_sys(sym_buffer.map(ready_ptr, dst_rank_idx),
                                    static_cast<uint32_t>(d6_cur_epoch));
            }
        };

        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            if constexpr (kFusedSharedInt7) {
                // The tail SharedLinear2 task folds every routed INT7 stripe into
                // `y`; wait once (all ranks) until every routed epilogue has
                // published its NVLink stores, then warm this task's L2 tile.
                if (task_info.block_phase == sched::BlockPhase::SharedLinear2 and not fused_shared_peers_ready) {
                    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                         kEpilogueGridSyncIndex, kFusedSharedInt7BarrierTag,
                                         comm::BarrierTimeoutPolicy::Diagnostic, kFsBarrierSleepNs>(
                        workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                        [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });
                    fused_shared_peers_ready = true;
                }
                if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                    const uint32_t fs_valid_m = task_info.valid_m;
                    const uint32_t fs_wg_start_m = epilogue_wg_idx * WG_BLOCK_M_RAW;
                    const uint32_t fs_num_rows = fs_valid_m <= fs_wg_start_m
                        ? 0u : cute::min(fs_valid_m - fs_wg_start_m, WG_BLOCK_M_RAW);
                    const uint32_t fs_num_sbs = math::ceil_div(fs_num_rows, STORE_BLOCK_M_L2);
                    if (fs_num_sbs > 0u and warp_idx_in_wg == 0u and cute::elect_one_sync()) {
                        const uint32_t fs_sb_idx =
                            (task_info.pool_block_idx * BLOCK_M + fs_wg_start_m) / STORE_BLOCK_M_L2;
                        const uint32_t fs_n_block_idx =
                            task_info.n_cluster_idx * 2u + (is_leader_cta ? 0u : 1u);
                        const uint8_t* const fs_pf_ptr =
                            buffer.combine_token_buffer.get_base_ptr<uint8_t>() +
                            static_cast<uint64_t>(fs_n_block_idx) * kFsInt7NBlkStride +
                            static_cast<uint64_t>(fs_sb_idx) * kFsInt7SbBytes;
                        const uint32_t fs_pf_bytes =
                            cute::min(fs_num_sbs * kFsInt7SbBytes, kFsInt7PrefetchBytes);
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                     :: "l"(fs_pf_ptr), "r"(fs_pf_bytes) : "memory");
                    }
                }
            }
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
            // Sub-tiled tasks start `task_row_off` rows into their pool block; every
            // global row base below includes it, TMEM columns stay tile-relative.
            const uint32_t task_row_off = kSubTileActive ? task_info.get_m_row_offset() : 0u;
            const uint32_t task_credits = kSubTileActive ? task_info.get_m_credits() : 1u;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M + task_row_off;
            const uint32_t m_idx = block_idx * BLOCK_M + task_row_off;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M + task_row_off;
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
                    const auto num_expected_blocks =
                        (L2_SHAPE_N / BLOCK_N) * kCreditsPerPoolBlock * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

#if DG_D3_DYN_EPI > 0
                if (wg_atoms == 0u) {
                    ptx::tcgen05_before_thread_sync();
                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                } else {
                    #pragma unroll
                    for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                        const uint32_t sub_accum_base = accum_stage_idx * kAccumStageStride + n_sub * UMMA_N;
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
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
                                    thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                                    thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
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
                                amax_values[i].x = fmaxf(amax_values[i].x, wp_amax.x);
                                amax_values[i].y = fmaxf(amax_values[i].y, wp_amax.y);

                                // Upstream UE8M0 selection (math::get_ue8m0_sf_exp): integer add/shift on the
                                // amax bits instead of FMUL(1/448) + log2/pow2 bit tricks.  Identical SF
                                // for every amax >= 448 * 2^-22; below that the upstream exponent floor
                                // (2^-22) applies, so the default path now matches upstream bit-for-bit.
                                const uint2 sf_exp = {math::get_ue8m0_sf_exp(amax_values[i].x),
                                                      math::get_ue8m0_sf_exp(amax_values[i].y)};
                                const float2 sf_inv = {math::get_ue8m0_sf_inv<float>(sf_exp.x),
                                                       math::get_ue8m0_sf_inv<float>(sf_exp.y)};

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
                                    const uint32_t token_base_idx = task_row_off + store_m_idx + i * ATOM_M;
                                    __builtin_assume(token_base_idx < BLOCK_M);
                                    const auto sf_token_idx = block_idx * SF_BLOCK_M
                                        + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                                    const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                                    sf_base_ptr[sf_addr] = static_cast<uint8_t>(sf_exp.x);
                                    sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                        static_cast<uint8_t>(sf_exp.y);
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
                    const uint32_t sub_accum_base = accum_stage_idx * kAccumStageStride + n_sub * UMMA_N;
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                        }
                    };
                    if constexpr (kNumEpiPrefetchStages > 0) {
                        if (num_valid_store_blocks_l1 == 0u) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                        } else {
                            issue_l1_epi_block(0u);
                        }
                    } else if constexpr (kBatchedEpiTmemLoads) {
                        // A warpgroup without valid rows releases the stage at once
                        // (upstream prefetch_prologue).
                        if (num_valid_store_blocks_l1 == 0u) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                        }
                    }

                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                        if (s >= num_valid_store_blocks_l1) {
                            // Partial warpgroup slice: the per-atom path releases the
                            // TMEM stage at its last atom, which a partial slice never
                            // reaches, so release it here (the prefetch and batched
                            // paths already released after the last valid block).
                            if constexpr (kNumEpiPrefetchStages == 0 and not kBatchedEpiTmemLoads) {
                                if (n_sub == kNumNSubtiles - 1) {
                                    ptx::tcgen05_before_thread_sync();
                                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                    if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                                }
                            }
                            break;
                        }
                        if constexpr (kNumEpiPrefetchStages > 0) {
                            cutlass::arch::fence_view_async_tmem_load();
                            issue_l1_epi_block(s + 1u);
                        } else if constexpr (kBatchedEpiTmemLoads) {
                            // Upstream wait_and_prefetch_next without prefetch: issue
                            // every atom of this store block, then wait once, instead
                            // of one tcgen05.wait::ld per 8-row atom.  The last valid
                            // block also releases the TMEM stage right here, before
                            // its SwiGLU, FP8 cast and TMA store.
                            issue_l1_epi_block(s);
                            if (s + 1u != num_valid_store_blocks_l1)
                                cutlass::arch::fence_view_async_tmem_load();
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
                            if constexpr (kNumEpiPrefetchStages > 0 or kBatchedEpiTmemLoads) {
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
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
                                thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                                thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
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
                            amax_values[i].x = fmaxf(amax_values[i].x, wp_amax.x);
                            amax_values[i].y = fmaxf(amax_values[i].y, wp_amax.y);

                            // Upstream UE8M0 selection (math::get_ue8m0_sf_exp): integer add/shift on the
                            // amax bits instead of FMUL(1/448) + log2/pow2 bit tricks.  Identical SF
                            // for every amax >= 448 * 2^-22; below that the upstream exponent floor
                            // (2^-22) applies, so the default path now matches upstream bit-for-bit.
                            const uint2 sf_exp = {math::get_ue8m0_sf_exp(amax_values[i].x),
                                                  math::get_ue8m0_sf_exp(amax_values[i].y)};
                            const float2 sf_inv = {math::get_ue8m0_sf_inv<float>(sf_exp.x),
                                                   math::get_ue8m0_sf_inv<float>(sf_exp.y)};

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
                                const uint32_t token_base_idx = task_row_off + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                                __builtin_assume(token_base_idx < BLOCK_M);
                                const auto sf_token_idx = block_idx * SF_BLOCK_M
                                    + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                                const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                                sf_base_ptr[sf_addr] = static_cast<uint8_t>(sf_exp.x);
                                sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                    static_cast<uint8_t>(sf_exp.y);
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
                        if constexpr (kL2KBlockMaskDep) {
                            ptx::red_xor_rel(workspace.get_l2_full_count_ptr(ring_block_idx), 1u << n_block_idx);
                        } else {
                            ptx::red_add_rel(
                                workspace.get_l2_full_count_ptr(ring_block_idx), kNumNSubtiles * task_credits);
                        }
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), task_credits);
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
                            workspace.get_l2_empty_count_ptr(ring_block_idx), task_credits);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M_L2 % 8 == 0, "Invalid store M");

#if DG_D7_HANDOFF_MODE != 0
                #pragma unroll
                for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                    const uint32_t sub_accum_base = accum_stage_idx * kAccumStageStride + n_sub * UMMA_N;
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                                tmem_released = true;
                            }

                            bool offload_to_dispatch = false;
                            const uint32_t mode_ring_idx = (epi_scatter_step ++) & 3u;
                            if constexpr (DG_D7_HANDOFF_MODE == 3) {
                                // Strict comm_only: block on a free slot, never scatter.
                                (void) mode_ring_idx;
                                scatter_handoff_enabled = true;
                                shared_storage.scatter_empty_barriers[epi_scatter_slot].wait(epi_scatter_phase ^ 1u);
                                offload_to_dispatch = true;
                            } else if constexpr (DG_D7_HANDOFF_MODE != 2) {
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
                                epi_scatter_slot = (epi_scatter_slot + 1u == DG_D7_RING_SLOTS) ? 0u : (epi_scatter_slot + 1u);
                                epi_scatter_phase ^= (epi_scatter_slot == 0u);
                            } else {
                                ++ d7_epi_remote_subtiles;
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                } else {
                    #pragma unroll
                    for (uint32_t n_sub = 0; n_sub < kNumNSubtiles; ++ n_sub) {
                        const uint32_t sub_accum_base = accum_stage_idx * kAccumStageStride + n_sub * UMMA_N;
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
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
                    const uint32_t sub_accum_base = accum_stage_idx * kAccumStageStride + n_sub * UMMA_N;
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                        }
                    };
                    if constexpr (kNumEpiPrefetchStages > 0) {
                        if (num_valid_store_blocks_l2 == 0u) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                        } else {
                            issue_l2_epi_block(0u);
                        }
                    } else if constexpr (kBatchedEpiTmemLoads) {
                        if (num_valid_store_blocks_l2 == 0u) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                        }
                    }

                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M_L2; ++ s) {
                        if (s >= num_valid_store_blocks_l2) {
                            // Partial warpgroup slice: release the TMEM stage on the
                            // per-atom path (its last-atom release is never reached).
                            if constexpr (kNumEpiPrefetchStages == 0 and not kBatchedEpiTmemLoads) {
                                if (n_sub == kNumNSubtiles - 1) {
                                    ptx::tcgen05_before_thread_sync();
                                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                                    if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
                                }
                            }
                            break;
                        }
                        if constexpr (kNumEpiPrefetchStages > 0) {
                            cutlass::arch::fence_view_async_tmem_load();
                            issue_l2_epi_block(s + 1u);
                        } else if constexpr (kBatchedEpiTmemLoads) {
                            // Upstream `wait_and_prefetch_next` without prefetch:
                            // issue every atom of this store block, then wait once
                            // (the last valid block releases the TMEM stage inside).
                            issue_l2_epi_block(s);
                            if (s + 1u != num_valid_store_blocks_l2)
                                cutlass::arch::fence_view_async_tmem_load();
                        }

                        // Read the source metadata of this warp's rows before the
                        // shared memory stores (upstream `cached_src_metadata`): the
                        // loads then overlap the STSM + warpgroup barrier instead of
                        // each one serialising behind the previous row's remote
                        // store (a plain global store the compiler cannot reorder).
                        layout::TokenSrcMetadata cached_src_metadata[kNumRowsPerWarp];
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 +
                                j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                            if (m_idx_in_block < valid_m) {
                                cached_src_metadata[j] = task_info.is_shared() ?
                                    layout::TokenSrcMetadata{static_cast<uint32_t>(sym_buffer.rank_idx), pool_m_idx + m_idx_in_block, kNumTopk} :
                                    *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                            }
                        }

                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_M_L2 / ATOM_M; ++ i) {
                            uint32_t values_local[ATOM_M];
                            uint32_t* values = values_local;
                            if constexpr (kNumEpiPrefetchStages > 0 or kBatchedEpiTmemLoads) {
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
if constexpr (kAccumOverlapCols > 0u) { if (accum_stage_idx == 0u ? (epilogue_wg_idx == kNumEpilogueWarpgroups - 1u) : (epilogue_wg_idx == 0u)) shared_storage.overlap_empty_barriers[accum_stage_idx].arrive(0u); }
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

                            const uint32_t dst_rank_idx = cached_src_metadata[j].rank_idx;
                            const uint32_t dst_token_idx = cached_src_metadata[j].token_idx;
                            const uint32_t dst_topk_idx = cached_src_metadata[j].topk_idx;

                            const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                                (lane_idx % 16 / 8) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                                row_in_store * kSwizzleCDMode +
                                (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                            const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));
                            if constexpr (kFusedSharedInt7) {
                                const uint32_t fs_lane16 = lane_idx & 15u;
                                const uint32_t fs_n_block_idx = n_idx / BLOCK_N;
                                uint8_t* const fs_comb_base =
                                    buffer.combine_token_buffer.get_base_ptr<uint8_t>();
                                const uint64_t fs_evict_first =
                                    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST);
                                if (task_info.is_shared()) {
                                    // Tail SharedLinear2 owns the whole combine: fold
                                    // its own row plus every routed INT7 slot into y.
                                    float4 fs_acc_packed = packed;
                                    auto* const fs_acc = reinterpret_cast<nv_bfloat162*>(&fs_acc_packed);
                                    const uint32_t fs_sb = dst_token_idx / STORE_BLOCK_M_L2;
                                    const uint32_t fs_row = dst_token_idx % STORE_BLOCK_M_L2;
                                    const uint8_t* const fs_sb_ptr = fs_comb_base +
                                        static_cast<uint64_t>(fs_n_block_idx) * kFsInt7NBlkStride +
                                        static_cast<uint64_t>(fs_sb) * kFsInt7SbBytes;
                                    const uint8_t* const fs_data_ptr = fs_sb_ptr +
                                        static_cast<uint64_t>(fs_row) * kFsInt7RowBytes + fs_lane16 * 8u;
                                    const uint8_t* const fs_sc_ptr = fs_sb_ptr + kFsInt7DataBytes +
                                        static_cast<uint64_t>(fs_row) * kFsInt7ScaleRowBytes;
                                    const uint32_t fs_group = (fs_lane16 & 8u) ? 1u : 0u;
                                    #pragma unroll
                                    for (uint32_t sl = 0u; sl < kNumTopk; ++ sl) {
                                        const uint2 q = *reinterpret_cast<const uint2*>(fs_data_ptr + sl * BLOCK_N);
                                        const uint32_t sc_pair = *reinterpret_cast<const uint32_t*>(fs_sc_ptr + sl * 4u);
                                        fused_shared_accumulate_int7(q.x, q.y, sc_pair, fs_group, fs_acc);
                                    }
                                    uint8_t* const fs_y_ptr = static_cast<uint8_t*>(y) +
                                        static_cast<uint64_t>(pool_m_idx + m_idx_in_block) *
                                            (kHidden * static_cast<uint32_t>(sizeof(nv_bfloat16))) +
                                        static_cast<uint64_t>(n_idx) * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                        fs_lane16 * 16u;
                                    const auto fs_out = reinterpret_cast<const uint4&>(fs_acc_packed);
                                    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
                                                 :: "l"(fs_y_ptr), "r"(fs_out.x), "r"(fs_out.y),
                                                    "r"(fs_out.z), "r"(fs_out.w), "l"(fs_evict_first) : "memory");
                                } else {
                                    // Routed GEMM2: 8 INT7 bytes + one per-64 BF16 scale.
                                    uint32_t q0, q1, sc_u16;
                                    fused_shared_quantize_int7(packed, q0, q1, sc_u16);
                                    const uint32_t sc_hi = __shfl_xor_sync(0xffffffffu, sc_u16, 8);
                                    const uint32_t fs_sb = dst_token_idx / STORE_BLOCK_M_L2;
                                    const uint32_t fs_row = dst_token_idx % STORE_BLOCK_M_L2;
                                    uint8_t* const fs_sb_ptr = fs_comb_base +
                                        static_cast<uint64_t>(fs_n_block_idx) * kFsInt7NBlkStride +
                                        static_cast<uint64_t>(fs_sb) * kFsInt7SbBytes;
                                    uint8_t* const fs_data_ptr = fs_sb_ptr +
                                        static_cast<uint64_t>(fs_row) * kFsInt7RowBytes +
                                        static_cast<uint64_t>(dst_topk_idx) * BLOCK_N + fs_lane16 * 8u;
                                    auto* const fs_data_peer = sym_buffer.map(fs_data_ptr, dst_rank_idx);
                                    asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;"
                                                 :: "l"(fs_data_peer), "r"(q0), "r"(q1),
                                                    "l"(fs_evict_first) : "memory");
                                    if (fs_lane16 == 0u) {
                                        uint8_t* const fs_sc_ptr = fs_sb_ptr + kFsInt7DataBytes +
                                            static_cast<uint64_t>(fs_row) * kFsInt7ScaleRowBytes +
                                            static_cast<uint64_t>(dst_topk_idx) * 4u;
                                        auto* const fs_sc_peer = sym_buffer.map(fs_sc_ptr, dst_rank_idx);
                                        const uint32_t sc_pair = sc_u16 | (sc_hi << 16);
                                        asm volatile("st.global.u32 [%0], %1;"
                                                     :: "l"(fs_sc_peer), "r"(sc_pair) : "memory");
                                    }
                                }
                            } else {
                                d7_scatter_or_prereduce_row(
                                    task_info.is_shared(), dst_rank_idx, dst_token_idx, dst_topk_idx, n_idx, packed);
                            }
                        }
                    }
                }

                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
#endif

                if constexpr (kD6GroupTiles > 0 and not kD7QueueD6Active) {
                    if (epilogue_warp_idx == 0) {
                        // Consume the previous task's pending group-counter
                        // result: the RTT has overlapped a full tile of work.
                        if (d6_pend_active != 0u) {
                            d6_pend_active = 0u;
                            const uint32_t old_cnt = ptx::exchange(d6_pend_old_cnt, 0);
                            if (((old_cnt + 1u) % kD6GroupTiles) == 0u) {
                                d6_publish_flags(d6_pend_m_base, d6_pend_group,
                                                 d6_pend_valid_m, d6_pend_is_shared);
                            }
                        }
                        const uint32_t group_idx = n_block_idx / kD6GroupTiles;
                        if constexpr (kD6GroupTiles > 1) {
                            if (cute::elect_one_sync()) {
                                auto* const cnt_ptr = buffer.get_d6_local_group_cnt_ptr(
                                    task_info.is_shared(), pool_block_idx, group_idx);
                                // acq_rel RMW: releases this tile's rows and,
                                // for the last arriver, acquires every
                                // predecessor tile's rows written by other SMs
                                // (see the D6 publish-state comment above).
                                d6_pend_old_cnt = ptx::atomic_add_acq_rel_gpu(cnt_ptr, 1u);
                            }
                            d6_pend_m_base = pool_m_idx;
                            d6_pend_group = group_idx;
                            d6_pend_valid_m = valid_m;
                            d6_pend_is_shared = task_info.is_shared() ? 1u : 0u;
                            d6_pend_active = 1u;
                        } else {
                            // Single-tile groups complete with this task: the
                            // epilogue's rows (this warp's program order) and
                            // every other warp's rows (the full-epilogue
                            // barrier above) precede the cumulative release.
                            d6_publish_flags(pool_m_idx, group_idx, valid_m,
                                             task_info.is_shared() ? 1u : 0u);
                        }
                    }
                }
#if DG_MEGA_TILE_PROF
                const auto d7_epi_stage_id = [&]() -> uint32_t {
#if DG_D7_HANDOFF_MODE != 0
                    const bool remote = d7_epi_remote_subtiles != 0u;
                    d7_epi_remote_subtiles = 0u;
                    return remote ? 12u : 24u;
#else
                    return kD7QueueScatterActive ? 24u : 12u;
#endif
                };
                // comm_only (SMEM handoff / HBM queue): the epilogue only converts
                // and hands the tile off locally (stage 24, no NVLink); the comm
                // warps own the Scatter.  epilogue_only: the epilogue scatters (12).
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_epi_s, tile_prof::read_gtimer(),
                                  d7_epi_stage_id(), pool_block_idx);
#endif
            }
        }

        // Consume the final pending group-counter result so the last
        // task's completion is published before the tail Reduce / Workspace
        // Cleanup rendezvous waits on the flags.
        if constexpr (kD6GroupTiles > 0 and not kD7QueueD6Active) {
            if (epilogue_warp_idx == 0 and d6_pend_active != 0u) {
                d6_pend_active = 0u;
                const uint32_t old_cnt = ptx::exchange(d6_pend_old_cnt, 0);
                if (((old_cnt + 1u) % kD6GroupTiles) == 0u) {
                    d6_publish_flags(d6_pend_m_base, d6_pend_group, d6_pend_valid_m, d6_pend_is_shared);
                }
            }
        }

        if constexpr (kFusedSharedInt7) {
            // SMs that received no SharedLinear2 tile still take part once so the
            // peers' fused epilogues can observe every routed store.
            if (not fused_shared_peers_ready) {
                comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                     kEpilogueGridSyncIndex, kFusedSharedInt7BarrierTag>(
                    workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                    [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });
                fused_shared_peers_ready = true;
            }
        }

#if DG_D7_HANDOFF_MODE != 0
        shared_storage.scatter_empty_barriers[epi_scatter_slot].wait(epi_scatter_phase ^ 1u);
        if (epilogue_warp_idx == 0 and lane_idx == 0) {
            ptx::st_shared(&shared_storage.scatter_jobs[epi_scatter_slot], 0u, 0u, 0x80000000u, 0u);
        }
        shared_storage.scatter_full_barriers[epi_scatter_slot].arrive();
        {
            const uint32_t d7_epi_scatter_t =
                kSpatialStreamReduce ? (kFirstDedicatedReduceWarp * 32u) : get_active_dispatch_threads();
            ptx::sync_unaligned(d7_epi_scatter_t + kNumEpilogueThreads, kScatterDrainBarrierIdx);
        }
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

        if constexpr (kFusedSharedInt7) {
            // The fused tail SharedLinear2 epilogue already wrote the final y;
            // dispatch warps only need the cleanup rendezvous.
            ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads,
                                kDispatchWithEpilogueBarrierIdx);
        } else if constexpr (kSpatialStreamReduce) {
            // Axis 6 (D6 All-Warp Reduce, spatial path): the 8 epilogue warps
            // join the tail-start barrier the instant TMEM is freed -- the GEMM
            // and epilogue are complete, so the reusable shared memory is free
            // -- and drain the remaining private tiles through the
            // double-buffered TMA path.  Unconditional: tiles are claimed from
            // the CTA-private cursor, so the tail finishes every tile the
            // Tier-3 gate kept the Middle Reduce warps from streaming.
            spatial_tail_start();
            run_spatial_tail_combine();
            ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        } else if constexpr (kD6ReduceWarps == 16u and not kD2SmidMasked) {
            // Every epilogue bulk store must have finished reading SMEM before
            // the CTA's reusable shared memory turns into the tail ring.
            ptx::tma_store_wait<0>();
            __syncwarp();
            if constexpr (kD6GroupTiles == 0) {
#if DG_MEGA_TILE_PROF
                const uint64_t prof_pre_s = tile_prof::read_gtimer();
#endif
                if constexpr (kPerPeerCombineReady) {
                    // Upstream per-peer readiness: one local grid sync, then SM 0
                    // tags every peer; each tail warp later polls only the ranks
                    // that own its token's experts (no all-rank barrier).
                    uint64_t peer_grid_tag = 0;
                    if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
                        peer_grid_tag = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);
                    comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                        workspace, sm_idx, epilogue_thread_idx,
                        [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
                    );
                    if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
                        ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx),
                                                       epilogue_thread_idx), peer_grid_tag);
                } else {
                    // Split-phase barrier: one local grid sync, then every SM waits on
                    // the rank-local signal (no second grid sync on the critical path).
                    comm::nvlink_barrier_split<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                               kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                        workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                        [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
                    );
                }
                ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#if DG_MEGA_TILE_PROF
                tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                                  prof_rec_idx, prof_t0_ns, prof_pre_s, tile_prof::read_gtimer(),
                                  13u, 0u);
#endif
            } else {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                ptx::sync_unaligned(get_active_dispatch_threads() + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            }
            if constexpr (kAll16Tail) {
                // This CTA's epilogue is complete (GEMM done, every TMA store
                // drained): release the TMA/MMA/CLC warps W04..W07 into the
                // tail without making the epilogue warps wait for them.
                asm volatile("bar.arrive %0, %1;" :: "n"(kAllWarpReduceBarrierIdx),
                             "n"(kNumEpilogueThreads + kNumNonEpilogueThreads) : "memory");
            }
        } else if constexpr (kD6GroupTiles == 0) {
#if DG_MEGA_TILE_PROF
            const uint64_t prof_pre_s = tile_prof::read_gtimer();
#endif
            [[maybe_unused]] uint64_t combine_own_grid_tag = 0;
            if constexpr (kPerPeerCombineReady) {
                // Upstream per-peer readiness: peers' grid indices are loaded
                // before (and overlap with) the local grid sync; after it every
                // local L2 row bound for peer p is written, so SM 0 tags
                // `combine_ready_grid_idx[this rank]` on p with p's own index.
                uint64_t peer_grid_tag = 0;
                if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
                    peer_grid_tag = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);
                comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                    workspace, sm_idx, epilogue_thread_idx,
                    [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
                );
                if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
                    ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx),
                                                   epilogue_thread_idx), peer_grid_tag);
                combine_own_grid_tag = ptx::get_grid_idx() + 1;
            } else {
                // Split-phase barrier (see comm::nvlink_barrier_split): one local
                // grid sync, then every SM waits on the rank-local signal word.
                comm::nvlink_barrier_split<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                           kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                    workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                    [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
                );
            }

            // Release the dispatch warps into the Workspace Cleanup.  With the
            // per-peer readiness this no longer waits for the slowest rank.
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
            // Same {1,2} ladder as the reference producer: a lane may buffer up
            // to 32 * kNumMaxRegistersForBuffer hidden elements, and there is no
            // 4-chunk rung.  Halving the per-lane budget here doubled the number
            // of serialized mbarrier waits / TMA stores per active warp.
            constexpr uint32_t kNumChunks =
                (kNumChunkSlots * kNumEpilogueWarps * (kNumHiddenBytes / kBaseChunks) <= kNumReusableSmemBytes and
                 (kHidden / kBaseChunks) <= 32 * kNumMaxRegistersForBuffer)
                    ? kBaseChunks : 2u;
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
            // Distribute (token, chunk) tasks, not tokens: with an inner chunk
            // loop only num_tokens of the kNumSMs * kNumEpilogueWarps warp slots
            // ever run (256 of 1184 at S=256/rank) and each serialises kNumChunks
            // mbarrier waits + TMA stores.  The reference producer spreads
            // num_tokens * kNumChunks tasks over the same slots, so twice as many
            // warps each do half the serialized tail work.
            // SM-major slot order (upstream `epilogue_warp_idx * kNumSMs + sm_idx`):
            // a partial last round lands on every SM's low warps instead of
            // packing all 8 warps of the first SMs.  At S=256/rank the 512 tasks
            // used to occupy SMs 0..63 only (8 tasks, 8 x 2 x 7 KiB TMA loads in
            // flight per SM, 84 SMs idle); now all 148 SMs carry 3-4 tasks.  Each
            // (token, chunk) is still reduced by exactly one warp in ascending
            // slot order, so the output is bit-identical.
            for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx;
                 token_chunk_idx < num_tokens * kNumChunks;
                 token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
                const uint32_t token_idx = token_chunk_idx / kNumChunks;
                const uint32_t chunk = token_chunk_idx % kNumChunks;
                const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                    static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                    (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                bool is_active_combine_slot = (stored_topk_slot_idx >= 0);
                if constexpr (kD7LocalDedupActive) {
                    const uint32_t exp_rank = (lane_idx < kNumTopk and stored_topk_slot_idx >= 0)
                        ? (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank)
                        : (0xffff0000u | lane_idx);
                    const uint32_t same_r_mask = __match_any_sync(0xffffffffu, exp_rank);
                    if (lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and
                        (kD7QueueScatterActive or exp_rank != sym_buffer.rank_idx)) {
                        is_active_combine_slot = (lane_idx == static_cast<uint32_t>(__ffs(same_r_mask) - 1));
                    }
                }
                const uint32_t total_mask = __ballot_sync(0xffffffff, is_active_combine_slot);

                if constexpr (kPerPeerCombineReady) {
                    // Wait only for the ranks of this token's routed experts
                    // (the shared-expert slot is produced locally).
                    const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
                    const auto* peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                        is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0u);
                    comm::wait_until(
                        [&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == combine_own_grid_tag); },
                        [&]() { printf("DeepGEMM combine peers timeout: rank=%d, token=%u\n", sym_buffer.rank_idx, token_idx); });
                }

                {
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
                        // Generic-proxy reads of this stage must be ordered before the next
                        // async-proxy TMA overwrite of the same buffer (official WAR fence).
                        cutlass::arch::fence_view_async_shared();
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
                // SM-major token order (upstream combine mapping): with the old
                // SM-packed `sm_idx * kActiveCombineWarps + warp` order, 256
                // tokens/rank landed on SMs 0..31 only.  One warp still owns a
                // whole token, so the per-token slot order (and the output bits)
                // are unchanged.
                for (uint32_t token_idx = epilogue_warp_idx * kNumSMs + sm_idx;
                     token_idx < num_tokens;
                     token_idx += kNumSMs * kActiveCombineWarps) {
                    const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                        static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                        (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                    bool is_active_combine_slot = (stored_topk_slot_idx >= 0);
                    if constexpr (kD7LocalDedupActive) {
                        // Queue dedup delivers one merged row per (token, expert rank) into the leader slot.
                        const uint32_t exp_rank = (lane_idx < kNumTopk and stored_topk_slot_idx >= 0)
                            ? (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank)
                            : (0xffff0000u | lane_idx);
                        const uint32_t same_r_mask = __match_any_sync(0xffffffffu, exp_rank);
                        if (lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and
                            (kD7QueueScatterActive or exp_rank != sym_buffer.rank_idx)) {
                            is_active_combine_slot = (lane_idx == static_cast<uint32_t>(__ffs(same_r_mask) - 1));
                        }
                    }
                    const uint32_t total_mask = __ballot_sync(0xffffffff, is_active_combine_slot);

                    for (uint32_t group_idx = 0; group_idx < kNumGroups; ++ group_idx) {
                        const uint32_t group_byte_offset = group_idx * kGroupBytes;

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
                                            if (not d6_flags_ready(token_idx, slot_idx, group_byte_offset, kGroupBytes, d6_cur_epoch)) {
                                                const auto start_clock = clock64();
                                                do {
                                                    if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                                                        printf("DeepGEMM D6 FineCombine timeout: rank=%u sm=%u warp=%u tok=%u grp=%u slot=%u ep=%u\n",
                                                               sym_buffer.rank_idx, sm_idx, epilogue_warp_idx,
                                                               token_idx, group_idx, slot_idx, static_cast<uint32_t>(d6_cur_epoch));
                                                        DG_DEVICE_ASSERT(false and "D6 FineCombine readiness timeout");
                                                    }
                                                    asm volatile("nanosleep.u32 32;");
                                                } while (not d6_flags_ready(token_idx, slot_idx, group_byte_offset, kGroupBytes, d6_cur_epoch));
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

    // Axis 6 (D6 All-Warp Reduce, spatial path): the four MMA non-epilogue
    // warps -- W04 TMA-A, W05 TMA-B, W06 UMMA, W07 CLC -- join the CTA-wide
    // TMA tail Reduce through the tail-start barrier.  W06/W07 only run on the
    // leader CTA, so on every odd SM they arrive here at t=0, before SM0 bumps
    // d6_epoch during Opening; the barrier also waits for W04/W05 (which drain
    // scheduler.get_next_task(), published only after Opening) and for the
    // epilogue warps (GEMM complete), so the tail's acquire read of d6_epoch
    // always sees the bump.  W04 lane 0 raises the tail-go flag.
    if constexpr (kSpatialStreamReduce) {
        if (warp_idx >= kNumDispatchWarps and warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
            spatial_tail_start();
            run_spatial_tail_combine();
        }
    }

    // ========================================================================
    // All-16-warp tail Combine / Reduce (RW = 0 path, WEAVE_COMBINE_REDUCE_WARPS
    // = 16): W00..W03 (after the Workspace Cleanup), W04..W07 (TMA / MMA /
    // CLC, released by the epilogue's bar.arrive once this CTA's GEMM is done)
    // and W08..W15 (right after the per-peer readiness tag) all reduce.
    //   * Task = (token, hidden sub-chunk); tile t -> token t % S, chunk t / S.
    //   * Zero-CAS work split: SM s owns the private set { s + k * num_sms };
    //     inside the CTA every warp claims k from one SMEM cursor
    //     (atom.shared.add, each k returned to exactly one warp), so a warp
    //     group that arrives late (W00..W03 after cleanup) simply claims less.
    //   * Readiness: per-peer grid-index tags (only the ranks owning the
    //     token's experts are polled), or the D6 group flags when
    //     WEAVE_COMBINE_SYNC_CHUNK > 0.
    //   * Loads: kAll16Stages-deep TMA -> SMEM ring per warp, so up to
    //     kAll16Stages top-k sources are in flight instead of 2.
    // Sources are summed in ascending combine-slot order in FP32 with one final
    // bf16 rounding, bit-identical to the 8-warp epilogue combine.
    // ========================================================================
    if constexpr (kAll16Tail) {
        constexpr uint32_t kAllReduceWarps = kNumThreads / 32u;
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kMinRoleRegs = cute::min(cute::min(kNumDispatchRegisters, kNumNonEpilogueRegisters),
                                                    kNumEpilogueRegisters);
        constexpr auto all16_fits = [](const uint32_t sub, const uint32_t stages) {
            return kNumHiddenBytes % sub == 0u and
                   kAllReduceWarps * ((stages + 1u) * sub + stages * static_cast<uint32_t>(sizeof(Barrier))) <=
                       kNumReusableSmemBytes;
        };
        // Compile-time (maximum) sub-chunk: sizes the SMEM ring and registers.
        constexpr uint32_t kSubChunkBytes =
            (DG_ALL16_SUB_BYTES == 1024 or DG_ALL16_SUB_BYTES == 2048)
                ? static_cast<uint32_t>(DG_ALL16_SUB_BYTES)
                : ((kMinRoleRegs >= 80u and all16_fits(2048u, 4u)) ? 2048u : 1024u);
        constexpr uint32_t kAll16Stages = all16_fits(kSubChunkBytes, 4u) ? 4u : 2u;
        DG_STATIC_ASSERT(all16_fits(kSubChunkBytes, kAll16Stages),
                         "All-16-warp tail Reduce does not fit in the reusable shared memory");
        constexpr uint32_t kVecsPerLane = kSubChunkBytes / (32u * sizeof(uint4));
        constexpr uint32_t kElemsPerVec = sizeof(uint4) / sizeof(nv_bfloat162);
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u,
                         "Top-k + shared must fit in a single warp");
        // Runtime sub-chunk: 2 KiB halves the claim count, but only while every
        // tail warp still gets a tile.  With fewer private tiles per SM than
        // warps (small S, e.g. Qwen3.8-Max S_rank=256: 256 x 8 / 148 = 13.8 < 16)
        // part of the 16 warps would idle, so the tail halves the sub-chunk to
        // 1 KiB (27.7 tiles per SM) inside the same 2 KiB SMEM slots.
        const bool all16_half_sub = (DG_ALL16_SUB_BYTES == 0) and kSubChunkBytes == 2048u and
            (num_tokens * (kNumHiddenBytes / kSubChunkBytes) + kNumSMs - 1u) / kNumSMs < kAllReduceWarps;
        const uint32_t sub_chunk_bytes = all16_half_sub ? (kSubChunkBytes / 2u) : kSubChunkBytes;
        const uint32_t num_sub_chunks = kNumHiddenBytes / sub_chunk_bytes;
        const uint32_t num_vecs_per_lane = all16_half_sub ? (kVecsPerLane / 2u) : kVecsPerLane;

        // W04..W07 must not touch the reusable SMEM before this CTA's epilogue
        // (and therefore every MMA / TMA load of the GEMM) has completed.
        if (warp_idx >= kNumDispatchWarps and warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
            asm volatile("bar.sync %0, %1;" :: "n"(kAllWarpReduceBarrierIdx),
                         "n"(kNumEpilogueThreads + kNumNonEpilogueThreads) : "memory");
        }
#if DG_MEGA_TILE_PROF
        const uint64_t prof_tail_join_s = tile_prof::read_gtimer();
#endif
        const uint16_t all_d6_epoch = (kD6GroupTiles > 0u) ?
            static_cast<uint16_t>(ptx::ld_acq(buffer.get_d6_epoch_ptr())) : static_cast<uint16_t>(0u);
        [[maybe_unused]] const uint64_t combine_own_grid_tag = ptx::get_grid_idx() + 1;

        // Per-warp SMEM: kAll16Stages load buffers + 1 store buffer, and the
        // warp's own kAll16Stages mbarriers (self-initialised, no CTA sync).
        uint8_t* const warp_smem = smem_buffer + warp_idx * (kAll16Stages + 1u) * kSubChunkBytes;
        auto* const all16_barriers = reinterpret_cast<Barrier*>(
            smem_buffer + kAllReduceWarps * (kAll16Stages + 1u) * kSubChunkBytes) + warp_idx * kAll16Stages;
        if (lane_idx < kAll16Stages)
            all16_barriers[lane_idx].init(1);
        cutlass::arch::fence_barrier_init();
        __syncwarp();
        const auto load_buffer = [&](const uint32_t st) {
            return reinterpret_cast<uint4*>(warp_smem + st * kSubChunkBytes);
        };
        auto* const store_buffer = reinterpret_cast<uint4*>(warp_smem + kAll16Stages * kSubChunkBytes);

        uint32_t num_issued = 0u, num_waited = 0u;   // ring position, monotonic across tiles
        const uint32_t total_tiles = num_tokens * num_sub_chunks;
        const uint32_t num_private_tiles = get_num_private_reduce_tiles(total_tiles);
#if DG_MEGA_TILE_PROF
        tile_prof::record(cumulative_local_expert_recv_stats, sm_idx, warp_idx, lane_idx,
                          prof_rec_idx, prof_t0_ns, prof_tail_join_s, tile_prof::read_gtimer(), 13u, 1u);
#endif
        while (true) {
            uint32_t claimed = 0u;
            if (lane_idx == 0u)
                claimed = ptx::atomic_add_shared(spatial_reduce_ctrl + 0, 1u);
            claimed = __shfl_sync(0xffffffffu, claimed, 0);
            if (claimed >= num_private_tiles)
                break;
            const uint32_t tile_idx = sm_idx + claimed * kNumSMs;
#if DG_MEGA_TILE_PROF
            uint64_t prof_red_s = tile_prof::read_gtimer();
#endif
            const uint32_t token_idx = tile_idx % num_tokens;
            const uint32_t sub_idx = tile_idx / num_tokens;
            const uint32_t sub_byte_offset = sub_idx * sub_chunk_bytes;

            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            bool is_active_combine_slot = (stored_topk_slot_idx >= 0);
            if constexpr (kD7LocalDedupActive) {
                const uint32_t exp_rank = (lane_idx < kNumTopk and stored_topk_slot_idx >= 0)
                    ? (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank)
                    : (0xffff0000u | lane_idx);
                const uint32_t same_r_mask = __match_any_sync(0xffffffffu, exp_rank);
                if (lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and
                    (kD7QueueScatterActive or exp_rank != sym_buffer.rank_idx)) {
                    is_active_combine_slot = (lane_idx == static_cast<uint32_t>(__ffs(same_r_mask) - 1));
                }
            }
            uint32_t mask = __ballot_sync(0xffffffff, is_active_combine_slot);

            if constexpr (kPerPeerCombineReady) {
                const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
                const auto* peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                    is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0u);
                comm::wait_until(
                    [&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == combine_own_grid_tag); },
                    [&]() { printf("DeepGEMM all-16 combine peers timeout: rank=%d, token=%u\n", sym_buffer.rank_idx, token_idx); });
            }
#if DG_MEGA_TILE_PROF
            prof_red_s = tile_prof::read_gtimer();
#endif
            const auto issue_next = [&]() {
                if (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1u << slot_idx;
                    const uint32_t st = num_issued % kAll16Stages;
                    if (cute::elect_one_sync()) {
                        if constexpr (kD6GroupTiles > 0u) {
                            if (not d6_flags_ready(token_idx, slot_idx, sub_byte_offset, sub_chunk_bytes, all_d6_epoch)) {
                                const auto start_clock = clock64();
                                do {
                                    if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                                        DG_DEVICE_ASSERT(false and "All-16 tail Reduce readiness timeout");
                                    }
                                    asm volatile("nanosleep.u32 32;");
                                } while (not d6_flags_ready(token_idx, slot_idx, sub_byte_offset, sub_chunk_bytes, all_d6_epoch));
                            }
                        }
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            sub_byte_offset);
                        ptx::tma_load_1d(load_buffer(st), src_ptr, &all16_barriers[st], sub_chunk_bytes);
                        ptx::mbarrier_arrive_and_set_tx(&all16_barriers[st], sub_chunk_bytes);
                    }
                    __syncwarp();
                    ++num_issued;
                }
            };
            #pragma unroll
            for (uint32_t i = 0; i < kAll16Stages; ++ i)
                issue_next();

            float2 reduced[kVecsPerLane * kElemsPerVec] = {};
            while (num_waited < num_issued) {
                const uint32_t st = num_waited % kAll16Stages;
                all16_barriers[st].wait((num_waited / kAll16Stages) & 1u);
                #pragma unroll
                for (uint32_t j = 0; j < kVecsPerLane; ++ j) {
                    if (j >= num_vecs_per_lane)
                        break;
                    const uint4 vec_values = load_buffer(st)[j * 32u + lane_idx];
                    const auto* bf16_values = reinterpret_cast<const nv_bfloat162*>(&vec_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kElemsPerVec; ++ l)
                        ptx::accumulate(reduced[j * kElemsPerVec + l], bf16_values[l]);
                }
                ++num_waited;
                // Generic-proxy reads of this stage are ordered before the next
                // async-proxy TMA overwrite of the same buffer (official WAR fence).
                cutlass::arch::fence_view_async_shared();
                __syncwarp();
                issue_next();
            }

            #pragma unroll
            for (uint32_t j = 0; j < kVecsPerLane; ++ j) {
                if (j >= num_vecs_per_lane)
                    break;
                uint4 casted;
                auto* casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kElemsPerVec; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kElemsPerVec + l]);
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(store_buffer + j * 32u + lane_idx, casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();
            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + sub_byte_offset),
                    store_buffer, sub_chunk_bytes);
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
        if constexpr (kAll16DeferCleanWait) {
            if (sm_idx == 0 and threadIdx.x == 0)
                comm::nvlink_barrier_deferred_wait<kNumRanks, kAfterWorkspaceCleanBarrierTag>(workspace, sym_buffer);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
