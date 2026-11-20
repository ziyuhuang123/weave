#pragma once

// 1 = enable 2-die (locality-domain) task partitioning + cross-domain stealing.
// Host-generated JIT sources define this explicitly; other users default to off.
#ifndef WEAVE_LOCALITY_DOMAINS
#define WEAVE_LOCALITY_DOMAINS 0
#endif

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



#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// ============================================================================
// Unified 7-Axis Macro Normalization (DG_TUNE_* <-> DG_ISOLATED_* / DG_D3_*)
// ============================================================================
#if defined(DG_TUNE_D1_G) && !defined(DG_ISOLATED_D1_G)
#define DG_ISOLATED_D1_G DG_TUNE_D1_G
#endif
#ifndef DG_ISOLATED_D1_G
#define DG_ISOLATED_D1_G 0
#endif
#ifndef DG_TUNE_D1_G
#define DG_TUNE_D1_G DG_ISOLATED_D1_G
#endif

#if defined(DG_TUNE_D3_SCHED_STAGES) && !defined(DG_D3_SCHED_STAGES)
#define DG_D3_SCHED_STAGES DG_TUNE_D3_SCHED_STAGES
#endif
#ifndef DG_D3_SCHED_STAGES
#define DG_D3_SCHED_STAGES 2u
#endif

#if defined(DG_TUNE_D3_EXPERT_STEP_MODE) && !defined(DG_D3_EXPERT_STEP_MODE)
#define DG_D3_EXPERT_STEP_MODE DG_TUNE_D3_EXPERT_STEP_MODE
#endif
#ifndef DG_D3_EXPERT_STEP_MODE
#define DG_D3_EXPERT_STEP_MODE 0
#endif

#if defined(DG_TUNE_D4_EPW) && !defined(DG_ISOLATED_D4_EPW)
#define DG_ISOLATED_D4_EPW DG_TUNE_D4_EPW
#endif
#ifndef DG_ISOLATED_D4_EPW
#define DG_ISOLATED_D4_EPW 0
#endif
#ifndef DG_TUNE_D4_EPW
#define DG_TUNE_D4_EPW DG_ISOLATED_D4_EPW
#endif

#if defined(DG_TUNE_D4_STATIC) && !defined(DG_ISOLATED_D4_STATIC)
#define DG_ISOLATED_D4_STATIC DG_TUNE_D4_STATIC
#endif
#ifndef DG_ISOLATED_D4_STATIC
#define DG_ISOLATED_D4_STATIC 0
#endif
#ifndef DG_TUNE_D4_STATIC
#define DG_TUNE_D4_STATIC DG_ISOLATED_D4_STATIC
#endif

#if defined(DG_TUNE_D4_READYQ) && !defined(DG_ISOLATED_D4_READYQ)
#define DG_ISOLATED_D4_READYQ DG_TUNE_D4_READYQ
#endif
#ifndef DG_ISOLATED_D4_READYQ
#define DG_ISOLATED_D4_READYQ 0
#endif

#if defined(DG_TUNE_D4_MODE) && !defined(DG_ISOLATED_D4_MODE)
#define DG_ISOLATED_D4_MODE DG_TUNE_D4_MODE
#endif
#ifndef DG_ISOLATED_D4_MODE
#define DG_ISOLATED_D4_MODE 0
#endif

#define DG_ISOLATED_D4_LUT_ACTIVE ((DG_ISOLATED_D4_EPW > 0) || (DG_ISOLATED_D4_STATIC > 0))

#if defined(DG_TUNE_D5_DEDUP) && !defined(DG_ISOLATED_D5_DEDUP)
#define DG_ISOLATED_D5_DEDUP DG_TUNE_D5_DEDUP
#endif
#ifndef DG_ISOLATED_D5_DEDUP
#define DG_ISOLATED_D5_DEDUP 0
#endif
#ifndef DG_TUNE_D5_DEDUP
#define DG_TUNE_D5_DEDUP DG_ISOLATED_D5_DEDUP
#endif

#if defined(DG_ISOLATED_D5_SCHED) && !defined(DG_TUNE_D5_SCHED)
#define DG_TUNE_D5_SCHED DG_ISOLATED_D5_SCHED
#endif
#ifndef DG_TUNE_D5_SCHED
#define DG_TUNE_D5_SCHED ((DG_ISOLATED_D5_DEDUP == 2) ? 1 : 0)
#endif

#if DG_ISOLATED_D5_DEDUP && (defined(__CUDACC__) or defined(__CLION_IDE__))
namespace deep_gemm::d5 {

static constexpr uint32_t kMaxBitmapWordsPerTier = 256u;

CUTLASS_DEVICE void st_release_gpu(uint32_t* ptr, const uint32_t& val) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}

CUTLASS_DEVICE void red_or_rel_gpu_u32(uint32_t* ptr, const uint32_t& mask) {
    asm volatile("red.release.gpu.global.or.b32 [%0], %1;" :: "l"(ptr), "r"(mask) : "memory");
}

CUTLASS_DEVICE uint32_t atomic_add_acq_rel_gpu(uint32_t* ptr, const uint32_t& val) {
    uint32_t ret;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val) : "memory");
    return ret;
}

CUTLASS_DEVICE uint32_t atomic_and_acq_rel_gpu_u32(uint32_t* ptr, const uint32_t& mask) {
    uint32_t ret;
    asm volatile("atom.acq_rel.gpu.global.and.b32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(mask) : "memory");
    return ret;
}

CUTLASS_DEVICE uint32_t* key_count_ptr(const layout::Workspace& ws, const uint32_t& r, const uint32_t& t) {
    return ws.get_src_token_topk_idx_ptr(0u, r, ws.num_max_tokens_per_rank + t);
}

CUTLASS_DEVICE uint32_t* key_member_ptr(const layout::Workspace& ws, const uint32_t& r, const uint32_t& t, const uint32_t& m) {
    return ws.get_src_token_topk_idx_ptr(1u + m, r, ws.num_max_tokens_per_rank + t);
}

CUTLASS_DEVICE uint32_t* local_key_ptr(const layout::Workspace& ws, const uint32_t& idx) {
    return ws.get_src_token_topk_idx_ptr(11u, 0u, ws.num_max_tokens_per_rank + idx);
}

CUTLASS_DEVICE uint32_t* remote_key_ptr(const layout::Workspace& ws, const uint32_t& idx) {
    return ws.get_src_token_topk_idx_ptr(12u, 0u, ws.num_max_tokens_per_rank + idx);
}

CUTLASS_DEVICE uint32_t* ctrl_base_ptr(const layout::Workspace& ws) {
    return ws.get_src_token_topk_idx_ptr(13u, 0u, ws.num_max_tokens_per_rank);
}

CUTLASS_DEVICE uint32_t* num_local_keys_ptr(const layout::Workspace& ws)  { return ctrl_base_ptr(ws) + 0u; }
CUTLASS_DEVICE uint32_t* num_remote_keys_ptr(const layout::Workspace& ws) { return ctrl_base_ptr(ws) + 32u; }
CUTLASS_DEVICE uint32_t* remote_cursor_ptr(const layout::Workspace& ws)   { return ctrl_base_ptr(ws) + 64u; }
CUTLASS_DEVICE uint32_t* plan_ready_ptr(const layout::Workspace& ws)      { return ctrl_base_ptr(ws) + 96u; }
CUTLASS_DEVICE uint32_t* tier0_count_ptr(const layout::Workspace& ws)       { return ctrl_base_ptr(ws) + 128u; }
CUTLASS_DEVICE uint32_t* tier1_count_ptr(const layout::Workspace& ws)       { return ctrl_base_ptr(ws) + 129u; }
CUTLASS_DEVICE uint32_t* tier2_count_ptr(const layout::Workspace& ws)       { return ctrl_base_ptr(ws) + 130u; }
CUTLASS_DEVICE uint32_t* total_loc_keys_ptr(const layout::Workspace& ws)    { return ctrl_base_ptr(ws) + 131u; }
CUTLASS_DEVICE uint32_t* total_rem_keys_ptr(const layout::Workspace& ws)    { return ctrl_base_ptr(ws) + 132u; }
CUTLASS_DEVICE uint32_t* total_members_ptr(const layout::Workspace& ws)     { return ctrl_base_ptr(ws) + 133u; }
CUTLASS_DEVICE uint32_t* fanout_sum_ptr(const layout::Workspace& ws)        { return ctrl_base_ptr(ws) + 160u; }

template <uint32_t kNumSMs, uint32_t kNumDispatchThreads, uint32_t kDispatchBarrierIdx, uint32_t kGridSyncIndex = 2>
CUTLASS_DEVICE void fast_grid_sync(const layout::Workspace& ws, const uint32_t& sm_idx, const uint32_t& thread_idx) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    if (thread_idx == 0u) {
        uint32_t* count_ptr = ws.get_grid_sync_count_ptr<kGridSyncIndex>();
        const uint32_t old_val = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0u ? (kFinishSumTag - (kNumSMs - 1u)) : 1u);
        while (((ptx::ld_acq(count_ptr) ^ old_val) & kFinishSumTag) == 0u) {}
    }
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
}

CUTLASS_DEVICE uint32_t* tier_bitmap_word_ptr(const layout::Workspace& ws, const uint32_t& tier, const uint32_t& word_idx) {
    return ctrl_base_ptr(ws) + 192u + tier * kMaxBitmapWordsPerTier + word_idx;
}

template <uint32_t kArrIdx>
CUTLASS_DEVICE uint32_t* tile_arr_ptr(const layout::Workspace& ws, const uint32_t& idx) {
    return ws.get_src_token_topk_idx_ptr(14u + kArrIdx, 0u, ws.num_max_tokens_per_rank + idx);
}

CUTLASS_DEVICE uint32_t* tile_expected_ptr(const layout::Workspace& ws, const uint32_t& tile_id) { return tile_arr_ptr<0u>(ws, tile_id); }
CUTLASS_DEVICE uint32_t* tile_ready_ptr(const layout::Workspace& ws, const uint32_t& tile_id)    { return tile_arr_ptr<1u>(ws, tile_id); }
CUTLASS_DEVICE uint32_t* tile_flags_ptr(const layout::Workspace& ws, const uint32_t& tile_id)    { return tile_arr_ptr<2u>(ws, tile_id); }
CUTLASS_DEVICE uint32_t* sched_m_block_ptr(const layout::Workspace& ws, const uint32_t& seq_idx) { return tile_arr_ptr<3u>(ws, seq_idx); }

CUTLASS_DEVICE uint32_t* telemetry_ptr(const layout::Workspace& ws) {
    return math::advance_ptr<uint32_t>(ws.base, 64u);
}

} // namespace deep_gemm::d5
#endif

namespace deep_gemm::sched {

#if defined(__CUDACC__) or defined(__CLION_IDE__)
namespace d1_ptx {
CUTLASS_DEVICE uint32_t atomic_add_sys_u32(uint32_t* ptr, const uint32_t& val) {
    uint32_t ret;
    asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val));
    return ret;
}
CUTLASS_DEVICE uint32_t atomic_add_rel_sys_u32(uint32_t* ptr, const uint32_t& val) {
    uint32_t ret;
    asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val) : "memory");
    return ret;
}
CUTLASS_DEVICE uint32_t atomic_add_acq_rel_sys_u32(uint32_t* ptr, const uint32_t& val) {
    uint32_t ret;
    asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val) : "memory");
    return ret;
}
CUTLASS_DEVICE uint32_t atomic_add_acq_rel_gpu_u32(uint32_t* ptr, const uint32_t& val) {
    uint32_t ret;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(val) : "memory");
    return ret;
}
CUTLASS_DEVICE void st_relaxed_sys_u32(uint32_t* ptr, const uint32_t& val) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(val));
}
CUTLASS_DEVICE void st_rel_sys_u32(uint32_t* ptr, const uint32_t& val) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}
CUTLASS_DEVICE void st_rel_sys_u64(uint64_t* ptr, const uint64_t& val) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(val) : "memory");
}
CUTLASS_DEVICE void st_rel_gpu_u64(uint64_t* ptr, const uint64_t& val) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(ptr), "l"(val) : "memory");
}
CUTLASS_DEVICE void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}
} // namespace d1_ptx
#endif

CUTLASS_HOST_DEVICE constexpr int get_min_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

CUTLASS_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden,
    const int& block_n = 128) {
    if (num_total_m_blocks <= 0) return 0;

    const int num_clusters = num_sms / 2;
    const int num_l1_n_clusters = (intermediate_hidden * 2 / block_n) / 2;
    const int num_l2_n_clusters = (hidden / block_n) / 2;

    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_total_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_l1_warmup_waves = cute::min(
        get_min_l1_warmup_waves(num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters),
        num_total_l1_waves);

    const int num_l1_warmup_clusters = cute::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(num_clusters, num_l1_n_clusters) +
        math::constexpr_ceil_div(num_clusters, num_l2_n_clusters) + 1;

    return cute::min(
        num_total_m_blocks,
        num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

enum class BlockPhase : uint32_t {
    Linear1,
    Linear2,
    SharedLinear1,
    SharedLinear2,
    None,
};

template <bool kHasShared>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    TaskInfo() = default;

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& n_cluster_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_cluster_idx(n_cluster_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m),
        shape_n(shape_n),
        shape_k(shape_k) {}

    CUTLASS_HOST_DEVICE bool is_valid() const {
        return block_phase != BlockPhase::None;
    }

    CUTLASS_HOST_DEVICE bool is_shared() const {
        return kHasShared and (block_phase == BlockPhase::SharedLinear1 or block_phase == BlockPhase::SharedLinear2);
    }

    CUTLASS_HOST_DEVICE uint32_t get_umma_aligned_valid_m() const {
        return cute::max(math::constexpr_align(valid_m, 16u), 16u);
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

template <uint32_t BLOCK_M>
CUTLASS_HOST_DEVICE constexpr uint32_t select_expert_bm(const uint32_t& num_tokens) {
#if defined(DG_D3_EXPERT_STEP_MODE) && (DG_D3_EXPERT_STEP_MODE > 0)
    if constexpr (BLOCK_M <= 32u) {
        return BLOCK_M;
    } else {
        if (num_tokens == 0u)
            return BLOCK_M;
        constexpr uint32_t kSlack =
        #ifdef DG_D3_BM_PAD_SLACK
            DG_D3_BM_PAD_SLACK;
        #else
            0u;
        #endif
        const uint32_t min_padded = math::constexpr_ceil_div(num_tokens, 32u) * 32u;
        uint32_t best_bm = 32u;
        if constexpr (BLOCK_M >= 64u) {
            const uint32_t p64 = math::constexpr_ceil_div(num_tokens, 64u) * 64u;
            if (p64 <= min_padded + kSlack) best_bm = 64u;
        }
        if constexpr (BLOCK_M >= 128u) {
            const uint32_t p128 = math::constexpr_ceil_div(num_tokens, 128u) * 128u;
            if (p128 <= min_padded + kSlack) best_bm = 128u;
        }
        if constexpr (BLOCK_M >= 256u) {
            const uint32_t p256 = math::constexpr_ceil_div(num_tokens, 256u) * 256u;
            if (p256 <= min_padded + kSlack) best_bm = 256u;
        }
        return best_bm;
    }
#else
    (void)num_tokens;
    return BLOCK_M;
#endif
}

#if defined(__CUDACC__) or defined(__CLION_IDE__)

template <
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
    uint32_t kNumExpertsPerRank,
    uint32_t kNumSMs, uint32_t kNumRanks,
    uint32_t kNumRingBlocks,
    uint32_t kNumSharedExperts = 0,
    uint32_t kNumLocalityDomains = (WEAVE_LOCALITY_DOMAINS ? 2u : 1u),
    uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
    uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
    uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
    uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
    uint32_t kNumL2Clusters = kNumL2BlockNs / 2
>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = (kNumSharedExperts > 0);
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid L1 shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid L2 shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid L2 shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == 2, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumL1Clusters % kNumLocalityDomains == 0 and
                     kNumL2Clusters % kNumLocalityDomains == 0,
                     "Each locality domain must take whole task clusters");

    const layout::Workspace& workspace;

    static constexpr uint32_t kNumScheduleStages = DG_D3_SCHED_STAGES;
    DG_STATIC_ASSERT(kNumScheduleStages == 2 or kNumScheduleStages == 3, "Invalid schedule stages");
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    #if DG_ISOLATED_D5_DEDUP == 1
    uint32_t stored_lane_block_offset[kNumExpertsPerLane] = {};
    #endif
    uint32_t num_total_m_blocks = 0;
    uint32_t num_sched_l1_waves = 0;

    // Locality-domain affinity. Each die owns a contiguous half of the N-cluster
    // range, so its L2 only sees half of the expert weight columns. When that half
    // is drained we steal from the peer domain, but only if at least
    // kNumMinStealWaves full CTA-pair waves of work exist (avoid cross-die latency
    // on small problems).
    static constexpr uint32_t kNumMinStealWaves = 2;
    uint32_t sm_locality_domain_idx = 0;

    CUTLASS_DEVICE
    MegaMoEScheduler(const layout::Workspace& workspace,
                     Barrier* task_info_full_barriers,
                     Barrier* task_info_empty_barriers,
                     task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        if constexpr (kNumScheduleStages == 2) {
            sched_stage_idx ^= 1;
            sched_phase ^= sched_stage_idx == 0;
        } else {
            sched_stage_idx = (sched_stage_idx + 1 == kNumScheduleStages) ? 0u : (sched_stage_idx + 1u);
            sched_phase ^= (sched_stage_idx == 0);
        }
    }

    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    CUTLASS_DEVICE void release_task_info() const {
        if constexpr (kNumScheduleStages == 2) {
            task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
        } else {
            const uint32_t prev_stage = (sched_stage_idx == 0u) ? (kNumScheduleStages - 1u) : (sched_stage_idx - 1u);
            task_info_empty_barriers[prev_stage].arrive(0u);
        }
    }

    CUTLASS_DEVICE void ack_task_info() const {
        __syncwarp();
        if (cute::elect_one_sync())
            release_task_info();
        __syncwarp();
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i == expert_idx / 32)
                valid_value = stored_num_tokens_per_expert[i];
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens_divergent(const uint32_t& expert_idx) const {
        const uint32_t src_lane = expert_idx & 31u;
        const uint32_t src_slice = expert_idx >> 5;
        uint32_t res = 0;
        #pragma unroll
        for (uint32_t j = 0; j < kNumExpertsPerLane; ++ j) {
            const uint32_t v = __shfl_sync(0xffffffffu, stored_num_tokens_per_expert[j], src_lane);
            if (j == src_slice)
                res = v;
        }
        return res;
    }

    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        #if DG_ISOLATED_D5_DEDUP == 1
        if (expert_idx >= kNumExpertsPerRank)
            return num_total_m_blocks;
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_lane_block_offset[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
        #else
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t expert_bm = select_expert_bm<BLOCK_M>(num_tokens);
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(num_tokens, expert_bm);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
        #endif
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            uint32_t tok = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
                tok = static_cast<uint32_t>(value);
#if DG_ISOLATED_D1_G > 0
                const uint64_t bmap = ptx::ld_acq_gpu(workspace.get_d1_ready_bitmap_word_ptr(expert_idx, 0));
                const uint32_t ready_groups = static_cast<uint32_t>(__popcll(bmap));
                tok = cute::min(tok, ready_groups * static_cast<uint32_t>(DG_ISOLATED_D1_G) * BLOCK_M);
#endif
            }
            stored_num_tokens_per_expert[i] = tok;
        }
        __syncwarp();

        #if DG_ISOLATED_D5_DEDUP == 1
        const uint32_t lane_idx = ptx::get_lane_idx();
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t expert_bm = select_expert_bm<BLOCK_M>(num_tokens);
            const uint32_t num_m_blocks = (expert_idx < kNumExpertsPerRank) ?
                math::ceil_div(num_tokens, expert_bm) : 0u;
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            stored_lane_block_offset[i] = block_offset + inclusive_num_m_blocks - num_m_blocks;
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        num_total_m_blocks = block_offset;
        #else
        num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank);
        #endif
        const uint32_t num_clusters = kNumSMs / 2;
        const uint32_t min_l1_warmup_waves = get_min_l1_warmup_waves(
            num_total_m_blocks, num_clusters, kNumL1Clusters, kNumL2Clusters);
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_m_blocks * kNumL1Clusters, num_clusters);
        num_sched_l1_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

#if DG_ISOLATED_D4_LUT_ACTIVE
    static CUTLASS_DEVICE uint32_t get_slot_divergent(
            const uint32_t (&arr)[kNumExpertsPerLane],
            const uint32_t& expert_idx) {
        const uint32_t src_lane = expert_idx & 31u;
        const uint32_t src_slice = expert_idx >> 5;
        uint32_t res = 0;
        #pragma unroll
        for (uint32_t j = 0; j < kNumExpertsPerLane; ++ j) {
            const uint32_t v = __shfl_sync(0xffffffffu, arr[j], src_lane);
            if (j == src_slice)
                res = v;
        }
        return res;
    }

    CUTLASS_DEVICE void build_d4_schedule_plan(
            const uint32_t& rank_idx,
            uint32_t (&d4_slot_sched)[kNumExpertsPerLane],
            uint32_t (&d4_slot_phys)[kNumExpertsPerLane],
            uint16_t* sched_cum_blocks,
            uint8_t* sched_expert_order) {
        (void)sched_cum_blocks;
        (void)sched_expert_order;
        DG_STATIC_ASSERT(kNumExpertsPerRank <= 256u, "D4 supports up to 256 local experts");
        const uint32_t lane_idx = ptx::get_lane_idx();

        constexpr uint32_t kRawEpw = static_cast<uint32_t>(DG_ISOLATED_D4_EPW);
        constexpr bool kUseInlineStatic = (kRawEpw >= 100u);
        constexpr uint32_t kCleanEpw = kUseInlineStatic ? (kRawEpw - 100u) : kRawEpw;
        constexpr uint32_t kEffEpw = (kCleanEpw < kNumExpertsPerRank) ? kCleanEpw : kNumExpertsPerRank;

        if ((DG_ISOLATED_D4_STATIC == 0 and not kUseInlineStatic and kEffEpw <= 1u) or (num_total_m_blocks > kNumRingBlocks)) {
            uint32_t running_phys = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                const uint32_t e = i * 32u + lane_idx;
                const uint32_t tok = (e < kNumExpertsPerRank) ? stored_num_tokens_per_expert[i] : 0u;
                const uint32_t blk = math::ceil_div(tok, select_expert_bm<BLOCK_M>(tok));
                const uint32_t incl = math::warp_inclusive_sum(blk, lane_idx);
                const uint32_t off = running_phys + incl - blk;
                running_phys += ptx::exchange(incl, 31);
                d4_slot_sched[i] = ((e & 0xffu) << 20) | ((blk & 0xffu) << 12) | (off & 0xfffu);
                d4_slot_phys[i] = (tok << 12) | (off & 0xfffu);
            }
            return;
        }

        #if defined(DG_ISOLATED_D4_HAS_STATIC_TABLE)
        if constexpr (DG_ISOLATED_D4_STATIC != 0) {
            uint32_t orig_phys[kNumExpertsPerLane];
            uint32_t running_phys = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                const uint32_t e = i * 32u + lane_idx;
                const uint32_t tok = (e < kNumExpertsPerRank) ? stored_num_tokens_per_expert[i] : 0u;
                const uint32_t blk = math::ceil_div(tok, select_expert_bm<BLOCK_M>(tok));
                const uint32_t incl = math::warp_inclusive_sum(blk, lane_idx);
                const uint32_t off = running_phys + incl - blk;
                running_phys += ptx::exchange(incl, 31);
                orig_phys[i] = (tok << 12) | (off & 0xfffu);
            }
            uint32_t sched_running = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                const uint32_t s = i * 32u + lane_idx;
                const uint32_t ord_e = (s < kNumExpertsPerRank) ?
                    static_cast<uint32_t>(kD4StaticExpertOrder[rank_idx][s]) : 0u;
                const uint32_t p_pack = (s < kNumExpertsPerRank) ? get_slot_divergent(orig_phys, ord_e) : 0u;
                const uint32_t ord_tok = p_pack >> 12;
                const uint32_t ord_blk = math::ceil_div(ord_tok, select_expert_bm<BLOCK_M>(ord_tok));
                const uint32_t incl = math::warp_inclusive_sum(ord_blk, lane_idx);
                const uint32_t s_off = sched_running + incl - ord_blk;
                sched_running += ptx::exchange(incl, 31);
                d4_slot_sched[i] = ((ord_e & 0xffu) << 20) | ((ord_blk & 0xffu) << 12) | (s_off & 0xfffu);
                d4_slot_phys[i] = p_pack;
            }
            return;
        }
        #else
        (void)rank_idx;
        #endif

        if constexpr (kEffEpw == 64u and (kNumExpertsPerRank == 64u or kNumExpertsPerRank == 128u)) {
            uint32_t running_phys = 0;
            uint32_t sched_running = 0;
            #pragma unroll
            for (uint32_t g = 0; g < kNumExpertsPerLane; g += 2u) {
                uint32_t key0 = 0u, val0 = 0u, phys0 = 0u;
                uint32_t key1 = 0u, val1 = 0u, phys1 = 0u;
                {
                    const uint32_t e0 = g * 32u + lane_idx;
                    const uint32_t tok0 = stored_num_tokens_per_expert[g];
                    const uint32_t blk0 = math::ceil_div(tok0, select_expert_bm<BLOCK_M>(tok0));
                    const uint32_t incl0 = math::warp_inclusive_sum(blk0, lane_idx);
                    const uint32_t poff0 = running_phys + incl0 - blk0;
                    running_phys += ptx::exchange(incl0, 31);
                    key0 = (DG_ISOLATED_D4_MODE == 1)
                        ? ((blk0 << 20) | (127u - (e0 & 127u)))
                        : ((blk0 << 20) | (cute::min(tok0, 0x1fffu) << 7) | (127u - (e0 & 127u)));
                    val0 = (blk0 << 8) | e0;
                    phys0 = (tok0 << 12) | (poff0 & 0xfffu);

                    const uint32_t e1 = (g + 1u) * 32u + lane_idx;
                    const uint32_t tok1 = stored_num_tokens_per_expert[g + 1u];
                    const uint32_t blk1 = math::ceil_div(tok1, select_expert_bm<BLOCK_M>(tok1));
                    const uint32_t incl1 = math::warp_inclusive_sum(blk1, lane_idx);
                    const uint32_t poff1 = running_phys + incl1 - blk1;
                    running_phys += ptx::exchange(incl1, 31);
                    key1 = (DG_ISOLATED_D4_MODE == 1)
                        ? ((blk1 << 20) | (127u - (e1 & 127u)))
                        : ((blk1 << 20) | (cute::min(tok1, 0x1fffu) << 7) | (127u - (e1 & 127u)));
                    val1 = (blk1 << 8) | e1;
                    phys1 = (tok1 << 12) | (poff1 & 0xfffu);
                }
                #pragma unroll
                for (uint32_t k = 2u; k <= 32u; k <<= 1) {
                    const bool d0 = (k == 32u) ? true : ((lane_idx & k) == 0u);
                    const bool d1 = (k == 32u) ? false : ((lane_idx & k) == 0u);
                    #pragma unroll
                    for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
                        const uint32_t pk0 = __shfl_xor_sync(0xffffffffu, key0, j);
                        const uint32_t pv0 = __shfl_xor_sync(0xffffffffu, val0, j);
                        const uint32_t pp0 = __shfl_xor_sync(0xffffffffu, phys0, j);
                        const uint32_t pk1 = __shfl_xor_sync(0xffffffffu, key1, j);
                        const uint32_t pv1 = __shfl_xor_sync(0xffffffffu, val1, j);
                        const uint32_t pp1 = __shfl_xor_sync(0xffffffffu, phys1, j);
                        const bool up = (lane_idx & j) != 0u;
                        if ((d0 != up) ? (pk0 > key0) : (pk0 < key0)) { key0 = pk0; val0 = pv0; phys0 = pp0; }
                        if ((d1 != up) ? (pk1 > key1) : (pk1 < key1)) { key1 = pk1; val1 = pv1; phys1 = pp1; }
                    }
                }
                if (key1 > key0) {
                    const uint32_t tk = key0, tv = val0, tp = phys0;
                    key0 = key1; val0 = val1; phys0 = phys1;
                    key1 = tk; val1 = tv; phys1 = tp;
                }
                #pragma unroll
                for (uint32_t j = 16u; j > 0u; j >>= 1) {
                    const uint32_t pk0 = __shfl_xor_sync(0xffffffffu, key0, j);
                    const uint32_t pv0 = __shfl_xor_sync(0xffffffffu, val0, j);
                    const uint32_t pp0 = __shfl_xor_sync(0xffffffffu, phys0, j);
                    const uint32_t pk1 = __shfl_xor_sync(0xffffffffu, key1, j);
                    const uint32_t pv1 = __shfl_xor_sync(0xffffffffu, val1, j);
                    const uint32_t pp1 = __shfl_xor_sync(0xffffffffu, phys1, j);
                    const bool up = (lane_idx & j) != 0u;
                    if ((!up) ? (pk0 > key0) : (pk0 < key0)) { key0 = pk0; val0 = pv0; phys0 = pp0; }
                    if ((!up) ? (pk1 > key1) : (pk1 < key1)) { key1 = pk1; val1 = pv1; phys1 = pp1; }
                }
                const uint32_t m = static_cast<uint32_t>(
                    __popc(__ballot_sync(0xffffffffu, (val0 >> 8) > 0u)) +
                    __popc(__ballot_sync(0xffffffffu, (val1 >> 8) > 0u)));
                const bool is_last_win = ((g + 2u) >= kNumExpertsPerLane);
                #pragma unroll
                for (uint32_t i = 0; i < 2u; ++ i) {
                    const uint32_t p = i * 32u + lane_idx;
                    uint32_t src_r = p;
                    if (p < m and not (DG_ISOLATED_D4_MODE == 2 and is_last_win))
                        src_r = ((p & 1u) == 0u) ? (p >> 1) : (m - 1u - (p >> 1));
                    const uint32_t v_lo = __shfl_sync(0xffffffffu, val0, src_r & 31u);
                    const uint32_t v_hi = __shfl_sync(0xffffffffu, val1, src_r & 31u);
                    const uint32_t pval = (src_r < 32u) ? v_lo : v_hi;
                    const uint32_t p_lo = __shfl_sync(0xffffffffu, phys0, src_r & 31u);
                    const uint32_t p_hi = __shfl_sync(0xffffffffu, phys1, src_r & 31u);
                    const uint32_t pphys = (src_r < 32u) ? p_lo : p_hi;
                    const uint32_t pblk = pval >> 8;
                    const uint32_t pexp = pval & 0xffu;
                    const uint32_t incl = math::warp_inclusive_sum(pblk, lane_idx);
                    const uint32_t s_off = sched_running + incl - pblk;
                    sched_running += ptx::exchange(incl, 31);
                    d4_slot_sched[g + i] = (pexp << 20) | ((pblk & 0xffu) << 12) | (s_off & 0xfffu);
                    d4_slot_phys[g + i] = pphys;
                }
            }
            return;
        }

        constexpr uint32_t kWinEpw = (kEffEpw > 32u) ? 32u : kEffEpw;
        DG_STATIC_ASSERT(kWinEpw == 0u or kWinEpw == 1u or kWinEpw == 2u or kWinEpw == 4u or
                         kWinEpw == 8u or kWinEpw == 16u or kWinEpw == 32u,
                         "DG_ISOLATED_D4_EPW must be power of two in {1,2,4,8,16,32,64}");
        uint32_t running_phys = 0;
        uint32_t sched_running = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t e = i * 32u + lane_idx;
            const uint32_t tok = (e < kNumExpertsPerRank) ? stored_num_tokens_per_expert[i] : 0u;
            const uint32_t blk = math::ceil_div(tok, select_expert_bm<BLOCK_M>(tok));
            const uint32_t incl_phys = math::warp_inclusive_sum(blk, lane_idx);
            const uint32_t poff = running_phys + incl_phys - blk;
            running_phys += ptx::exchange(incl_phys, 31);

            uint32_t kv_key = (DG_ISOLATED_D4_MODE == 1)
                ? ((blk << 20) | (127u - (e & 127u)))
                : ((blk << 20) | (cute::min(tok, 0x1fffu) << 7) | (127u - (e & 127u)));
            if (e >= kNumExpertsPerRank)
                kv_key = 0u;
            uint32_t kv_val = (blk << 8) | (e & 0xffu);
            uint32_t kv_phys = (tok << 12) | (poff & 0xfffu);

            #pragma unroll
            for (uint32_t k = 2u; k <= kWinEpw; k <<= 1) {
                const bool dir_desc = (k == kWinEpw) ? true : ((lane_idx & k) == 0u);
                #pragma unroll
                for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
                    const uint32_t partner_key = __shfl_xor_sync(0xffffffffu, kv_key, j);
                    const uint32_t partner_val = __shfl_xor_sync(0xffffffffu, kv_val, j);
                    const uint32_t partner_phys = __shfl_xor_sync(0xffffffffu, kv_phys, j);
                    const bool upper_lane = (lane_idx & j) != 0u;
                    const bool want_max = (dir_desc != upper_lane);
                    const bool swap = want_max ? (partner_key > kv_key) : (partner_key < kv_key);
                    if (swap) {
                        kv_key = partner_key;
                        kv_val = partner_val;
                        kv_phys = partner_phys;
                    }
                }
            }

            const uint32_t w_base = lane_idx & ~(kWinEpw - 1u);
            const uint32_t p = lane_idx & (kWinEpw - 1u);
            const uint32_t full_active = __ballot_sync(0xffffffffu, (kv_val >> 8) > 0u);
            constexpr uint32_t kWinMask = (kWinEpw == 32u) ? 0xffffffffu : ((1u << kWinEpw) - 1u);
            const uint32_t m = static_cast<uint32_t>(__popc((full_active >> w_base) & kWinMask));

            uint32_t src_r = p;
            if (p < m) {
                if constexpr (DG_ISOLATED_D4_MODE == 2) {
                    const bool is_last_win = (i * 32u + w_base + kWinEpw >= kNumExpertsPerRank);
                    if (not is_last_win)
                        src_r = ((p & 1u) == 0u) ? (p >> 1) : (m - 1u - (p >> 1));
                } else {
                    src_r = ((p & 1u) == 0u) ? (p >> 1) : (m - 1u - (p >> 1));
                }
            }
            const uint32_t paired_val = __shfl_sync(0xffffffffu, kv_val, w_base + src_r);
            const uint32_t paired_phys = __shfl_sync(0xffffffffu, kv_phys, w_base + src_r);
            const uint32_t paired_blk = paired_val >> 8;
            const uint32_t paired_exp = paired_val & 0xffu;

            const uint32_t incl = math::warp_inclusive_sum(paired_blk, lane_idx);
            const uint32_t s_off = sched_running + incl - paired_blk;
            sched_running += ptx::exchange(incl, 31);
            d4_slot_sched[i] = (paired_exp << 20) | ((paired_blk & 0xffu) << 12) | (s_off & 0xfffu);
            d4_slot_phys[i] = paired_phys;
        }
    }
#endif

    CUTLASS_DEVICE task_info_t create_task_for_block(
            const BlockPhase& block_phase,
            const uint32_t& sched_m_block,
            const uint32_t& n_cluster_idx,
            const uint32_t& shape_n,
            const uint32_t& shape_k
#if DG_ISOLATED_D4_LUT_ACTIVE
            , const uint32_t (&d4_slot_sched)[kNumExpertsPerLane]
            , const uint32_t (&d4_slot_phys)[kNumExpertsPerLane]
#endif
            ) const {
#if DG_ISOLATED_D4_READYQ
        if (num_total_m_blocks <= kNumRingBlocks) {
            uint32_t packed = 0u;
            if (cute::elect_one_sync()) {
                do {
                    packed = ptx::ld_acq(workspace.get_d4_ready_queue_ptr(sched_m_block));
                } while (packed == 0u);
            }
            packed = ptx::exchange(packed, 0);
            const uint32_t expert_idx = (packed >> 24) & 0x7fu;
            const uint32_t m_in_expert = (packed >> 12) & 0xfffu;
            const uint32_t pool_blk = packed & 0xfffu;
            const uint32_t num_tok = get_num_tokens(expert_idx);
            const uint32_t expert_bm = select_expert_bm<BLOCK_M>(num_tok);
            const uint32_t valid_m = cute::min(num_tok - m_in_expert * expert_bm, expert_bm);
            return task_info_t(block_phase, expert_idx, m_in_expert, n_cluster_idx, pool_blk, valid_m, shape_n, shape_k);
        }
#endif

#if DG_ISOLATED_D4_LUT_ACTIVE
        {
            task_info_t result(block_phase, 0, 0, n_cluster_idx, sched_m_block, 0, shape_n, shape_k);
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                const uint32_t s_pack = d4_slot_sched[i];
                const uint32_t s_off = s_pack & 0xfffu;
                const uint32_t s_blks = (s_pack >> 12) & 0xffu;
                const bool is_owner = (sched_m_block >= s_off) and (sched_m_block < s_off + s_blks);
                const uint32_t owner_mask = __ballot_sync(0xffffffffu, is_owner);
                if (owner_mask) {
                    const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                    const uint32_t p_pack = d4_slot_phys[i];
                    const uint32_t owner_m_in_exp = sched_m_block - s_off;
                    const uint32_t owner_exp = s_pack >> 20;
                    const uint32_t owner_pool_blk = (p_pack & 0xfffu) + owner_m_in_exp;
                    const uint32_t owner_tok = p_pack >> 12;
                    const uint32_t expert_bm = select_expert_bm<BLOCK_M>(owner_tok);
                    const uint32_t owner_valid_m = cute::min(owner_tok - owner_m_in_exp * expert_bm, expert_bm);
                    result.local_expert_idx = ptx::exchange(owner_exp, owner_lane_idx);
                    result.m_block_idx = ptx::exchange(owner_m_in_exp, owner_lane_idx);
                    result.pool_block_idx = ptx::exchange(owner_pool_blk, owner_lane_idx);
                    result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
                }
            }
            return result;
        }
#else
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t m_block_idx = sched_m_block;
        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        #if DG_ISOLATED_D5_DEDUP == 1
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t expert_bm = select_expert_bm<BLOCK_M>(num_tokens);
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, expert_bm);
            const uint32_t lane_pool_block_offset = stored_lane_block_offset[i];
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * expert_bm, expert_bm);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
                return result;
            }
        }
        #else
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t expert_bm = select_expert_bm<BLOCK_M>(num_tokens);
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, expert_bm);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * expert_bm, expert_bm);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        #endif
        return result;
#endif
    }

    CUTLASS_DEVICE task_info_t create_task(
            const BlockPhase& block_phase,
            const uint32_t& task_idx,
            const uint32_t& num_clusters,
            const uint32_t& shape_n,
            const uint32_t& shape_k
#if DG_ISOLATED_D4_LUT_ACTIVE
            , const uint32_t (&d4_slot_sched)[kNumExpertsPerLane]
            , const uint32_t (&d4_slot_phys)[kNumExpertsPerLane]
#endif
            ) const {
        return create_task_for_block(
            block_phase, task_idx / num_clusters, task_idx % num_clusters, shape_n, shape_k
#if DG_ISOLATED_D4_LUT_ACTIVE
            , d4_slot_sched, d4_slot_phys
#endif
        );
    }

    static CUTLASS_DEVICE uint32_t get_next_task_idx(const uint32_t* global_task_count_ptr) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(global_task_count_ptr, 1u);
        return ptx::exchange(result, 0);
    }

    // Claim one task from the queue of `locality_domain_idx`. The counter is
    // private to that domain, and the decoded N clusters stay in its half.
    static CUTLASS_DEVICE uint32_t get_next_local_task_idx(
            const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr + locality_domain_idx, 1u);
        return ptx::exchange(result, 0);
    }

    template <uint32_t kNumClusters>
    static CUTLASS_DEVICE bool get_next_task_idx_for_domain(
            const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx,
            const uint32_t& num_m_blocks, uint32_t& m_block_idx, uint32_t& n_cluster_idx) {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
        return true;
    }

    // Domain-local claim first; steal from the peer only when the whole device
    // has at least kNumMinStealWaves waves of work, so small problems stay
    // die-local and do not pay cross-die latency.
    template <uint32_t kNumClusters>
    CUTLASS_DEVICE bool get_next_task_idx(
            const uint32_t* task_count_ptr, const uint32_t& num_m_blocks,
            uint32_t& m_block_idx, uint32_t& n_cluster_idx) const {
        if (get_next_task_idx_for_domain<kNumClusters>(
                task_count_ptr, sm_locality_domain_idx, num_m_blocks, m_block_idx, n_cluster_idx))
            return true;
        return kNumLocalityDomains > 1 and
               num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
               get_next_task_idx_for_domain<kNumClusters>(
                   task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, n_cluster_idx);
    }

    // An L2 K block may only start after every domain has published the L1 N
    // blocks of that M block. Counters are per domain, so wait on each half.
    CUTLASS_DEVICE void wait_l1_tasks(const uint32_t& num_m_blocks_done) const {
        constexpr uint32_t kNumLocalL1Clusters = kNumL1Clusters / kNumLocalityDomains;
        const uint32_t num_required_l1_tasks = num_m_blocks_done * kNumLocalL1Clusters;
        #pragma unroll
        for (uint32_t d = 0; d < kNumLocalityDomains; ++ d)
            while (ptx::ld_volatile(workspace.get_l1_task_count_ptr(d)) < num_required_l1_tasks) {}
    }

#if DG_ISOLATED_D5_DEDUP && DG_TUNE_D5_SCHED
    CUTLASS_DEVICE task_info_t get_next_task_d5(
            const uint32_t& d5_tier0_end,
            const uint32_t& d5_tier1_end
#if DG_ISOLATED_D4_LUT_ACTIVE
            , const uint32_t (&d4_slot_sched)[kNumExpertsPerLane]
            , const uint32_t (&d4_slot_phys)[kNumExpertsPerLane]
#endif
            ) {
        const bool use_three_tier_queue = (num_total_m_blocks <= kNumRingBlocks);
        BlockPhase sel_phase = BlockPhase::None;
        uint32_t sel_m_block = 0u;
        uint32_t sel_n_cluster = 0u;
        #pragma unroll 1
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                -- num_sched_l1_waves;

                const uint32_t l1_task_idx = get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (l1_task_idx >= num_total_m_blocks * kNumL1Clusters) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                const uint32_t seq_idx = l1_task_idx / kNumL1Clusters;
                sel_n_cluster = l1_task_idx % kNumL1Clusters;
                sel_phase = BlockPhase::Linear1;
                if (not use_three_tier_queue) {
                    sel_m_block = seq_idx;
                    break;
                }

                if (sel_n_cluster == 0u) {
                    uint32_t picked = 0u;
                    if (cute::elect_one_sync()) {
                        const uint32_t tier = (seq_idx < d5_tier0_end) ? 0u : ((seq_idx < d5_tier1_end) ? 1u : 2u);
                        uint32_t* tier_bmap = d5::tier_bitmap_word_ptr(workspace, tier, 0u);
                        const uint32_t num_words = math::ceil_div(num_total_m_blocks, 32u);
                        uint32_t w_cur = seq_idx % num_words;
                        const uint32_t rot = (seq_idx * 13u) & 31u;
                        #pragma unroll 1
                        while (true) {
                            const uint32_t word = ptx::ld_acq(tier_bmap + w_cur);
                            if (word != 0u) {
                                const uint32_t rword = __funnelshift_r(word, word, rot);
                                const uint32_t rbit = static_cast<uint32_t>(__ffs(static_cast<int>(rword)) - 1);
                                const uint32_t bit = (rbit + rot) & 31u;
                                const uint32_t mask = 1u << bit;
                                const uint32_t prev = d5::atomic_and_acq_rel_gpu_u32(tier_bmap + w_cur, ~mask);
                                if ((prev & mask) != 0u) {
                                    picked = w_cur * 32u + bit;
                                    d5::st_release_gpu(d5::sched_m_block_ptr(workspace, seq_idx), picked + 1u);
                                    break;
                                }
                            }
                            w_cur = (w_cur + 1u == num_words) ? 0u : (w_cur + 1u);
                        }
                    }
                    sel_m_block = ptx::exchange(picked, 0);
                } else {
                    uint32_t b_plus_1 = 0u;
                    if (cute::elect_one_sync()) {
                        uint32_t* slot_ptr = d5::sched_m_block_ptr(workspace, seq_idx);
                        do {
                            b_plus_1 = ptx::ld_acq(slot_ptr);
                        } while (b_plus_1 == 0u);
                    }
                    sel_m_block = ptx::exchange(b_plus_1, 0) - 1u;
                }
                break;
            } else {
                const uint32_t l2_task_idx = get_next_task_idx(workspace.get_l2_task_count_ptr());
                if (l2_task_idx >= num_total_m_blocks * kNumL2Clusters)
                    return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);

                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                const uint32_t seq_idx = l2_task_idx / kNumL2Clusters;
                sel_n_cluster = l2_task_idx % kNumL2Clusters;
                sel_phase = BlockPhase::Linear2;
                const uint32_t num_required_l1_tasks = (seq_idx + 1u) * kNumL1Clusters;
                if (not use_three_tier_queue) {
                    sel_m_block = seq_idx;
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < num_required_l1_tasks) {}
                    break;
                }

                uint32_t b_plus_1 = 0u;
                if (cute::elect_one_sync()) {
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < num_required_l1_tasks) {}
                    uint32_t* slot_ptr = d5::sched_m_block_ptr(workspace, seq_idx);
                    do {
                        b_plus_1 = ptx::ld_acq(slot_ptr);
                    } while (b_plus_1 == 0u);
                }
                sel_m_block = ptx::exchange(b_plus_1, 0) - 1u;
                break;
            }
        }
        const uint32_t shape_n = (sel_phase == BlockPhase::Linear1) ? L1_SHAPE_N : L2_SHAPE_N;
        const uint32_t shape_k = (sel_phase == BlockPhase::Linear1) ? L1_SHAPE_K : L2_SHAPE_K;
        return create_task_for_block(
            sel_phase, sel_m_block, sel_n_cluster, shape_n, shape_k
#if DG_ISOLATED_D4_LUT_ACTIVE
            , d4_slot_sched, d4_slot_phys
#endif
        );
    }
#else
    CUTLASS_DEVICE task_info_t get_next_task(
#if DG_ISOLATED_D4_LUT_ACTIVE
            const uint32_t (&d4_slot_sched)[kNumExpertsPerLane],
            const uint32_t (&d4_slot_phys)[kNumExpertsPerLane]
#endif
            ) {
        while (true) {
            uint32_t sel_m_block = 0u, sel_n_cluster = 0u;
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                -- num_sched_l1_waves;

                if (not get_next_task_idx<kNumL1Clusters>(
                        workspace.get_l1_task_count_ptr(), num_total_m_blocks, sel_m_block, sel_n_cluster)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                return create_task_for_block(BlockPhase::Linear1, sel_m_block, sel_n_cluster, L1_SHAPE_N, L1_SHAPE_K
#if DG_ISOLATED_D4_LUT_ACTIVE
                    , d4_slot_sched, d4_slot_phys
#endif
                );
            } else {
                if (not get_next_task_idx<kNumL2Clusters>(
                        workspace.get_l2_task_count_ptr(), num_total_m_blocks, sel_m_block, sel_n_cluster))
                    break;

                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                auto task_info = create_task_for_block(BlockPhase::Linear2, sel_m_block, sel_n_cluster, L2_SHAPE_N, L2_SHAPE_K
#if DG_ISOLATED_D4_LUT_ACTIVE
                    , d4_slot_sched, d4_slot_phys
#endif
                );

                wait_l1_tasks(sel_m_block + 1u);
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }
#endif

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumClusters = (kShapeN / BLOCK_N) / 2;
        DG_STATIC_ASSERT(kNumClusters % kNumLocalityDomains == 0, "Each domain must take whole shared clusters");
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            uint32_t m_block_idx, n_cluster_idx;
            if (not get_next_task_idx<kNumClusters>(task_count_ptr, num_m_blocks, m_block_idx, n_cluster_idx))
                break;

            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    CUTLASS_DEVICE void mainloop(
            const uint32_t& num_tokens,
            const uint32_t& rank_idx = 0,
            uint16_t* sched_cum_blocks = nullptr,
            uint8_t* sched_expert_order = nullptr,
            const uint8_t* sm_locality_domains = nullptr) {
        const auto lane_idx = ptx::get_lane_idx();
        sm_locality_domain_idx = (kNumLocalityDomains == 1 or sm_locality_domains == nullptr) ?
            0u : sm_locality_domains[ptx::get_sm_idx()];

        if constexpr (kHasShared) {
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

#if DG_ISOLATED_D4_LUT_ACTIVE
        uint32_t d4_slot_sched[kNumExpertsPerLane] = {};
        uint32_t d4_slot_phys[kNumExpertsPerLane] = {};
        build_d4_schedule_plan(rank_idx, d4_slot_sched, d4_slot_phys, sched_cum_blocks, sched_expert_order);
#else
        (void)rank_idx;
        (void)sched_cum_blocks;
        (void)sched_expert_order;
#endif

#if DG_ISOLATED_D5_DEDUP && DG_TUNE_D5_SCHED
        uint32_t d5_tier0_end = 0u, d5_tier1_end = 0u;
        if (num_total_m_blocks > 0u and num_total_m_blocks <= kNumRingBlocks) {
            uint32_t b0 = 0u, b1 = 0u;
            if (cute::elect_one_sync()) {
                while (ptx::ld_acq(d5::plan_ready_ptr(workspace)) == 0u) {}
                b0 = *d5::tier0_count_ptr(workspace);
                b1 = *d5::tier1_count_ptr(workspace);
            }
            b0 = ptx::exchange(b0, 0);
            b1 = ptx::exchange(b1, 0);
            d5_tier0_end = b0;
            d5_tier1_end = b0 + b1;
        }
#endif

        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
#if DG_ISOLATED_D5_DEDUP && DG_TUNE_D5_SCHED
            task_info = get_next_task_d5(
                d5_tier0_end, d5_tier1_end
    #if DG_ISOLATED_D4_LUT_ACTIVE
                , d4_slot_sched, d4_slot_phys
    #endif
            );
#else
            task_info = get_next_task(
    #if DG_ISOLATED_D4_LUT_ACTIVE
                d4_slot_sched, d4_slot_phys
    #endif
            );
#endif
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        #pragma unroll
        for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
        }

        #pragma unroll
        for (uint32_t i = 0; i < kNumScheduleStages; ++ i)
            task_info_empty_barriers[i].wait((sched_phase + i >= kNumScheduleStages) ^ 1);
    }
};

#endif

} // namespace deep_gemm::sched
