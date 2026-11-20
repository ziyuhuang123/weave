#pragma once
// =============================================================================
// Weave unified spatial / temporal / dataflow knobs.
//
// One ABI is shared by:
//   * NVIDIA B300 / B300 NVFP4: CUDA C++ (sm_100a / sm_103a)
//   * NVIDIA H20:                CUDA C++ (sm_90a)
//   * AMD MI308X:                HIP C++   (gfx942)
//
// Compile-time macros remain the fallback for reproducible ablations.  The
// runtime WeaveKnobs structure can override every policy field.  A selected
// r_mask packs the spatial schedule as:
//   bits [ 7: 0] N_c  (communication warps / waves per GPU)
//   bits [15: 8] N_e  (compute / epilogue warps or waves per GPU)
//   bits [23:16] number of role groups in one CTA/CU
//   bit  [31]    heterogeneous-SM/CU mixing is enabled
// =============================================================================

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
// __forceinline__ is a built-in keyword under nvcc / clang-CUDA, but under HIP
// it is only a macro supplied by hip/hip_runtime.h.  megamoe_amd*.hip includes
// this scheduler header BEFORE any HIP runtime header, so the macro is not yet
// defined when clang -x hip parses it (gfx942 build failed with "unknown type
// name '__forceinline__'").  Provide a portable always-inline fallback that is
// semantically identical on both toolchains and leaves the CUDA path unchanged.
#if defined(__forceinline__)
#define WEAVE_FORCEINLINE __forceinline__
#else
#define WEAVE_FORCEINLINE __attribute__((always_inline)) inline
#endif
#define WEAVE_HOST_DEVICE __host__ __device__ WEAVE_FORCEINLINE
#define WEAVE_DEVICE __device__ WEAVE_FORCEINLINE
#else
#define WEAVE_HOST_DEVICE inline
#define WEAVE_DEVICE inline
#endif

#ifndef WEAVE_ABLATION_PRESET
#define WEAVE_ABLATION_PRESET 0
#endif
#if WEAVE_ABLATION_PRESET < 0 || WEAVE_ABLATION_PRESET > 3
#error "WEAVE_ABLATION_PRESET must be in [0, 3]"
#endif

// Public WEAVE_* names are primary. The archived WEAVE_KNOB_* spellings
// remain compile-time compatibility aliases only.
#ifndef WEAVE_ENABLE_COST_MODEL
#ifdef WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL
#define WEAVE_ENABLE_COST_MODEL WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL
#else
#define WEAVE_ENABLE_COST_MODEL ((WEAVE_ABLATION_PRESET == 0) ? 1u : 0u)
#endif
#endif
#ifndef WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL
#define WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL WEAVE_ENABLE_COST_MODEL
#endif
#ifndef WEAVE_COMM_UNITS
#ifdef WEAVE_KNOB_COMM_UNITS
#define WEAVE_COMM_UNITS WEAVE_KNOB_COMM_UNITS
#else
#define WEAVE_COMM_UNITS 4u
#endif
#endif
#ifndef WEAVE_KNOB_COMM_UNITS
#define WEAVE_KNOB_COMM_UNITS WEAVE_COMM_UNITS
#endif
#ifndef WEAVE_COMP_UNITS
#ifdef WEAVE_KNOB_COMP_UNITS
#define WEAVE_COMP_UNITS WEAVE_KNOB_COMP_UNITS
#else
#define WEAVE_COMP_UNITS 4u
#endif
#endif
#ifndef WEAVE_KNOB_COMP_UNITS
#define WEAVE_KNOB_COMP_UNITS WEAVE_COMP_UNITS
#endif
#ifndef WEAVE_DUALGROUP_MIN_TOKENS
#ifdef WEAVE_KNOB_DUALGROUP_MIN_TOKENS
#define WEAVE_DUALGROUP_MIN_TOKENS WEAVE_KNOB_DUALGROUP_MIN_TOKENS
#else
#define WEAVE_DUALGROUP_MIN_TOKENS 64u
#endif
#endif
#ifndef WEAVE_KNOB_DUALGROUP_MIN_TOKENS
#define WEAVE_KNOB_DUALGROUP_MIN_TOKENS WEAVE_DUALGROUP_MIN_TOKENS
#endif
#ifndef WEAVE_BM_TIER_MODE
#ifdef WEAVE_KNOB_BM_TIER_MODE
#define WEAVE_BM_TIER_MODE WEAVE_KNOB_BM_TIER_MODE
#else
#define WEAVE_BM_TIER_MODE 2u
#endif
#endif
#ifndef WEAVE_KNOB_BM_TIER_MODE
#define WEAVE_KNOB_BM_TIER_MODE WEAVE_BM_TIER_MODE
#endif
#ifndef WEAVE_BN1
#ifdef WEAVE_KNOB_BN1
#define WEAVE_BN1 WEAVE_KNOB_BN1
#else
#define WEAVE_BN1 128u
#endif
#endif
#ifndef WEAVE_KNOB_BN1
#define WEAVE_KNOB_BN1 WEAVE_BN1
#endif
#ifndef WEAVE_BN2_MODE
#ifdef WEAVE_KNOB_BN2_MODE
#define WEAVE_BN2_MODE WEAVE_KNOB_BN2_MODE
#else
#define WEAVE_BN2_MODE 0u
#endif
#endif
#ifndef WEAVE_KNOB_BN2_MODE
#define WEAVE_KNOB_BN2_MODE WEAVE_BN2_MODE
#endif
#ifndef WEAVE_BK
#ifdef WEAVE_KNOB_BK
#define WEAVE_BK WEAVE_KNOB_BK
#else
#define WEAVE_BK 128u
#endif
#endif
#ifndef WEAVE_KNOB_BK
#define WEAVE_KNOB_BK WEAVE_BK
#endif
#ifndef WEAVE_S2_NGROUP_MODE
#ifdef WEAVE_KNOB_S2_NGROUP_MODE
#define WEAVE_S2_NGROUP_MODE WEAVE_KNOB_S2_NGROUP_MODE
#else
#define WEAVE_S2_NGROUP_MODE 0u
#endif
#endif
#ifndef WEAVE_KNOB_S2_NGROUP_MODE
#define WEAVE_KNOB_S2_NGROUP_MODE WEAVE_S2_NGROUP_MODE
#endif
#ifndef WEAVE_HETERO_SM_MASK
#ifdef WEAVE_KNOB_HETERO_SM_MASK
#define WEAVE_HETERO_SM_MASK WEAVE_KNOB_HETERO_SM_MASK
#else
#define WEAVE_HETERO_SM_MASK 0u
#endif
#endif
#ifndef WEAVE_KNOB_HETERO_SM_MASK
#define WEAVE_KNOB_HETERO_SM_MASK WEAVE_HETERO_SM_MASK
#endif
#ifndef WEAVE_OVERLAP_PIPELINE
#ifdef WEAVE_KNOB_OVERLAP_PIPELINE
#define WEAVE_OVERLAP_PIPELINE WEAVE_KNOB_OVERLAP_PIPELINE
#else
#define WEAVE_OVERLAP_PIPELINE ((WEAVE_ABLATION_PRESET >= 2) ? 0u : 1u)
#endif
#endif
#ifndef WEAVE_KNOB_OVERLAP_PIPELINE
#define WEAVE_KNOB_OVERLAP_PIPELINE WEAVE_OVERLAP_PIPELINE
#endif
#ifndef WEAVE_SCATTER_MODE
#ifdef WEAVE_KNOB_SCATTER_MODE
#define WEAVE_SCATTER_MODE WEAVE_KNOB_SCATTER_MODE
#else
#define WEAVE_SCATTER_MODE 1u
#endif
#endif
#ifndef WEAVE_KNOB_SCATTER_MODE
#define WEAVE_KNOB_SCATTER_MODE WEAVE_SCATTER_MODE
#endif
#ifndef WEAVE_GEMM_STAGES
#ifdef WEAVE_KNOB_GEMM_STAGES
#define WEAVE_GEMM_STAGES WEAVE_KNOB_GEMM_STAGES
#else
#define WEAVE_GEMM_STAGES 2u
#endif
#endif
#ifndef WEAVE_KNOB_GEMM_STAGES
#define WEAVE_KNOB_GEMM_STAGES WEAVE_GEMM_STAGES
#endif
#ifndef WEAVE_ALL_WAVE_COMBINE
#ifdef WEAVE_KNOB_ALL_WAVE_COMBINE
#define WEAVE_ALL_WAVE_COMBINE WEAVE_KNOB_ALL_WAVE_COMBINE
#else
#define WEAVE_ALL_WAVE_COMBINE ((WEAVE_ABLATION_PRESET >= 2) ? 0u : 1u)
#endif
#endif
#ifndef WEAVE_KNOB_ALL_WAVE_COMBINE
#define WEAVE_KNOB_ALL_WAVE_COMBINE WEAVE_ALL_WAVE_COMBINE
#endif
#ifndef WEAVE_SPIN_BEFORE_SLEEP
#ifdef WEAVE_KNOB_SPIN_BEFORE_SLEEP
#define WEAVE_SPIN_BEFORE_SLEEP WEAVE_KNOB_SPIN_BEFORE_SLEEP
#else
#define WEAVE_SPIN_BEFORE_SLEEP 256u
#endif
#endif
#ifndef WEAVE_KNOB_SPIN_BEFORE_SLEEP
#define WEAVE_KNOB_SPIN_BEFORE_SLEEP WEAVE_SPIN_BEFORE_SLEEP
#endif
#ifndef WEAVE_SELECTIVE_DEDUP
#ifdef WEAVE_KNOB_SELECTIVE_DEDUP
#define WEAVE_SELECTIVE_DEDUP WEAVE_KNOB_SELECTIVE_DEDUP
#else
#define WEAVE_SELECTIVE_DEDUP ((WEAVE_ABLATION_PRESET >= 3) ? 0u : 1u)
#endif
#endif
#ifndef WEAVE_KNOB_SELECTIVE_DEDUP
#define WEAVE_KNOB_SELECTIVE_DEDUP WEAVE_SELECTIVE_DEDUP
#endif
#ifndef WEAVE_REGISTER_DIRECT_COMM
#ifdef WEAVE_KNOB_REGISTER_DIRECT_COMM
#define WEAVE_REGISTER_DIRECT_COMM WEAVE_KNOB_REGISTER_DIRECT_COMM
#else
#define WEAVE_REGISTER_DIRECT_COMM 1u
#endif
#endif
#ifndef WEAVE_KNOB_REGISTER_DIRECT_COMM
#define WEAVE_KNOB_REGISTER_DIRECT_COMM WEAVE_REGISTER_DIRECT_COMM
#endif
#ifndef WEAVE_PARALLEL_ROUTE_PLAN
#ifdef WEAVE_KNOB_PARALLEL_ROUTE_PLAN
#define WEAVE_PARALLEL_ROUTE_PLAN WEAVE_KNOB_PARALLEL_ROUTE_PLAN
#else
#define WEAVE_PARALLEL_ROUTE_PLAN 1u
#endif
#endif
#ifndef WEAVE_KNOB_PARALLEL_ROUTE_PLAN
#define WEAVE_KNOB_PARALLEL_ROUTE_PLAN WEAVE_PARALLEL_ROUTE_PLAN
#endif
#ifndef WEAVE_LOCAL_FIRST_ORDER
#ifdef WEAVE_KNOB_LOCAL_FIRST_ORDER
#define WEAVE_LOCAL_FIRST_ORDER WEAVE_KNOB_LOCAL_FIRST_ORDER
#else
#define WEAVE_LOCAL_FIRST_ORDER 1u
#endif
#endif
#ifndef WEAVE_KNOB_LOCAL_FIRST_ORDER
#define WEAVE_KNOB_LOCAL_FIRST_ORDER WEAVE_LOCAL_FIRST_ORDER
#endif
#ifndef WEAVE_FUSED_SWIGLU_REQUANT
#ifdef WEAVE_KNOB_FUSED_SWIGLU_REQUANT
#define WEAVE_FUSED_SWIGLU_REQUANT WEAVE_KNOB_FUSED_SWIGLU_REQUANT
#else
#define WEAVE_FUSED_SWIGLU_REQUANT 1u
#endif
#endif
#ifndef WEAVE_KNOB_FUSED_SWIGLU_REQUANT
#define WEAVE_KNOB_FUSED_SWIGLU_REQUANT WEAVE_FUSED_SWIGLU_REQUANT
#endif
#ifndef WEAVE_TAIL_FINE_GRAINED
#ifdef WEAVE_KNOB_TAIL_FINE_GRAINED
#define WEAVE_TAIL_FINE_GRAINED WEAVE_KNOB_TAIL_FINE_GRAINED
#else
#define WEAVE_TAIL_FINE_GRAINED ((WEAVE_ABLATION_PRESET >= 2) ? 0u : 1u)
#endif
#endif
#ifndef WEAVE_KNOB_TAIL_FINE_GRAINED
#define WEAVE_KNOB_TAIL_FINE_GRAINED WEAVE_TAIL_FINE_GRAINED
#endif
#ifndef WEAVE_V_HEAD
#ifdef WEAVE_KNOB_V_HEAD
#define WEAVE_V_HEAD WEAVE_KNOB_V_HEAD
#else
#define WEAVE_V_HEAD 0xffffffffu
#endif
#endif
#ifndef WEAVE_KNOB_V_HEAD
#define WEAVE_KNOB_V_HEAD WEAVE_V_HEAD
#endif
#ifndef WEAVE_PRECISION_MODE
#ifdef WEAVE_KNOB_PRECISION_MODE
#define WEAVE_PRECISION_MODE WEAVE_KNOB_PRECISION_MODE
#else
#define WEAVE_PRECISION_MODE 0u
#endif
#endif
#ifndef WEAVE_KNOB_PRECISION_MODE
#define WEAVE_KNOB_PRECISION_MODE WEAVE_PRECISION_MODE
#endif
#if WEAVE_PRECISION_MODE > 3
#error "WEAVE_PRECISION_MODE must be in [0, 3]"
#endif

// =============================================================================
// Verified architectural knob space (Spatial / Temporal / Implementation).
//
// These macros are the compile-time defaults carried into JIT configuration.
// Host dispatchers override them per launch through WeaveKnobs and the
// per-backend JIT define blocks.  Every macro below is part of the public
// surface and is documented in docs/TUNABLE_PARAMETERS.md.
// =============================================================================

// --- Spatial ---------------------------------------------------------------
// 0 = auto (4 warps on SM103/H20, 4 on MI308X); 4 or 8 = fixed dispatch warps.
#ifndef WEAVE_DISPATCH_WARPS
#define WEAVE_DISPATCH_WARPS 0u
#endif
// 0 = every SM/CU participates in scatter; K = only K SMs/CUs drain the queue.
#ifndef WEAVE_SCATTER_ACTIVE_SMS
#define WEAVE_SCATTER_ACTIVE_SMS 0u
#endif
// 0 = auto; 1 = single domain; 2 = dual-die SM103; 4 = XCD MI308X.
#ifndef WEAVE_LOCALITY_DOMAINS
#define WEAVE_LOCALITY_DOMAINS 0u
#endif
// 0 = SM 0 communication warps pull like every other SM; 1 = they become the
// dedicated streaming scheduler lane and the remaining SMs own the pull.
#ifndef WEAVE_SM0_STREAM_SCHED
#define WEAVE_SM0_STREAM_SCHED 0u
#endif
// Minimum local tokens per CU-expert before the MI308X 8-wave schedule is
// admitted.  Defaults to the historical dual-group threshold.
#ifndef WEAVE_AMD_8WAVE_MIN_TOK
#define WEAVE_AMD_8WAVE_MIN_TOK WEAVE_DUALGROUP_MIN_TOKENS
#endif
// 0 = every SM runs the dispatch pull; D = only D SMs pull remote tokens and
// the remaining SMs start GEMM work immediately.
#ifndef WEAVE_DISPATCH_ACTIVE_SMS
#define WEAVE_DISPATCH_ACTIVE_SMS 0u
#endif
// Strict binary scatter ownership (never mixed):
//   0 = epilogue_only: epilogue warps issue the remote FC2 stores directly;
//   1 = comm_only:     epilogue warps only stage locally and publish
//                      descriptors; communication warps own every remote store.
#ifndef WEAVE_SCATTER_OWNER
#define WEAVE_SCATTER_OWNER 0u
#endif
#if WEAVE_SCATTER_OWNER > 1
#error "WEAVE_SCATTER_OWNER must be 0 (epilogue_only) or 1 (comm_only)"
#endif

// --- Temporal --------------------------------------------------------------
// 0 = default L1-first order; 1 = 64-bit XOR L2KBlockDependency bitmask with a
// tight 1 L2 <-> 1 L1 alternating wave schedule.
#ifndef WEAVE_L2_KBLOCK_DEP
#define WEAVE_L2_KBLOCK_DEP 0u
#endif
// 0 = auto ring capacity from get_num_max_live_pool_blocks; N = forced ring
// block count (validated against the workspace capacity).
#ifndef WEAVE_RING_BLOCKS
#define WEAVE_RING_BLOCKS 0u
#endif
// 0 = global barrier before combine; 1 = barrier-free per-token cnt==topk
// tail-interleaved early reduce.
#ifndef WEAVE_EARLY_STREAM_REDUCE
#define WEAVE_EARLY_STREAM_REDUCE 0u
#endif
// Integer permille encoding of the TMEM epilogue pipeline depth:
//   0 = auto (2.0 at BM<=240 with one prefetch stage; 1.75 at BM=256),
//   100 = 1.0 single stage, 175 = 1.75 overlapping, 200 = 2.0 two stages.
#ifndef WEAVE_TMEM_PIPELINE_MODE
#define WEAVE_TMEM_PIPELINE_MODE 0u
#endif
// 0 = pull the opening head budget as one prefix; N = tokens per head-pull
// chunk, interleaving dispatch and the first GEMM waves.
#ifndef WEAVE_HEAD_PULL_CHUNK
#define WEAVE_HEAD_PULL_CHUNK 0u
#endif
// 0 = one global barrier before combine; G = chunked dedup combine sync: one
// cross-GPU release per (source rank, G-token chunk) once every deduplicated
// row of the chunk has been sent.  Requires local scatter dedup.
#ifndef WEAVE_COMBINE_SYNC_CHUNK
#define WEAVE_COMBINE_SYNC_CHUNK 0u
#endif
#if WEAVE_COMBINE_SYNC_CHUNK != 0 && WEAVE_COMBINE_SYNC_CHUNK != 16 && \
    WEAVE_COMBINE_SYNC_CHUNK != 32 && WEAVE_COMBINE_SYNC_CHUNK != 64 && \
    WEAVE_COMBINE_SYNC_CHUNK != 128
#error "WEAVE_COMBINE_SYNC_CHUNK must be 0, 16, 32, 64 or 128"
#endif
// Epilogue TMEM -> register pipeline: 0 = auto (2 stages at BM>=240 with a
// TMEM depth of at least 1.75, else 1); 1 = single buffer; 2 = two register
// stages, prefetching store block s+1 while store block s is converted.
#ifndef WEAVE_EPI_REG_STAGES
#define WEAVE_EPI_REG_STAGES 0u
#endif
#if WEAVE_EPI_REG_STAGES > 2
#error "WEAVE_EPI_REG_STAGES must be 0 (auto), 1 or 2"
#endif
// L2-to-L1 interleaved scheduling ratio (1..4).
#ifndef WEAVE_SCHED_L2_RATIO
#define WEAVE_SCHED_L2_RATIO 1u
#endif
#if WEAVE_SCHED_L2_RATIO < 1 || WEAVE_SCHED_L2_RATIO > 4
#error "WEAVE_SCHED_L2_RATIO must be in [1, 4]"
#endif
// Register budget rebalance on SM103 (0 = 96/72/168, 1 = 64/56/192 at E/rank > 64).
#ifndef WEAVE_REG_BALANCE
#define WEAVE_REG_BALANCE 0u
#endif
#if WEAVE_REG_BALANCE > 1
#error "WEAVE_REG_BALANCE must be 0 or 1"
#endif

// --- Implementation --------------------------------------------------------
// 0 = fixed per-expert BM; 1 = per-expert dynamic BM ladder.
#ifndef WEAVE_ADAPT_BM
#define WEAVE_ADAPT_BM 0u
#endif
// 0 = auto; per-GEMM per-expert block-M envelopes.
#ifndef WEAVE_BLOCK_M_L1
#define WEAVE_BLOCK_M_L1 0u
#endif
#ifndef WEAVE_BLOCK_M_L2
#define WEAVE_BLOCK_M_L2 0u
#endif
// 0 = auto; asymmetric epilogue store heights for GEMM1 and GEMM2.
#ifndef WEAVE_STORE_BLOCK_M_L1
#define WEAVE_STORE_BLOCK_M_L1 0u
#endif
#ifndef WEAVE_STORE_BLOCK_M_L2
#define WEAVE_STORE_BLOCK_M_L2 0u
#endif
// Historical primary spelling: WEAVE_STORE_BLOCK_M now aliases the GEMM1
// store height so existing launch scripts keep working.
#ifndef WEAVE_STORE_BLOCK_M
#define WEAVE_STORE_BLOCK_M WEAVE_STORE_BLOCK_M_L1
#endif
// 0 = full-precision local accumulation; 1 = hardware bf16x2 L2 atomic dedup
// before the NVLink/xGMI send.
#ifndef WEAVE_LOCAL_SCATTER_DEDUP
#define WEAVE_LOCAL_SCATTER_DEDUP 0u
#endif
// Scatter staging path (all three are kept selectable):
//   0 = reg_2stage: double-buffered register transfer;
//   1 = smem:       per-SM on-chip handoff to the same SM's dispatch warps;
//   2 = hbm_dedup:  L2-resident HBM buffer with bf16x2 atomic dedup and a
//                   global descriptor queue drained by the scatter SMs.
#ifndef WEAVE_SCATTER_STAGING
#define WEAVE_SCATTER_STAGING 0u
#endif
#if WEAVE_SCATTER_STAGING > 2
#error "WEAVE_SCATTER_STAGING must be 0 (reg_2stage), 1 (smem) or 2 (hbm_dedup)"
#endif
#if WEAVE_SCATTER_STAGING != 0 && WEAVE_SCATTER_OWNER == 0
#error "smem / hbm_dedup scatter staging requires WEAVE_SCATTER_OWNER=1 (comm_only)"
#endif
#if WEAVE_COMBINE_SYNC_CHUNK != 0 && WEAVE_SCATTER_STAGING != 2 && WEAVE_LOCAL_SCATTER_DEDUP == 0
#error "WEAVE_COMBINE_SYNC_CHUNK > 0 must be paired with local scatter dedup"
#endif

namespace weave {

enum class HardwareArch : uint32_t {
  B300_SM100 = 0,
  H20_SM90 = 1,
  MI308X_GFX942 = 2,
};

// Unified B300 precision knob.  The same enum is mirrored by mma_type in
// deep_gemm.mega.weave_mega_moe and by csrc/apis/mega.hpp.
enum class PrecisionMode : uint32_t {
  FP8xFP4 = 0,
  NVFP4 = 1,
  FP8xFP8 = 2,
  BF16 = 3,
};

WEAVE_HOST_DEVICE constexpr const char* precision_mode_name(
    PrecisionMode mode) {
  return mode == PrecisionMode::FP8xFP4 ? "fp8xfp4" :
         mode == PrecisionMode::NVFP4 ? "fp4xfp4" :
         mode == PrecisionMode::FP8xFP8 ? "fp8" : "bf16";
}

enum class ScatterPath : uint32_t {
  EpilogueInline = 0,
  CommQueueAsync = 1,
  LocalDedupFirst = 2,
};

enum OverrideBit : uint32_t {
  kOverrideSpatial = 1u << 0,
  kOverrideScatter = 1u << 1,
  kOverrideVHead = 1u << 2,
  kOverrideTemporal = 1u << 3,
};

static constexpr uint32_t kAutoU32 = 0xffffffffu;

WEAVE_HOST_DEVICE constexpr uint32_t encode_r_mask(
    uint32_t comm_units, uint32_t comp_units, uint32_t groups,
    bool heterogeneous = false) {
  return ((comm_units & 0xffu) |
          ((comp_units & 0xffu) << 8u) |
          ((groups & 0xffu) << 16u) |
          (heterogeneous ? 0x80000000u : 0u));
}

WEAVE_HOST_DEVICE constexpr uint32_t r_comm_units(uint32_t r_mask) {
  return r_mask & 0xffu;
}
WEAVE_HOST_DEVICE constexpr uint32_t r_comp_units(uint32_t r_mask) {
  return (r_mask >> 8u) & 0xffu;
}
WEAVE_HOST_DEVICE constexpr uint32_t r_groups(uint32_t r_mask) {
  return (r_mask >> 16u) & 0xffu;
}
WEAVE_HOST_DEVICE constexpr bool r_heterogeneous(uint32_t r_mask) {
  return (r_mask & 0x80000000u) != 0u;
}

struct WeaveKnobs {
  // --- CTA-0 policy ABI: every field below is read by weave_cost_model.cuh.
  uint32_t enable_runtime_cost_model;
  uint32_t override_mask;
  uint32_t override_r_mask;
  uint32_t override_scatter_mode;
  uint32_t override_v_head;
  uint32_t block_m_fixed;   // Spatial fallback when the override r_mask is bad.
  uint32_t precision_mode;  // Implementation: numeric precision and MMA family.

  // --- Spatial -------------------------------------------------------------
  uint32_t dispatch_warps;      // 0 = auto; 4 or 8 warps per CTA.
  uint32_t scatter_active_sms;  // 0 = auto/all; K = throttled scatter SMs.
  uint32_t locality_domains;    // 0 = auto; 1, 2 (dual-die), 4 (XCD).
  uint32_t sm0_stream_sched;    // 0/1: SM 0 comm warps become scheduler lane.
  uint32_t amd_8wave_min_tok;   // MI308X 8-wave admission floor in tokens.
  uint32_t dispatch_active_sms; // 0 = all; D = only D SMs run dispatch pull.
  uint32_t scatter_owner;       // 0 = epilogue_only; 1 = comm_only.

  // --- Temporal ------------------------------------------------------------
  uint32_t l2_kblock_dep;       // 0/1: tight 1 L2 <-> 1 L1 wave schedule.
  uint32_t ring_blocks;         // 0 = auto live-pool ring; N = forced blocks.
  uint32_t early_stream_reduce; // 0/1: barrier-free per-token early reduce.
  uint32_t tmem_pipeline_mode;  // 0 auto; 100, 175, 200 permille stages.
  uint32_t head_pull_chunk;     // 0 = one prefix; N = tokens per chunk.
  uint32_t combine_sync_chunk;  // 0 = global barrier; G = chunked dedup sync.
  uint32_t epi_reg_stages;      // 0 = auto; 1 or 2 register stages.
  uint32_t sched_l2_ratio;      // 1..4: L2-to-L1 interleaved schedule ratio.

  // --- Implementation ------------------------------------------------------
  uint32_t adapt_bm;            // 0/1: per-expert dynamic BM ladder.
  uint32_t block_m_l1;          // 0 = auto; GEMM1 per-expert BM envelope.
  uint32_t block_m_l2;          // 0 = auto; GEMM2 per-expert BM envelope.
  uint32_t store_block_m_l1;    // 0 = auto; GEMM1 epilogue store height.
  uint32_t store_block_m_l2;    // 0 = auto; GEMM2 epilogue store height.
  uint32_t local_scatter_dedup; // 0/1: bf16x2 L2 atomic dedup before send.
  uint32_t scatter_staging;     // 0 = reg_2stage; 1 = smem; 2 = hbm_dedup.
  uint32_t reg_balance;         // 0/1: SM103 register rebalance (64/56/192).
};
WEAVE_HOST_DEVICE constexpr uint32_t default_r_mask_for_arch(
    HardwareArch arch) {
  return arch == HardwareArch::B300_SM100
      ? encode_r_mask(4u, 8u, 4u, false)
      : (arch == HardwareArch::H20_SM90
             ? encode_r_mask(2u, 16u, 5u, false)
             : encode_r_mask(0u, 8u, 8u, false));
}

WEAVE_HOST_DEVICE constexpr WeaveKnobs default_weave_knobs(
    HardwareArch arch,
    PrecisionMode precision =
        static_cast<PrecisionMode>(WEAVE_PRECISION_MODE)) {
  // Presets 1/2/3 correspond to the -S, -S-T and -S-T-C ablations: the
  // runtime spatial search is disabled and a fixed per-architecture spatial
  // candidate is used.  Preset 0 enables the full CTA-0 cost model.
  const uint32_t ablation_override_mask =
      (WEAVE_ABLATION_PRESET == 0) ? 0u : 3u;  // spatial | scatter
  const uint32_t ablation_r_mask =
      (WEAVE_ABLATION_PRESET == 0) ? kAutoU32 : default_r_mask_for_arch(arch);
  const uint32_t ablation_scatter =
      (WEAVE_ABLATION_PRESET == 0) ? kAutoU32 : WEAVE_SCATTER_MODE;
  return WeaveKnobs{
      WEAVE_ENABLE_COST_MODEL,
      ablation_override_mask, ablation_r_mask, ablation_scatter, kAutoU32,
      64u,
      static_cast<uint32_t>(precision),
      WEAVE_DISPATCH_WARPS,
      WEAVE_SCATTER_ACTIVE_SMS,
      WEAVE_LOCALITY_DOMAINS,
      WEAVE_SM0_STREAM_SCHED,
      WEAVE_AMD_8WAVE_MIN_TOK,
      WEAVE_DISPATCH_ACTIVE_SMS,
      WEAVE_SCATTER_OWNER,
      WEAVE_L2_KBLOCK_DEP,
      WEAVE_RING_BLOCKS,
      WEAVE_EARLY_STREAM_REDUCE,
      WEAVE_TMEM_PIPELINE_MODE,
      WEAVE_HEAD_PULL_CHUNK,
      WEAVE_COMBINE_SYNC_CHUNK,
      WEAVE_EPI_REG_STAGES,
      WEAVE_SCHED_L2_RATIO,
      WEAVE_ADAPT_BM,
      WEAVE_BLOCK_M_L1,
      WEAVE_BLOCK_M_L2,
      WEAVE_STORE_BLOCK_M_L1,
      WEAVE_STORE_BLOCK_M_L2,
      WEAVE_LOCAL_SCATTER_DEDUP,
      WEAVE_SCATTER_STAGING,
      WEAVE_REG_BALANCE,
  };
}

// ---------------------------------------------------------------------------
// Host-side resolvers.  A zero value means "auto" and resolves to the
// architecture-specific default; a non-zero value is a validated override.
// ---------------------------------------------------------------------------
WEAVE_HOST_DEVICE constexpr uint32_t resolve_dispatch_warps(
    const WeaveKnobs& knobs, uint32_t auto_warps) {
  return knobs.dispatch_warps == 0u ? auto_warps : knobs.dispatch_warps;
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_scatter_active_sms(
    const WeaveKnobs& knobs, uint32_t num_sms) {
  return (knobs.scatter_active_sms == 0u ||
          knobs.scatter_active_sms >= num_sms)
      ? num_sms : knobs.scatter_active_sms;
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_locality_domains(
    const WeaveKnobs& knobs, HardwareArch arch) {
  if (knobs.locality_domains != 0u) return knobs.locality_domains;
  return arch == HardwareArch::MI308X_GFX942 ? 4u :
         (arch == HardwareArch::B300_SM100 ? 2u : 1u);
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_amd_8wave_min_tok(
    const WeaveKnobs& knobs, uint32_t auto_tokens = WEAVE_DUALGROUP_MIN_TOKENS) {
  return knobs.amd_8wave_min_tok == 0u ? auto_tokens : knobs.amd_8wave_min_tok;
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_ring_blocks(
    const WeaveKnobs& knobs, uint32_t auto_blocks) {
  return knobs.ring_blocks == 0u ? auto_blocks : knobs.ring_blocks;
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_tmem_pipeline_mode(
    const WeaveKnobs& knobs, uint32_t auto_mode) {
  return knobs.tmem_pipeline_mode == 0u ? auto_mode : knobs.tmem_pipeline_mode;
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_dispatch_active_sms(
    const WeaveKnobs& knobs, uint32_t num_sms) {
  return (knobs.dispatch_active_sms == 0u ||
          knobs.dispatch_active_sms >= num_sms)
      ? num_sms : knobs.dispatch_active_sms;
}
// Auto register pipeline: two stages where the verified SM103 register
// prefetch pays off (BM in {80, 96, 240, 256} with TMEM depth >= 1.75), else 1.
WEAVE_HOST_DEVICE constexpr uint32_t resolve_epi_reg_stages(
    const WeaveKnobs& knobs, uint32_t block_m, uint32_t tmem_mode) {
  return knobs.epi_reg_stages != 0u ? knobs.epi_reg_stages
      : (((block_m == 80u || block_m == 96u || block_m >= 240u) &&
          tmem_mode >= 175u) ? 2u : 1u);
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_block_m(
    const WeaveKnobs& knobs, uint32_t phase_l1, uint32_t phase_l2,
    uint32_t auto_block_m) {
  const uint32_t requested = phase_l2 != 0u ? phase_l2 : phase_l1;
  return requested == 0u ? auto_block_m : requested;
}
WEAVE_HOST_DEVICE constexpr uint32_t resolve_store_block_m(
    const WeaveKnobs& knobs, uint32_t phase_l1, uint32_t phase_l2,
    uint32_t auto_l1, uint32_t auto_l2) {
  return phase_l2 != 0u ? phase_l2 : (phase_l1 != 0u ? phase_l1 : auto_l2);
}

WEAVE_HOST_DEVICE constexpr bool has_override(
    const WeaveKnobs& knobs, uint32_t bit) {
  return (knobs.override_mask & bit) != 0u;
}

WEAVE_HOST_DEVICE constexpr uint32_t effective_scatter_mode(
    const WeaveKnobs& knobs, uint32_t auto_mode) {
  return has_override(knobs, kOverrideScatter)
      ? knobs.override_scatter_mode
      : auto_mode;
}

WEAVE_HOST_DEVICE constexpr uint32_t effective_r_mask(
    const WeaveKnobs& knobs, uint32_t auto_mask) {
  return has_override(knobs, kOverrideSpatial)
      ? knobs.override_r_mask
      : auto_mask;
}

WEAVE_HOST_DEVICE constexpr uint32_t effective_v_head(
    const WeaveKnobs& knobs, uint32_t auto_v_head) {
  return has_override(knobs, kOverrideVHead)
      ? knobs.override_v_head
      : auto_v_head;
}

}  // namespace weave
