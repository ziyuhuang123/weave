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
  uint32_t enable_runtime_cost_model;
  uint32_t ablation_preset;

  // Explicit overrides for progressive ablation and manual tuning.
  uint32_t override_mask;
  uint32_t override_r_mask;
  uint32_t override_scatter_mode;
  uint32_t override_v_head;

  // Spatial defaults.
  uint32_t comm_units;
  uint32_t comp_units;
  uint32_t dualgroup_min_tokens;
  uint32_t bm_tier_mode;
  uint32_t block_m_fixed;
  uint32_t block_n1;
  uint32_t block_n2;
  uint32_t block_k;
  uint32_t stage2_n_group;
  uint32_t heterogeneous_sm_mask;

  // Temporal defaults.
  uint32_t overlap_pipeline;
  uint32_t scatter_mode;
  uint32_t gemm_stages;
  uint32_t all_wave_combine;
  uint32_t spin_before_sleep;
  uint32_t tail_fine_grained;
  uint32_t v_head;

  // Dataflow defaults.
  uint32_t selective_dedup;
  uint32_t register_direct_comm;
  uint32_t parallel_route_plan;
  uint32_t local_first_order;
  uint32_t fused_input_quant;
  uint32_t fused_swiglu_requant;
  uint32_t profiler_mode;
  uint32_t precision_mode;
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
    HardwareArch arch, uint32_t fused_input_quant = 0u,
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
      WEAVE_ABLATION_PRESET,
      ablation_override_mask, ablation_r_mask, ablation_scatter, kAutoU32,
      WEAVE_COMM_UNITS,
      WEAVE_COMP_UNITS,
      WEAVE_DUALGROUP_MIN_TOKENS,
      WEAVE_BM_TIER_MODE,
      64u,
      WEAVE_BN1,
      WEAVE_BN2_MODE,
      WEAVE_BK,
      WEAVE_S2_NGROUP_MODE,
      WEAVE_HETERO_SM_MASK,
      WEAVE_OVERLAP_PIPELINE,
      WEAVE_SCATTER_MODE,
      WEAVE_GEMM_STAGES,
      WEAVE_ALL_WAVE_COMBINE,
      WEAVE_SPIN_BEFORE_SLEEP,
      WEAVE_TAIL_FINE_GRAINED,
      WEAVE_V_HEAD,
      WEAVE_SELECTIVE_DEDUP,
      WEAVE_REGISTER_DIRECT_COMM,
      WEAVE_PARALLEL_ROUTE_PLAN,
      WEAVE_LOCAL_FIRST_ORDER,
      fused_input_quant,
      WEAVE_FUSED_SWIGLU_REQUANT,
      0u,
      static_cast<uint32_t>(precision)};
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
