#pragma once
// =============================================================================
// Weave / MegaMoE Unified Cross-Hardware Tuning Knobs Specification
// Shared across NVIDIA Blackwell B300 (sm_100a/sm_103a), NVIDIA Hopper H20
// (sm_90a), and AMD Instinct MI308X (gfx942).
//
// Organizes all compile-time and runtime knobs into the three orthogonal
// dimensions of the paper (Sections III-B, III-C, III-D, and Table IV):
//   1. Spatial Dimension (空间维度): Warp/Wave role ratio, intra-SM/CU split,
//      cross-SM/CU heterogeneous mask, dynamic dual-group transition, and
//      per-expert adaptive tile sizes (BM tiers, BN1, BN2, BK, N-grouping).
//   2. Temporal Dimension (时间维度): Intra-SM/CU pipeline overlap schedule,
//      Stage-2 scatter executor path (epilogue vs comm/queue), GEMM software
//      pipeline depth, cooperative tail combine waves, and spin-sleep backoff.
//   3. Operator Fusion & Dataflow Dimension (单算子融合与去重底座):
//      Cross-GPU selective token deduplication, register-direct zero-copy
//      gather (0 B comm SMEM/LDS), parallel multi-CTA route planning,
//      local-first ready-tile worklist, on-the-fly input quantization,
//      in-core SwiGLU + blockscale requant, and fine-grained home reduction.
// =============================================================================

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define WEAVE_HOST_DEVICE __attribute__((host, device, always_inline)) inline
#else
#define WEAVE_HOST_DEVICE inline
#endif

// -----------------------------------------------------------------------------
// Progressive Ablation Presets (Paper Section IV-D, Table IV)
//   0 = WEAVE_PRESET_CUSTOM / FULL (All 3 pillars enabled + Cost-Model tuned)
//   1 = WEAVE_PRESET_MINUS_S       (Disable Spatial adaptive warp/tile sizing)
//   2 = WEAVE_PRESET_MINUS_S_T     (Also disable Temporal overlap & role switch)
//   3 = WEAVE_PRESET_MINUS_S_T_C   (Use static scatter; BM tiers and dedup stay on)
// -----------------------------------------------------------------------------
#ifndef WEAVE_ABLATION_PRESET
#define WEAVE_ABLATION_PRESET 0
#endif
#if WEAVE_ABLATION_PRESET < 0 || WEAVE_ABLATION_PRESET > 3
#error "WEAVE_ABLATION_PRESET must be 0..3"
#endif
#define WEAVE_ENABLE_SPATIAL_ADAPTATION (WEAVE_ABLATION_PRESET == 0)
#define WEAVE_ENABLE_TEMPORAL_OVERLAP (WEAVE_ABLATION_PRESET < 2)
#define WEAVE_ENABLE_ONLINE_COST_MODEL (WEAVE_ABLATION_PRESET < 3)

// =============================================================================
// 1. SPATIAL DIMENSION KNOBS (空间维度旋钮)
// =============================================================================

// Number of communication vs compute warps/waves per CTA at kernel entry.
// - MI308X (8 waves = 512 threads): default (4 comm, 4 comp)
// - H20    (16-20 warps):           (2D/6D comm, 2N TMA, 8E/16E math+epi)
// - B300   (16 warps = 4 WGs):      (4D+4N+8E) <-> (8D+4N+4E)
#ifndef WEAVE_KNOB_COMM_UNITS
#define WEAVE_KNOB_COMM_UNITS 4u
#endif

#ifndef WEAVE_KNOB_COMP_UNITS
#define WEAVE_KNOB_COMP_UNITS 4u
#endif

// Dynamic Dual-Group Role Transition Threshold (local tokens per GPU `t_local`):
// Once Dispatch finishes, if `t_local >= WEAVE_KNOB_DUALGROUP_MIN_TOKENS`, the
// communication wave-group dynamically rebinds into a second compute wave-group
// (expanding 4 compute waves -> 8 compute waves per CU/SM).
// Set to 0xFFFFFFFFu to keep static 4+4 roles (-S ablation).
#ifndef WEAVE_KNOB_DUALGROUP_MIN_TOKENS
#if WEAVE_ABLATION_PRESET >= 1
#define WEAVE_KNOB_DUALGROUP_MIN_TOKENS 0xFFFFFFFFu
#else
#define WEAVE_KNOB_DUALGROUP_MIN_TOKENS 64u
#endif
#endif

// Adaptive Per-Expert BLOCK_M Tier Policy:
//   EP16 production fixes this at mode 2 in every arm. The EP16 tile planner
//   interprets mode 2 as {32, 64, 128, 256}, minimizing row padding.
#ifndef WEAVE_KNOB_BM_TIER_MODE
#define WEAVE_KNOB_BM_TIER_MODE 2
#endif
#if WEAVE_KNOB_BM_TIER_MODE != 2
#error "EP16 requires per-expert adaptive BM_e in every ablation arm"
#endif

// The EP16 exact tile planner rounds capacity to the smallest multiple of 32
// that covers m_e, then decomposes that capacity into {32,64,128,256} tiles.
// This setting is retained for source compatibility and is not used by EP16.
#ifndef WEAVE_ROUNDUP_SINGLE_TILE
#define WEAVE_ROUNDUP_SINGLE_TILE 0
#endif

// Stage-1 & Stage-2 GEMM Tile Shapes (BN1, BN2, BK) and Stage-2 N-Column Grouping:
// - WEAVE_KNOB_BN2_MODE: 0 = Auto (selects BN2=64 at small/mid grid to double CU
//   parallelism, BN2=128 at large grid), 64 = Force BN2=64, 128 = Force BN2=128.
// - WEAVE_KNOB_S2_NGROUP_MODE: 0 = Auto {1, 2, 4}, or explicit {1, 2, 4}.
#ifndef WEAVE_KNOB_BN1
#define WEAVE_KNOB_BN1 128u
#endif

#ifndef WEAVE_KNOB_BN2_MODE
#if WEAVE_ABLATION_PRESET >= 1
#define WEAVE_KNOB_BN2_MODE 128u
#else
#define WEAVE_KNOB_BN2_MODE 0u
#endif
#endif

#ifndef WEAVE_KNOB_BK
#define WEAVE_KNOB_BK 128u
#endif

#ifndef WEAVE_KNOB_S2_NGROUP_MODE
#if WEAVE_ABLATION_PRESET >= 1
#define WEAVE_KNOB_S2_NGROUP_MODE 1u
#else
#define WEAVE_KNOB_S2_NGROUP_MODE 0u
#endif
#endif

// Cross-SM/CU Heterogeneous Warp Ratio Mask (%smid / CU parity mixing):
// 0 = Homogeneous across all SMs/CUs; 1 = 50/50 high-comm / pure-compute mix.
#ifndef WEAVE_KNOB_HETERO_SM_MASK
#define WEAVE_KNOB_HETERO_SM_MASK 0u
#endif

// =============================================================================
// 2. TEMPORAL DIMENSION KNOBS (时间维度旋钮)
// =============================================================================

// Intra-SM/CU Pipeline Overlap Mode:
//   1 = Fine-grained tile-ready overlap (Dispatch overlaps RoutePlan + GEMM1;
//       GEMM2 overlaps Scatter + Combine, only 1 entryCTA barrier).
//   0 = Bulk-synchronous stage serialization (-S-T ablation: waits for full
//       Dispatch completion before starting GEMM1, and full GEMM2 completion
//       before starting Scatter/Combine).
#ifndef WEAVE_KNOB_OVERLAP_PIPELINE
#if WEAVE_ABLATION_PRESET >= 2
#define WEAVE_KNOB_OVERLAP_PIPELINE 0
#else
#define WEAVE_KNOB_OVERLAP_PIPELINE 1
#endif
#endif

// Stage-2 Remote Scatter Execution Path (`s in {epi, comm}` in Paper Eq. 2):
//   0 = Inline Epilogue Scatter (`s = epi`)
//   1 = Decoupled Comm/Worker Queue Scatter (`s = comm`, hoisted lock-free queue)
//   2 = Local HBM/L2 Dedup Pre-Reduce before Scatter (`K / EP > 1` regime)
#ifndef WEAVE_KNOB_SCATTER_MODE
#define WEAVE_KNOB_SCATTER_MODE 1u
#endif

// GEMM Mainloop Software Pipeline Depth:
//   1 = Single-stage synchronous
//   2 = Double-buffered register + LDS/SMEM prefetch (MI308X 64 KB LDS optimal)
//   3..4 = Multi-stage async TMA/VMEM ring (H20 / B300 228 KB SMEM optimal)
#ifndef WEAVE_KNOB_GEMM_STAGES
#define WEAVE_KNOB_GEMM_STAGES 2u
#endif

// Cooperative Tail Combine Active Waves per CTA (4 vs all 8 waves / 16 warps):
#ifndef WEAVE_KNOB_ALL_WAVE_COMBINE
#if WEAVE_ABLATION_PRESET >= 2
#define WEAVE_KNOB_ALL_WAVE_COMBINE 0
#else
#define WEAVE_KNOB_ALL_WAVE_COMBINE 1
#endif
#endif

// Polling Spin-Before-Sleep Threshold (balances wakeup latency vs issue-slot contention):
#ifndef WEAVE_KNOB_SPIN_BEFORE_SLEEP
#define WEAVE_KNOB_SPIN_BEFORE_SLEEP 256u
#endif

// =============================================================================
// 3. OPERATOR FUSION & DATAFLOW KNOBS (单算子融合与去重维度旋钮)
// =============================================================================

// Cross-GPU Selective Token Deduplication (always enabled in EP16 Table IV):
//   1 = Send at most 1 copy of token x_i per unique destination rank r (cuts
//       inter-GPU Dispatch traffic by up to 48% at K=8..10, EP=8).
//   0 = Naive per-route duplicate transmission (for non-EP16 experiments only).
#ifndef WEAVE_KNOB_SELECTIVE_DEDUP
#define WEAVE_KNOB_SELECTIVE_DEDUP 1
#endif
#if WEAVE_KNOB_SELECTIVE_DEDUP != 1
#error "EP16 requires selective input deduplication in every ablation arm"
#endif

// Register-Direct Zero-Copy Gather & Dispatch (0 B comm LDS/SMEM, 0 B dense_a):
#ifndef WEAVE_KNOB_REGISTER_DIRECT_COMM
#define WEAVE_KNOB_REGISTER_DIRECT_COMM 1
#endif

// All-CTA Parallel Route Histogram, Prefix-Scan & Scatter:
#ifndef WEAVE_PAR_RP
#define WEAVE_PAR_RP 1
#endif

// Local-First Three-Tier Worklist Priority (LOCAL_READY -> REMOTE_PRIMARY -> TAIL):
#ifndef WEAVE_KNOB_LOCAL_FIRST_ORDER
#define WEAVE_KNOB_LOCAL_FIRST_ORDER 1
#endif

// On-the-Fly Fused BF16 -> FP8 Blockscale Quantization during Dispatch:
// Controlled dynamically at runtime via `envelope.payload_bytes == hidden` (0=prequant,
// 1=on-the-fly BF16->FP8 inside Dispatch waves) or forced via host CLI `--input-dtype`.
#ifndef WEAVE_KNOB_SUPPORT_FUSED_INPUT_QUANT
#define WEAVE_KNOB_SUPPORT_FUSED_INPUT_QUANT 1
#endif

// Fused Stage-1 SwiGLU + Per-128 Blockscale Dynamic Requantization:
#ifndef WEAVE_KNOB_FUSED_SWIGLU_REQUANT
#define WEAVE_KNOB_FUSED_SWIGLU_REQUANT 1
#endif

namespace weave {

enum class HardwareArch : uint32_t {
  B300_SM100  = 0,
  H20_SM90    = 1,
  MI308X_GFX942 = 2,
};

enum class ScatterPath : uint32_t {
  EpilogueInline = 0,  // s = epi
  CommQueueAsync = 1,  // s = comm
  LocalDedupFirst = 2, // Mode 2 (K / EP > 1)
};

struct KnobConfig {
  // Spatial
  uint32_t comm_units;
  uint32_t comp_units;
  uint32_t dualgroup_min_tokens;
  uint32_t bm_tier_mode;       // EP16 mode 2 = {32,64,128,256}, all arms
  uint32_t block_m_fixed;      // used when bm_tier_mode == 0
  uint32_t block_n1;
  uint32_t block_n2;           // 0=auto, 64, 128, 256
  uint32_t block_k;
  uint32_t stage2_n_group;     // 0=auto, 1, 2, 4
  uint32_t hetero_sm_mask;
  // Temporal
  uint32_t overlap_pipeline;   // 0=bulk-sync, 1=overlapped
  uint32_t scatter_mode;       // 0=epi, 1=comm, 2=local_dedup
  uint32_t gemm_stages;
  uint32_t all_wave_combine;
  uint32_t spin_before_sleep;
  // Operator / Dataflow
  uint32_t selective_dedup;
  uint32_t register_direct_comm;
  uint32_t parallel_route_plan;
  uint32_t local_first_order;
  uint32_t fused_input_quant;
  uint32_t fused_swiglu_requant;
};

WEAVE_HOST_DEVICE KnobConfig default_knobs_from_macros(uint32_t fused_input_quant = 0u) {
  KnobConfig cfg{};
  cfg.comm_units            = WEAVE_KNOB_COMM_UNITS;
  cfg.comp_units            = WEAVE_KNOB_COMP_UNITS;
  cfg.dualgroup_min_tokens  = WEAVE_KNOB_DUALGROUP_MIN_TOKENS;
  cfg.bm_tier_mode          = WEAVE_KNOB_BM_TIER_MODE;
  cfg.block_m_fixed         = 64u;
  cfg.block_n1              = WEAVE_KNOB_BN1;
  cfg.block_n2              = WEAVE_KNOB_BN2_MODE;
  cfg.block_k               = WEAVE_KNOB_BK;
  cfg.stage2_n_group        = WEAVE_KNOB_S2_NGROUP_MODE;
  cfg.hetero_sm_mask        = WEAVE_KNOB_HETERO_SM_MASK;
  cfg.overlap_pipeline      = WEAVE_KNOB_OVERLAP_PIPELINE;
  cfg.scatter_mode          = WEAVE_KNOB_SCATTER_MODE;
  cfg.gemm_stages           = WEAVE_KNOB_GEMM_STAGES;
  cfg.all_wave_combine      = WEAVE_KNOB_ALL_WAVE_COMBINE;
  cfg.spin_before_sleep     = WEAVE_KNOB_SPIN_BEFORE_SLEEP;
  cfg.selective_dedup       = WEAVE_KNOB_SELECTIVE_DEDUP;
  cfg.register_direct_comm  = WEAVE_KNOB_REGISTER_DIRECT_COMM;
  cfg.parallel_route_plan   = WEAVE_PAR_RP;
  cfg.local_first_order     = WEAVE_KNOB_LOCAL_FIRST_ORDER;
  cfg.fused_input_quant     = fused_input_quant;
  cfg.fused_swiglu_requant  = WEAVE_KNOB_FUSED_SWIGLU_REQUANT;
  return cfg;
}

}  // namespace weave
