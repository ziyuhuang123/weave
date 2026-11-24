#pragma once
// =============================================================================
// Weave CTA-0 on-device cost model.
//
// This header is the executable form of Paper Section III-C:
//   Eq. (1): F_gemm(r), V_mem(r)
//   Eq. (2): T_mma(r), T_epi(N_e), T_comp(r)
//   Eq. (3): T(r, epi), T(r, comm)
//
// All arithmetic is float.  CTA 0 reads the live post-dispatch histogram
// m_e[0..E_local), derives M = sum_e m_e, and evaluates the candidate set for
// the architecture that compiled this translation unit.
// =============================================================================

#include <math.h>
#include <stdint.h>

#include <deep_gemm/scheduler/weave_knobs.cuh>

namespace weave {

// Compile-time defaults consumed by the host-side golden tests and by
// documentation tooling. Runtime selection uses WeaveKnobs fields, whose
// values originate from these same public macros.
static constexpr uint32_t kCompiledPrecisionMode = WEAVE_PRECISION_MODE;
static constexpr uint32_t kCompiledAblationPreset = WEAVE_ABLATION_PRESET;
static constexpr uint32_t kCompiledCostModelDefault = WEAVE_ENABLE_COST_MODEL;
static constexpr uint32_t kCompiledCommUnitsDefault = WEAVE_COMM_UNITS;
static constexpr uint32_t kCompiledCompUnitsDefault = WEAVE_COMP_UNITS;
static constexpr uint32_t kCompiledVHeadDefault = WEAVE_V_HEAD;
static constexpr uint32_t kCompiledScatterModeDefault = WEAVE_SCATTER_MODE;
static constexpr uint32_t kCompiledOverlapPipelineDefault = WEAVE_OVERLAP_PIPELINE;
static constexpr uint32_t kCompiledSelectiveDedupDefault = WEAVE_SELECTIVE_DEDUP;
static constexpr uint32_t kCompiledAllWaveCombineDefault = WEAVE_ALL_WAVE_COMBINE;

static constexpr int kMaxSpatialCandidates = 4;
static constexpr uint32_t kPolicyMagic = 0x57415645u;  // "WEAVE"
static constexpr uint32_t kPolicyVersion = 4u;
// Live post-dispatch histogram capacity of PolicyControl.  E_local <= 256
// covers every supported MoE layout (256 experts at EP1, 512 at EP2, ...); a
// larger E_local is rejected by cta0_select_from_packed_hist() rather than
// silently overflowing the control header.
static constexpr uint32_t kMaxRuntimeExperts = 256u;

// Per-warp fine-grained Reduce streaming rate in GB/s.  This is the same
// calibrated unit rate the Eq. (3) tail term uses, reused for the dedicated
// WEAVE_REDUCE_WARPS lanes so both paths share one bandwidth constant.
static constexpr float kReduceGbpsPerWarp = 0.55f;

struct HardwareSpec {
  HardwareArch arch;
  uint32_t num_sms_or_cus;
  uint32_t max_units_per_sm;

  float p_tc_full_tflops;
  float p_tc_half_tflops;
  float b_hbm_gbps;
  float b_link_gbps;

  float b_pull_per_unit_gbps;
  float b_scat_comm_per_unit_gbps;
  float b_scat_epi_per_unit_gbps;
  float b_epi_write_per_unit_gbps;

  float l2_cache_mb;
  float t_head_us;
  float t_tail_us;
  float t_handoff_us;
  float weight_bytes;
  float act_in_bytes;
  float out_bytes;
  uint32_t head_tasks_per_comm_unit;
};

struct WorkloadSpec {
  uint32_t local_experts;
  uint32_t hidden;
  uint32_t intermediate;
};

struct SpatialCandidate {
  uint32_t r_mask;
  uint32_t comm_units;
  uint32_t comp_units;
  uint32_t block_m;
  uint32_t role_groups;
  float p_tc_tflops;
  bool heterogeneous;
};

struct CandidateSet {
  SpatialCandidate items[kMaxSpatialCandidates];
  uint32_t count;
};

struct EquationBreakdown {
  uint32_t r_mask;
  uint32_t scatter_mode;
  uint32_t total_tokens;
  uint32_t v_head;
  float f_gemm_flops;
  float v_mem_bytes;
  float v_epi_bytes;
  float t_mma_us;
  float t_epi_us;
  float t_comp_us;
  float t_pull_us;
  float t_scat_us;
  float t_tail_reduce_us;
  float t_mid_us;
  float t_total_us;
};

struct PolicyDecision {
  uint32_t magic;
  uint32_t version;
  uint32_t arch;
  uint32_t valid;

  uint32_t selected_candidate;
  uint32_t selected_r_mask;
  uint32_t selected_scatter_mode;
  uint32_t selected_v_head;

  uint32_t selected_comm_units;
  uint32_t selected_comp_units;
  uint32_t selected_block_m;
  uint32_t selected_role_groups;

  float predicted_mid_us;
  float predicted_total_us;
  float predicted_comp_us;
  float predicted_pull_us;
  float predicted_scat_us;
  uint32_t selection_cycles;   // CTA-0 clock64() cycles, not microseconds
  uint32_t input_total_tokens;  // M = sum_e m_e, live post-dispatch
  uint32_t input_dedup_pulls;   // D_in supplied to Eq. (3)
  uint32_t input_scatter_rows;  // M_out supplied to Eq. (3)
};

// ---------------------------------------------------------------------------
// Two-Level framework, Level 1: offline hardware calibration constants.
//
// Calibrated by deep_gemm/mega/calibrate.py from the real verified 3-hardware
// measurements (results/hardware_calibration_3hw.json records every source
// sha256 and derivation rule; zero fabricated numbers):
//   SM103   results/sm103_ep4_prior_verified.json (B8 710.53 us; nsys HBM/TMA
//           mean 76.0 %, Tensor Active median 38.0 % of the 4200 TFLOPS peak)
//   H20     pure21_s1024.json (p15_adapt_bm 2515.232 us = 1.28591x; rank0
//           routing total=2482 max=131 nonzero=123 dedup_rows=964; p19 dedup
//           regressed at every measured mu in [2.4189, 2.9311])
//   MI308X  S4_MI308X_FINAL.json (allopt_c 2316.36/2341.10 us; PMC compute
//           max 16.4 % of 606.2 T, memory max 70.72 % of 3684 GB/s)
// The float literals below are the binary32 round-trip of the calibrated
// values; tests/test_two_level_adaptive_policy.py enforces bit-exact parity
// with calibrate.py / cost_model.py.
// ---------------------------------------------------------------------------
struct HardwareCalibration {
  float f_eff_tflops;         // sustained TC rate at the verified MoE optimum
  float b_hbm_gbps;           // HBM roofline cap (Table I)
  float b_comm_gbps;          // communication link cap (Table I)
  float tau_sync_us;          // head synchronization constant (Table I t_head)
  float mu_dedup_thresh;      // rank-hit multiplicity where dedup pays off
  float hbm_headroom_thresh;  // Middle HBM util below which Reduce lanes fit
  float tail_ratio_thresh;    // tail Combine fraction above which Reduce pays
  uint32_t v_head_hot_budget; // auto-V_HEAD head budget under hot-expert skew
  uint32_t bm_pad_slack;      // 32-row padding slack of the per-expert BM ladder
};

WEAVE_HOST_DEVICE HardwareCalibration get_hardware_calibration(
    HardwareArch arch) {
  if (arch == HardwareArch::B300_SM100) {
    // SM103: f_eff = 4200 x 38.0 %; headroom = measured HBM/TMA mean 76.0 %;
    // tail = Eq.(3) at the verified S_rank=1024 EP4 reference (Zipf proxy);
    // mu = (1286+8954)/(1286+5304) of the recorded routing replay S=1024 cell.
    return HardwareCalibration{
        1596.0f, 7680.0f, 780.0f, 14.0f,
        1.55386949f, 0.75999999f, 0.295702934f, 32u, 0u};
  }
  if (arch == HardwareArch::H20_SM90) {
    // H20: f_eff = Eq.(1) unpadded F_gemm(rank0) / p15 median 2515.232 us;
    // headroom/tail = Eq.(2)/(3) at the rank0 max-anchored reference proxy;
    // mu = ceil(max measured mean_hits 2.9311) = 3 (p19 dedup regressed);
    // hot budget 64 = p09_v_head measured best temporal arm.
    return HardwareCalibration{
        99.3333054f, 3600.0f, 370.0f, 18.0f,
        3.0f, 0.455443323f, 0.0238960907f, 64u, 0u};
  }
  // MI308X: f_eff = 606.2 T x PMC compute max 16.4 %; headroom = PMC memory
  // max 70.72 %; tail = Eq.(3) at the uniform S1024 EP8 reference; mu =
  // (K/ep)/(1-C(E-E/ep,K)/C(E,K)) of the verified allopt_c workload (dedup=1).
  return HardwareCalibration{
      99.4168015f, 3684.0f, 287.200012f, 22.0f,
      1.68861687f, 0.707199991f, 0.0550832227f, 16u, 0u};
}

// ---------------------------------------------------------------------------
// Two-Level framework, Level 2: online CTA-0 routing-adaptive policy.
//
// Published in the global PolicyControl (never in SMEM: zero SMEM increase).
// Tier-3 outputs act on the kernel per step (active_reduce_warps,
// active_v_head); the ready-group / dedup / BM-tier fields are the live
// verdicts on the Tier-1 structural requests (explicit pins always override).
// ---------------------------------------------------------------------------
struct OnlineRoutingPolicy {
  uint32_t active_reduce_warps;     // 0 or the compiled R_max (runtime gate)
  uint32_t active_v_head;           // head pull budget (auto adapts to skew)
  uint32_t active_ready_group_size; // 1/2/4/8 publication group verdict
  uint32_t enable_dedup_runtime;    // live mu verdict vs mu_dedup_thresh
  uint32_t bm_tier_mean;            // per-expert BM ladder tier for mean_n
  uint32_t bm_envelope;             // BLOCK_M envelope the tier was cut from
  uint32_t active_experts;          // nonzero bins of the live histogram
  uint32_t max_n;                   // hottest expert token count
  uint32_t total_routes;            // M = sum_e m_e
  uint32_t adaptive_active;         // 1 iff WEAVE_ADAPT_BM=1 (online tier on)
  uint32_t reduce_gate_hbm_ok;      // middle_hbm_util <= hbm_headroom_thresh
  uint32_t reduce_gate_tail_ok;     // tail_combine_frac >= tail_ratio_thresh
  float mean_n;                     // M / active_experts
  float rank_hit_multiplicity_mu;   // M / unique_tokens (live dedup verdict)
  float middle_hbm_util;            // Eq.(2): V_mem / (T_comp * B_hbm)
  float tail_combine_frac;          // Eq.(3): T_tail_reduce / T_total

  // --- 3-Tier upgrade: independent per-(Phase,N,K) profile publication -----
  // Appended after the legacy fields (backward compatible: every existing
  // field keeps its offset and semantics; pinned / non-adaptive builds
  // publish zeros here and keep their compiled structure).  Global memory
  // only -- the SMEM PolicyDecision copy is unchanged (zero SMEM increase).
  // Consumers read ONE 4-byte code per expert via load_expert_profile_code()
  // in weave_policy_runtime.cuh; nothing copies the full LUT per step.
  uint32_t phase_profile_active;    // 1 iff the adaptive tier published LUTs
  uint32_t l1_envelope_code;        // packed PhaseEnvelope of GEMM1 (N1=2I,K1=H)
  uint32_t l2_envelope_code;        // packed PhaseEnvelope of GEMM2 (N2=H,K2=I)
  uint32_t l1_profile_code[kMaxRuntimeExperts];  // per-expert GEMM1 inner code
  uint32_t l2_profile_code[kMaxRuntimeExperts];  // per-expert GEMM2 inner code
};

struct alignas(64) PolicyControl {
  // Written only by CTA 0.  Readers use acquire loads after observing epoch.
  uint32_t magic;
  uint32_t version;
  uint32_t epoch;
  uint32_t published_epoch;
  uint32_t status;
  uint32_t stats_ready_epoch;
  uint32_t decision_epoch;
  uint32_t reserved0;
  PolicyDecision decision;
  // Two-Level Tier-3 policy, published alongside the decision.  Global memory
  // only: the SMEM PolicyDecision copy is unchanged (zero SMEM increase).
  OnlineRoutingPolicy online;

  // CTA 0 copies the live post-dispatch histogram here so the control header is
  // self-contained and can be profiled without chasing a second allocation.
  uint32_t histogram[kMaxRuntimeExperts];
};

WEAVE_HOST_DEVICE constexpr uint32_t ceil_div_u32(uint32_t n, uint32_t d) {
  return d == 0u ? 0u : (n + d - 1u) / d;
}

WEAVE_HOST_DEVICE constexpr uint32_t align_up_u32(uint32_t n, uint32_t a) {
  return a == 0u ? n : ceil_div_u32(n, a) * a;
}

// Tensor-core throughput and byte-width scaling for the precision knob.  The
// FP8xFP4 row is exactly Paper Table I; every other row leaves that default
// untouched.  b_a/b_w include block-scale metadata:
//   FP8xFP4: FP8 activation with UTCCP-packed SF, measured as 0.5625
//            B/element in Paper Table I; weights are FP4 (0.5 B/element).
//   NVFP4:   E2M1 4-bit data (0.5 B) + E4M3 SF 1/16*1 B = 0.5625 B/element.
//   FP8xFP8: E4M3 data (1 B) + FP32 SF 1/128*4 B = 1.03125 B/element.
//   BF16:    2 B data + FP32 SF 1/128*4 B = 2.03125 B/element.
WEAVE_HOST_DEVICE constexpr float precision_byte_width(
    HardwareArch arch, PrecisionMode precision, bool activation) {
  (void)arch;  // reserved for future per-arch SF packing widths
  if (precision == PrecisionMode::FP8xFP4) {
    return activation ? 0.5625f : 0.5f;
  }
  if (precision == PrecisionMode::NVFP4) {
    return 0.5625f;
  }
  if (precision == PrecisionMode::FP8xFP8) {
    return 1.03125f;
  }
  return activation ? 2.03125f : 2.0f;
}

WEAVE_HOST_DEVICE constexpr float precision_tc_scale(
    PrecisionMode precision) {
  // B300 NVFP4 tensor cores expose 2x the FP8xFP4 rate; BF16 exposes 0.5x.
  return precision == PrecisionMode::NVFP4 ? 2.0f :
         precision == PrecisionMode::BF16 ? 0.5f : 1.0f;
}

WEAVE_HOST_DEVICE HardwareSpec get_hardware_spec(
    HardwareArch arch, PrecisionMode precision = PrecisionMode::FP8xFP4) {
  HardwareSpec hw{};
  if (arch == HardwareArch::B300_SM100) {
    hw = HardwareSpec{
        HardwareArch::B300_SM100, 148u, 16u,
        4200.0f, 3100.0f, 7680.0f, 780.0f,
        1.35f, 1.20f, 0.35f, 8.0f,
        128.0f, 14.0f, 8.0f, 48.0f,
        0.5f, 0.5625f, 2.0f, 64u};
  } else if (arch == HardwareArch::H20_SM90) {
    hw = HardwareSpec{
        HardwareArch::H20_SM90, 78u, 20u,
        255.0f, 210.0f, 3600.0f, 370.0f,
        0.58f, 0.48f, 0.14f, 4.0f,
        60.0f, 18.0f, 9.0f, 0.8f,
        1.0f, 1.03125f, 2.0f, 32u};
  } else {
    hw = HardwareSpec{
        HardwareArch::MI308X_GFX942, 80u, 8u,
        360.0f, 205.0f, 3684.0f, 287.2f,
        0.92f, 0.82f, 0.28f, 8.0f,
        256.0f, 22.0f, 14.0f, 1.5f,
        1.0f, 1.03125f, 2.0f, 16u};
  }
  if (precision != PrecisionMode::FP8xFP4) {
    hw.act_in_bytes = precision_byte_width(arch, precision, true);
    hw.weight_bytes = precision_byte_width(arch, precision, false);
    const float scale = precision_tc_scale(precision);
    hw.p_tc_full_tflops *= scale;
    hw.p_tc_half_tflops *= scale;
  }
  return hw;
}

WEAVE_HOST_DEVICE CandidateSet get_candidate_set(
    HardwareArch arch, PrecisionMode precision = PrecisionMode::FP8xFP4) {
  CandidateSet out{};
  if (arch == HardwareArch::B300_SM100) {
    // Paper Table I: (4 comm, 8 epi) and (8 comm, 4 epi) per SM.
    out.items[0] = SpatialCandidate{
        encode_r_mask(4u, 8u, 4u, false), 4u, 8u, 128u, 4u, 4200.0f, false};
    out.items[1] = SpatialCandidate{
        encode_r_mask(8u, 4u, 4u, true), 8u, 4u, 64u, 4u, 3100.0f, true};
    out.count = 2u;
  } else if (arch == HardwareArch::H20_SM90) {
    // Paper Table I: (2 comm, 16 comp) and (10 comm, 8 comp) per SM.
    out.items[0] = SpatialCandidate{
        encode_r_mask(2u, 16u, 5u, false), 2u, 16u, 128u, 5u, 255.0f, false};
    out.items[1] = SpatialCandidate{
        encode_r_mask(10u, 8u, 5u, true), 10u, 8u, 64u, 5u, 210.0f, true};
    out.count = 2u;
  } else {
    // Paper Section III-D: (0 comm, 8 math) and (4 comm, 4 math) waves.
    out.items[0] = SpatialCandidate{
        encode_r_mask(0u, 8u, 8u, false), 0u, 8u, 128u, 8u, 360.0f, false};
    out.items[1] = SpatialCandidate{
        encode_r_mask(4u, 4u, 8u, false), 4u, 4u, 64u, 8u, 205.0f, false};
    out.count = 2u;
  }
  if (precision != PrecisionMode::FP8xFP4) {
    const float scale = precision_tc_scale(precision);
    for (uint32_t c = 0u; c < out.count; ++c) {
      out.items[c].p_tc_tflops *= scale;
    }
  }
  return out;
}

WEAVE_HOST_DEVICE uint32_t sum_histogram(
    const uint32_t* m_e, uint32_t num_experts) {
  uint32_t total = 0u;
  if (m_e == nullptr) {
    return 0u;
  }
  for (uint32_t e = 0u; e < num_experts; ++e) {
    total += m_e[e];
  }
  return total;
}

WEAVE_HOST_DEVICE uint32_t choose_v_head(
    const HardwareSpec& hw, const SpatialCandidate& candidate,
    uint32_t dedup_pulls) {
  if (dedup_pulls == 0u) {
    return 0u;
  }
  const uint32_t comm_units = candidate.comm_units == 0u
      ? candidate.comp_units : candidate.comm_units;
  const uint32_t window = hw.num_sms_or_cus * comm_units * hw.head_tasks_per_comm_unit;
  return align_up_u32(window < dedup_pulls ? window : dedup_pulls, 32u);
}

// One pass over the live histogram: M, the nonzero-bin count and the hottest
// expert.  Shared by the online policy so Tier 3 adds a single traversal.
WEAVE_HOST_DEVICE void histogram_stats(
    const uint32_t* m_e, uint32_t num_experts,
    uint32_t* total, uint32_t* active, uint32_t* max_n) {
  uint32_t t = 0u, a = 0u, mx = 0u;
  if (m_e != nullptr) {
    for (uint32_t e = 0u; e < num_experts; ++e) {
      const uint32_t n = m_e[e];
      t += n;
      if (n != 0u) {
        ++a;
        mx = n > mx ? n : mx;
      }
    }
  }
  *total = t;
  *active = a;
  *max_n = mx;
}

// Runtime mirror of sched::select_expert_tile_bm<BLOCK_M> (mega_moe.cuh):
// the largest ladder member {32,64,128,256} (plus a non-power-of-two envelope)
// whose 32-row padded size fits min_padded + slack.  Tier-3 publishes the tier
// for the mean expert load; the scheduler applies the same ladder per n_e.
// constexpr so the Tier-1 (Phase,N,K) engine can fold it at compile time.
WEAVE_HOST_DEVICE constexpr uint32_t select_online_expert_bm(
    uint32_t num_tokens, uint32_t envelope, uint32_t slack) {
  if (num_tokens == 0u || envelope % 32u != 0u) {
    return envelope;
  }
  const uint32_t min_padded = ceil_div_u32(num_tokens, 32u) * 32u;
  uint32_t best_bm = 32u;
  if (envelope >= 64u && envelope % 64u == 0u) {
    if (ceil_div_u32(num_tokens, 64u) * 64u <= min_padded + slack) best_bm = 64u;
  }
  if (envelope >= 128u && envelope % 128u == 0u) {
    if (ceil_div_u32(num_tokens, 128u) * 128u <= min_padded + slack) best_bm = 128u;
  }
  if (envelope >= 256u && envelope % 256u == 0u) {
    if (ceil_div_u32(num_tokens, 256u) * 256u <= min_padded + slack) best_bm = 256u;
  }
  if (envelope > 32u && (envelope & (envelope - 1u)) != 0u) {
    if (ceil_div_u32(num_tokens, envelope) * envelope <= min_padded + slack) best_bm = envelope;
  }
  return best_bm;
}

// ---------------------------------------------------------------------------
// 3-Tier upgrade, Tier 1: offline (Hardware, Precision, Phase, N, K) GEMM
// envelope + per-M inner-tuple LUT.
//
// GEMM1 (N1 = 2*I, K1 = H, SwiGLU + quant epilogue) and GEMM2 (N2 = H, K2 = I,
// BF16 weight-mul + NVLink/xGMI scatter epilogue) are solved COMPLETELY
// independently.  The engine is analytical legal pruning + integer scoring
// (zero blind grid search, zero floating point): every step is uint64/uint32
// integer arithmetic, so the Python mirror in deep_gemm/mega/calibrate.py
// (select_phase_envelope / select_phase_inner_profile) is bit-exact by
// construction.  Physics rules, scoring terms and constants are documented
// there; the two implementations must be changed together (the parity gate
// tests/test_two_level_adaptive_policy.py enforces it).
// ---------------------------------------------------------------------------

struct PnkHardware {
  uint32_t num_sms;
  uint32_t smem_bytes;      // max per-CTA SMEM/LDS (227 KB SM100/SM90, 64 KB CU)
  uint32_t smem_reserve;    // epilogue / scheduler / ring reserve
  uint32_t l2_bytes;        // Table I l2_cache_mb
  uint32_t p_tc_tflops;     // Table I GPU-wide FP8xFP4 peak
  uint32_t b_hbm_gbps;
  uint32_t b_comm_dgbps;    // link cap in deci-GB/s (287.2 -> 2872)
  uint32_t acc_tile_limit;  // zero-spill BM*BN accumulator budget
  uint32_t tma_m_cap;       // per-arch TMA/WGMMA/MFMA load-height atom
};

WEAVE_HOST_DEVICE constexpr PnkHardware get_pnk_hardware(HardwareArch arch) {
  if (arch == HardwareArch::B300_SM100) {
    return PnkHardware{148u, 232448u, 16384u, 128u * 1024u * 1024u,
                       4200u, 7680u, 7800u, 32768u, 128u};
  }
  if (arch == HardwareArch::H20_SM90) {
    return PnkHardware{78u, 232448u, 16384u, 60u * 1024u * 1024u,
                       255u, 3600u, 3700u, 32768u, 64u};
  }
  return PnkHardware{80u, 65536u, 16384u, 256u * 1024u * 1024u,
                     360u, 3684u, 2872u, 65536u, 64u};
}

// Precision byte widths in 1/32-byte units (activation, weight): the exact
// Table-I / precision_byte_width values (0.5625 = 18/32, 1.03125 = 33/32, ...).
WEAVE_HOST_DEVICE constexpr uint32_t pnk_act_bytes32(PrecisionMode precision) {
  return precision == PrecisionMode::FP8xFP8 ? 33u :
         precision == PrecisionMode::BF16 ? 65u : 18u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_weight_bytes32(PrecisionMode precision) {
  return precision == PrecisionMode::FP8xFP4 ? 16u :
         precision == PrecisionMode::FP8xFP8 ? 33u :
         precision == PrecisionMode::BF16 ? 64u : 18u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_tflops(HardwareArch arch,
                                                PrecisionMode precision) {
  const uint32_t base = get_pnk_hardware(arch).p_tc_tflops;
  return precision == PrecisionMode::NVFP4 ? base * 2u :
         precision == PrecisionMode::BF16 ? base / 2u : base;
}

// Candidate / ladder accessors are constexpr FUNCTIONS (not namespace-scope
// arrays): nvcc rejects ODR-used namespace constexpr arrays in device code.
//
// The tile ladders are the LEGAL kernel configurations of each backend,
// transcribed from the kernels' own static asserts -- an envelope that the
// backend cannot instantiate is not a recommendation:
//   SM103  sm100_fp8_fp4_mega_moe.cuh:1148-1150
//          BLOCK_M % 16 == 0, BLOCK_N in {64,128,256}, BLOCK_K in {128,256}
//   H20    sm90_fp8_mega_moe.cuh:623-634
//          BLOCK_M % 64 == 0 (16/32 only under kFP8SwapAB),
//          BLOCK_N in {64,128,256,512}, BLOCK_K in {128,256}, BLOCK_K % 128 == 0
//   MI308X megamoe_amd.hip:54-60 + impls/amd/include/weave_ck_bm_configs.hpp
//          BLOCK_M in {16,32,64,128}, BLOCK_N in {64,128,256},
//          BLOCK_K in {64,128,256}
// Ladders are walked DESCENDING (largest surviving tile wins), so the BK/BN
// preference penalties below act on a legal candidate set only.
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bn_count(HardwareArch arch) {
  return arch == HardwareArch::H20_SM90 ? 4u : 3u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bn_candidate(HardwareArch arch,
                                                      uint32_t i) {
  if (arch == HardwareArch::H20_SM90) {
    return i == 0u ? 512u : (i == 1u ? 256u : (i == 2u ? 128u : 64u));
  }
  return i == 0u ? 256u : (i == 1u ? 128u : 64u);
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bk_count(HardwareArch arch) {
  return arch == HardwareArch::MI308X_GFX942 ? 3u : 2u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bk_candidate(HardwareArch arch,
                                                      uint32_t i) {
  if (arch == HardwareArch::MI308X_GFX942) {
    return i == 0u ? 256u : (i == 1u ? 128u : 64u);
  }
  return i == 0u ? 256u : 128u;
}
// Smallest legal BK: the short-K preference target (a K <= 1024 GEMM has too
// few iterations to amortize a wide K tile's pipeline fill).
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bk_min(HardwareArch arch) {
  return pnk_bk_candidate(arch, pnk_bk_count(arch) - 1u);
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bm_count(HardwareArch arch) {
  return arch == HardwareArch::B300_SM100 ? 8u : 4u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bm_candidate(HardwareArch arch,
                                                      uint32_t i) {
  if (arch == HardwareArch::B300_SM100) {
    return i == 0u ? 256u : i == 1u ? 240u : i == 2u ? 192u : i == 3u ? 128u :
           i == 4u ? 96u : i == 5u ? 80u : i == 6u ? 64u : 32u;
  }
  if (arch == HardwareArch::H20_SM90) {
    return i == 0u ? 256u : i == 1u ? 192u : i == 2u ? 128u : 64u;
  }
  return i == 0u ? 128u : i == 1u ? 64u : i == 2u ? 32u : 16u;
}
// MMA M atom: WGMMA::M / UMMA tile height 64 on both NVIDIA paths, MFMA 32 on
// CDNA3.  A BLOCK_M that is not a whole atom burns padded MMA rows.
WEAVE_HOST_DEVICE constexpr uint32_t pnk_mma_m_atom(HardwareArch arch) {
  return arch == HardwareArch::MI308X_GFX942 ? 32u : 64u;
}
// K-length stage preference: a short K loop cannot amortize a deep prefetch
// pipeline, a long K loop hides latency with fewer stages (Tier-1 stage
// ladder).  The SMEM budget and the K-iteration count still cap it.
WEAVE_HOST_DEVICE constexpr uint32_t pnk_stage_ladder(uint32_t k) {
  return k <= 512u ? 6u : k <= 1024u ? 5u : k <= 2048u ? 4u :
         k <= 4096u ? 3u : 2u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_design_m(uint32_t i) {
  return i == 0u ? 32u : 192u;
}
// Prefill anchor of the STRUCTURAL verdicts (L1 swizzle / wave granularity /
// M-row padding).  240 is the top of the runtime M ladder, so the envelope's
// max_BM is actually exercised by the same ladder the per-M tuples use.
static constexpr uint32_t kPnkPrefillM = 240u;
// Structural penalty weights (milli-units, inherited from an earlier tuned
// design arm).
static constexpr uint32_t kPnkWaveWeight = 2u;
static constexpr uint32_t kPnkMmaPadWeight = 2u;
static constexpr uint32_t kPnkRowPadWeight = 1u;
static constexpr uint32_t kPnkBkBias = 96u;
static constexpr uint32_t kPnkNarrowN = 128u;
static constexpr uint32_t kPnkDepthBonus = 24u;
WEAVE_HOST_DEVICE constexpr uint32_t m_inner_ladder(uint32_t i) {
  return i == 0u ? 8u : i == 1u ? 16u : i == 2u ? 32u : i == 3u ? 48u :
         i == 4u ? 64u : i == 5u ? 96u : i == 6u ? 128u :
         i == 7u ? 192u : 240u;
}

// Smallest M-ladder member >= m_tokens (capped at 240): the per-M inner-tuple
// bucket shared by the offline LUT and the online per-expert publication.
WEAVE_HOST_DEVICE constexpr uint32_t m_inner_bucket(uint32_t m_tokens) {
  for (uint32_t b = 0u; b < 9u; ++b) {
    if (m_tokens <= m_inner_ladder(b)) return m_inner_ladder(b);
  }
  return m_inner_ladder(8u);
}
WEAVE_HOST_DEVICE constexpr uint32_t m_inner_bucket_index(uint32_t m_tokens) {
  for (uint32_t b = 0u; b < 9u; ++b) {
    if (m_tokens <= m_inner_ladder(b)) return b;
  }
  return 8u;
}

// Launch envelope of one (Hardware, Precision, Phase, N, K) GEMM.  Every
// per-M inner tuple of the phase shares (bn, bk, max_bm, num_stages).
struct PhaseEnvelope {
  uint16_t bn;
  uint16_t bk;
  uint16_t max_bm;
  uint8_t num_stages;
  uint8_t l2_swizzle_group;   // 1..8
  uint8_t interleave_mode;    // 0 tight, 1 expert-major 2D swizzle, 2 grouped
  uint8_t wg_layout_mode;     // phase-level warpgroup layout at max_BM
  uint32_t score_ns;          // integer roofline estimate at the design anchors
  uint32_t epw;               // experts per wave (derived)
  uint32_t penalty;           // structural objective that SELECTED this envelope
};

// Compact 26-bit envelope code, LOSSLESS for every structural field (only the
// score_ns / penalty diagnostics are dropped).  BN/BK are powers of two
// (log2(v/64) fits 2 bits, BN up to 512 on the SM90 path); BLOCK_M also takes
// the non-power-of-two envelope heights the runtime ladder supports, so it
// uses a 4-bit index into the shared ascending height table.  epw is the
// scheduler-consumed experts-per-wave verdict and is part of the envelope.
WEAVE_HOST_DEVICE constexpr uint32_t pnk_tile_index(uint32_t tile) {
  return tile == 64u ? 0u : tile == 128u ? 1u : tile == 256u ? 2u : 3u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bm_index(uint32_t bm) {
  return bm == 16u ? 0u : bm == 32u ? 1u : bm == 64u ? 2u : bm == 80u ? 3u :
         bm == 96u ? 4u : bm == 128u ? 5u : bm == 192u ? 6u :
         bm == 240u ? 7u : 8u;
}
WEAVE_HOST_DEVICE constexpr uint32_t pnk_bm_from_index(uint32_t i) {
  return i == 0u ? 16u : i == 1u ? 32u : i == 2u ? 64u : i == 3u ? 80u :
         i == 4u ? 96u : i == 5u ? 128u : i == 6u ? 192u :
         i == 7u ? 240u : 256u;
}
WEAVE_HOST_DEVICE constexpr uint32_t encode_phase_envelope(
    const PhaseEnvelope& e) {
  return (pnk_tile_index(e.bn) & 0x3u) |
         ((pnk_tile_index(e.bk) & 0x3u) << 2) |
         ((pnk_bm_index(e.max_bm) & 0xfu) << 4) |
         (((e.num_stages - 2u) & 0x7u) << 8) |
         (((e.l2_swizzle_group - 1u) & 0x7u) << 11) |
         ((static_cast<uint32_t>(e.interleave_mode) & 0x3u) << 14) |
         ((static_cast<uint32_t>(e.wg_layout_mode) & 0x3u) << 16) |
         ((e.epw & 0xffu) << 18);
}
WEAVE_HOST_DEVICE constexpr PhaseEnvelope decode_phase_envelope(uint32_t code) {
  PhaseEnvelope e{};
  e.bn = static_cast<uint16_t>(64u << (code & 0x3u));
  e.bk = static_cast<uint16_t>(64u << ((code >> 2) & 0x3u));
  e.max_bm = static_cast<uint16_t>(pnk_bm_from_index((code >> 4) & 0xfu));
  e.num_stages = static_cast<uint8_t>(((code >> 8) & 0x7u) + 2u);
  e.l2_swizzle_group = static_cast<uint8_t>(((code >> 11) & 0x7u) + 1u);
  e.interleave_mode = static_cast<uint8_t>((code >> 14) & 0x3u);
  e.wg_layout_mode = static_cast<uint8_t>((code >> 16) & 0x3u);
  e.epw = (code >> 18) & 0xffu;
  return e;
}

// Phase-level warpgroup layout verdict at the envelope's max_BM.  The per-M
// inner tuples refine it per bucket; this is the phase default the envelope
// carries so GEMM1 and GEMM2 can differ even at equal tile shapes.
WEAVE_HOST_DEVICE constexpr uint32_t pnk_wg_layout(HardwareArch arch,
                                                   uint32_t bm) {
  if (arch == HardwareArch::H20_SM90) return bm < 64u ? 1u : 0u;
  if (arch == HardwareArch::B300_SM100) return bm <= 64u ? 2u : 0u;
  return bm < 32u ? 1u : 0u;
}

// Per-M inner tuple (mainloop + epilogue), POD, trivially copyable, <= 16 B.
// pipe_mode is the raw per-arch value: SM103 TMEM permille (100/175/200),
// H20 WGMMA pipe depth (1..3), MI308X MFMA pipe (1..2).  The compact 22-bit
// code stores a 2-bit pipe code instead (see encode/decode below).
struct GemmInnerProfile {
  uint16_t block_m;
  uint16_t tma_load_m;
  uint16_t store_block_m;
  uint8_t wg_layout_mode;     // 0 split-M, 1 split-N, 2 SM103 sub-tile
  uint8_t pipe_mode;
  uint8_t l2_swizzle_group;   // 1..8
  uint8_t epi_reg_stages;     // 1 tiny M, 2 fat M (hides TMEM/WGMMA latency)
  uint8_t scatter_path;       // GEMM2: 0 direct reg, 1 SMEM handoff,
                              // 2 lock-free local pre-reduce; GEMM1: 0
  uint8_t reserved;
};
static_assert(sizeof(GemmInnerProfile) <= 16u,
              "GemmInnerProfile must stay a compact <= 16-byte POD");

WEAVE_HOST_DEVICE constexpr uint32_t inner_pipe_code(
    HardwareArch arch, uint32_t pipe_mode) {
  if (arch == HardwareArch::B300_SM100) {
    return pipe_mode == 200u ? 2u : (pipe_mode == 175u ? 1u : 0u);
  }
  return pipe_mode - 1u;
}
WEAVE_HOST_DEVICE constexpr uint32_t inner_pipe_mode(
    HardwareArch arch, uint32_t pipe_code) {
  if (arch == HardwareArch::B300_SM100) {
    return pipe_code == 2u ? 200u : (pipe_code == 1u ? 175u : 100u);
  }
  return pipe_code + 1u;
}

WEAVE_HOST_DEVICE constexpr uint32_t encode_inner_profile(
    HardwareArch arch, const GemmInnerProfile& p) {
  return ((p.block_m / 16u) & 0x1fu) |
         (((p.tma_load_m / 16u) & 0xfu) << 5) |
         (((p.store_block_m / 16u) & 0x7u) << 9) |
         ((static_cast<uint32_t>(p.wg_layout_mode) & 0x3u) << 12) |
         ((inner_pipe_code(arch, p.pipe_mode) & 0x3u) << 14) |
         (((p.l2_swizzle_group - 1u) & 0x7u) << 16) |
         (((p.epi_reg_stages - 1u) & 0x1u) << 19) |
         ((static_cast<uint32_t>(p.scatter_path) & 0x3u) << 20);
}
WEAVE_HOST_DEVICE constexpr GemmInnerProfile decode_inner_profile(
    HardwareArch arch, uint32_t code) {
  GemmInnerProfile p{};
  p.block_m = static_cast<uint16_t>((code & 0x1fu) * 16u);
  p.tma_load_m = static_cast<uint16_t>(((code >> 5) & 0xfu) * 16u);
  p.store_block_m = static_cast<uint16_t>(((code >> 9) & 0x7u) * 16u);
  p.wg_layout_mode = static_cast<uint8_t>((code >> 12) & 0x3u);
  p.pipe_mode = static_cast<uint8_t>(inner_pipe_mode(arch, (code >> 14) & 0x3u));
  p.l2_swizzle_group = static_cast<uint8_t>(((code >> 16) & 0x7u) + 1u);
  p.epi_reg_stages = static_cast<uint8_t>(((code >> 19) & 0x1u) + 1u);
  p.scatter_path = static_cast<uint8_t>((code >> 20) & 0x3u);
  return p;
}

WEAVE_HOST_DEVICE constexpr uint32_t pnk_stage_bytes(
    uint32_t max_bm, uint32_t bn, uint32_t bk,
    uint32_t act32, uint32_t w32) {
  return (max_bm * bk * act32 + bn * bk * w32) / 32u;
}

// Integer time estimate (ns) of one envelope at one design anchor M.  Term
// semantics are documented in calibrate.py _pnk_anchor_ns (wave-quantized
// GPU-wide TC time, L1-protected HBM streaming with the >= 4-stage saturation
// rule, pipeline fill, 2 ns/K-iter TMA issue overhead, epilogue, scatter).
WEAVE_HOST_DEVICE constexpr uint64_t pnk_anchor_ns(
    HardwareArch arch, uint32_t tflops, uint32_t phase, uint32_t n, uint32_t k,
    uint32_t bn, uint32_t bk, uint32_t max_bm, uint32_t stages, uint32_t m,
    uint32_t act32, uint32_t w32) {
  const PnkHardware hw = get_pnk_hardware(arch);
  const uint32_t bm_eff = select_online_expert_bm(m, max_bm, 0u);
  const uint32_t tiles_m = ceil_div_u32(m, bm_eff);
  const uint32_t tiles_n = n / bn;
  const uint32_t ctas = tiles_m * tiles_n;
  const uint32_t waves = ceil_div_u32(ctas, hw.num_sms);
  const uint64_t m_pad = static_cast<uint64_t>(tiles_m) * bm_eff;
  const uint64_t flops = 2ull * m_pad * n * k;
  const uint64_t compute_ns =
      (flops / (static_cast<uint64_t>(tflops) * 1000ull)) * waves;
  const uint64_t weight_bytes =
      static_cast<uint64_t>(n) * k * w32 / 32ull;
  const bool prot = tiles_m >= 2u && weight_bytes * 2ull <= hw.l2_bytes;
  const uint64_t hbm_bytes =
      weight_bytes * (prot ? 1ull : static_cast<uint64_t>(tiles_m)) +
      m_pad * k * act32 / 32ull;
  uint64_t hbm_ns = hbm_bytes / hw.b_hbm_gbps;
  if (stages < 4u) hbm_ns = hbm_ns * 4ull / stages;
  const uint64_t fill_ns =
      static_cast<uint64_t>(stages) *
      pnk_stage_bytes(max_bm, bn, bk, act32, w32) / hw.b_hbm_gbps;
  const uint64_t iter_ns = static_cast<uint64_t>(k / bk) * 2ull * waves;
  uint64_t epi_bytes = 0ull;
  uint64_t scat_ns = 0ull;
  if (phase == 1u) {
    epi_bytes = m_pad * (static_cast<uint64_t>(n) * act32 / 32ull) +
                m_pad * (static_cast<uint64_t>(n / 2u) * act32 / 32ull);
  } else {
    epi_bytes = m_pad * (static_cast<uint64_t>(k) * act32 / 32ull) +
                m_pad * n * 2ull;
    scat_ns = m_pad * n * 2ull * 10ull / hw.b_comm_dgbps;
  }
  const uint64_t epi_ns = epi_bytes / hw.b_hbm_gbps;
  const uint64_t t = (compute_ns > hbm_ns ? compute_ns : hbm_ns) +
                     fill_ns + iter_ns + epi_ns + scat_ns;
  return t;
}

// Analytical legal pruning + integer STRUCTURAL scoring.
//
// The envelope is a structural choice (tile shape / pipeline depth / layout),
// so the objective is the summed structural WASTE of the candidate, not a
// roofline time: a roofline sum is dominated by candidate-independent terms
// (the M ladder clamps BLOCK_M at the design anchors) and collapses distinct
// geometries onto one constant envelope.  The integer roofline estimate
// pnk_anchor_ns is still evaluated and reported as score_ns for auditing.
//
// Terms (all milli-units, integer):
//   col_pen  SM wave quantization of the persistent scheduler's work unit.
//            mega_moe.cuh:1805 issues N-CLUSTERS of 2 BLOCK_N tiles and the
//            N % (2*BN) prune guarantees tiles_n is even, so the wave width is
//            num_sms/2 resident 2-CTA clusters: idle-cluster fraction.
//   mma_pad  BLOCK_M rows burned by the MMA M atom (WGMMA/UMMA 64, MFMA 32).
//   pad_pen  M-row padding of the prefill anchor under the runtime BM ladder.
//   bk_bias  K-length BK preference: short K wants the narrowest legal BK,
//            long K wants BK=128 (whole BK128 TMA tiles, SM90 assert).
//   narrow   narrow-N tile explosion on wide GEMMs (scheduler/epilogue cost).
//   depth    -24 per pipeline stage (latency hiding), clamped at 0.
// Candidate order BN, BK descending, stages descending, BLOCK_M = largest
// surviving legal height; strict-< tie rule (first candidate wins ties) --
// identical to calibrate.py select_phase_envelope.
WEAVE_HOST_DEVICE constexpr PhaseEnvelope select_phase_envelope(
    HardwareArch arch, PrecisionMode precision, uint32_t phase,
    uint32_t n, uint32_t k) {
  const PnkHardware hw = get_pnk_hardware(arch);
  const uint32_t act32 = pnk_act_bytes32(precision);
  const uint32_t w32 = pnk_weight_bytes32(precision);
  const uint32_t tflops = pnk_tflops(arch, precision);
  const uint32_t cluster_slots = hw.num_sms / 2u;
  const uint32_t mma_atom = pnk_mma_m_atom(arch);
  const uint32_t bk_min = pnk_bk_min(arch);
  const uint64_t weight_bytes = static_cast<uint64_t>(n) * k * w32 / 32ull;
  const uint32_t stage_top = pnk_stage_ladder(k) < (k <= 1024u ? 4u : 8u)
      ? pnk_stage_ladder(k) : (k <= 1024u ? 4u : 8u);
  PhaseEnvelope best{};
  uint32_t best_penalty = 0xffffffffu;
  bool have = false;
  for (uint32_t bi = 0u; bi < pnk_bn_count(arch); ++bi) {
    const uint32_t bn = pnk_bn_candidate(arch, bi);
    // Zero N-tail waste + 2-CTA cluster pairing (GEMM1: SwiGLU halves).
    if (n % bn != 0u || n % (2u * bn) != 0u) continue;
    const uint32_t tiles_n = n / bn;
    const uint32_t clusters_n = tiles_n / 2u;
    const uint32_t cw = ceil_div_u32(clusters_n, cluster_slots);
    const uint32_t col_pen =
        (cw * cluster_slots - clusters_n) * 1024u / clusters_n;
    const uint32_t narrow = (n >= 4096u && bn < 128u) ? kPnkNarrowN : 0u;
    for (uint32_t ki = 0u; ki < pnk_bk_count(arch); ++ki) {
      const uint32_t bk = pnk_bk_candidate(arch, ki);
      if (k % bk != 0u) continue;
      const uint32_t bk_bias =
          ((k <= 1024u && bk != bk_min) || (k >= 4096u && bk != 128u))
              ? kPnkBkBias : 0u;
      const uint32_t pipe_cap = k / bk > 2u ? k / bk : 2u;
      const uint32_t st_cap = stage_top < pipe_cap ? stage_top : pipe_cap;
      for (uint32_t stages = st_cap; stages >= 2u; --stages) {
        // Largest surviving legal BLOCK_M (accumulator + SMEM stage budget).
        uint32_t max_bm = 0u;
        for (uint32_t mi = 0u; mi < pnk_bm_count(arch); ++mi) {
          const uint32_t cand = pnk_bm_candidate(arch, mi);
          if (cand * bn > hw.acc_tile_limit) continue;
          const uint32_t stage_bytes =
              pnk_stage_bytes(cand, bn, bk, act32, w32);
          if (stages * stage_bytes > hw.smem_bytes - hw.smem_reserve) continue;
          max_bm = cand;
          break;
        }
        if (max_bm == 0u) continue;
        const uint32_t bm_d =
            select_online_expert_bm(kPnkPrefillM, max_bm, 0u);
        const uint32_t tm_d = ceil_div_u32(kPnkPrefillM, bm_d);
        const uint32_t m_pad = tm_d * bm_d;
        const uint32_t mma_pad =
            (align_up_u32(max_bm, mma_atom) - max_bm) * 1024u / max_bm;
        const uint32_t pad_pen = (m_pad - kPnkPrefillM) * 1024u / m_pad;
        uint32_t penalty = kPnkWaveWeight * col_pen +
                           kPnkMmaPadWeight * mma_pad +
                           kPnkRowPadWeight * pad_pen + bk_bias + narrow;
        const uint32_t bonus = stages * kPnkDepthBonus;
        penalty = penalty > bonus ? penalty - bonus : 0u;
        if (!have || penalty < best_penalty) {
          // L1 intra-expert 2D L2-swizzle protection / wave-granularity
          // verdict at the prefill anchor, using the SAME BM ladder the
          // runtime inner tuples use.
          const uint32_t ctas = tm_d * tiles_n;
          uint32_t epw = hw.num_sms / ctas;
          if (epw == 0u) epw = 1u;
          uint8_t interleave = 0u;
          uint32_t group = 1u;
          if (tm_d >= 2u && weight_bytes * 2ull <= hw.l2_bytes) {
            interleave = 1u;
            group = tm_d < 8u ? tm_d : 8u;
            epw = 1u;
          } else if (ctas <= hw.num_sms) {
            interleave = 0u;
            group = 1u;
          } else {
            interleave = 2u;
            group = epw < 8u ? epw : 8u;
          }
          uint64_t roof = 0ull;
          for (uint32_t ai = 0u; ai < 2u; ++ai) {
            roof += pnk_anchor_ns(arch, tflops, phase, n, k, bn, bk, max_bm,
                                  stages, pnk_design_m(ai), act32, w32);
          }
          best.bn = static_cast<uint16_t>(bn);
          best.bk = static_cast<uint16_t>(bk);
          best.max_bm = static_cast<uint16_t>(max_bm);
          best.num_stages = static_cast<uint8_t>(stages);
          best.l2_swizzle_group = static_cast<uint8_t>(group);
          best.interleave_mode = interleave;
          best.wg_layout_mode =
              static_cast<uint8_t>(pnk_wg_layout(arch, max_bm));
          best.score_ns = static_cast<uint32_t>(roof);
          best.epw = epw;
          best.penalty = penalty;
          best_penalty = penalty;
          have = true;
        }
      }
    }
  }
  return best;  // bn == 0 marks "no legal envelope" (invalid N/K geometry).
}

// Per-M inner tuple constrained to the phase envelope.  mu_dedup is the LIVE
// DEDUP VERDICT (0/1) reduced from mu vs mu_dedup_thresh before the call, so
// the selection stays integer and bit-exact with calibrate.py.
WEAVE_HOST_DEVICE constexpr GemmInnerProfile select_inner_from_envelope(
    HardwareArch arch, PrecisionMode precision, uint32_t phase,
    uint32_t n, uint32_t k, const PhaseEnvelope& env, uint32_t m_tokens,
    uint32_t mu_dedup, uint32_t is_tight_reg_arch) {
  const PnkHardware hw = get_pnk_hardware(arch);
  const uint32_t w32 = pnk_weight_bytes32(precision);
  const uint32_t bm = select_online_expert_bm(m_tokens, env.max_bm, 0u);
  const uint32_t tma = bm < hw.tma_m_cap ? bm : hw.tma_m_cap;
  uint32_t wg = 0u;
  if (arch == HardwareArch::H20_SM90) {
    wg = bm < 64u ? 1u : 0u;   // split-N keeps 4 WGs active at small M
  } else if (arch == HardwareArch::B300_SM100) {
    wg = bm <= 64u ? 2u : 0u;  // SM103 sub-tile mode
  } else {
    wg = bm < 32u ? 1u : 0u;
  }
  uint32_t pipe = 1u;
  if (arch == HardwareArch::B300_SM100) {
    pipe = (bm >= 128u && env.num_stages >= 4u) ? 200u :
           (bm >= 64u ? 175u : 100u);
  } else if (arch == HardwareArch::H20_SM90) {
    pipe = env.num_stages >= 5u ? 3u : (env.num_stages >= 3u ? 2u : 1u);
  } else {
    pipe = env.num_stages >= 4u ? 2u : 1u;
  }
  const uint32_t m_blocks = ceil_div_u32(m_tokens, bm);
  const uint64_t weight_bytes =
      static_cast<uint64_t>(n) * k * w32 / 32ull;
  const uint32_t group =
      (m_blocks >= 2u && weight_bytes * 2ull <= hw.l2_bytes)
          ? (m_blocks < 8u ? m_blocks : 8u) : 1u;
  uint32_t store = 16u;
  if (bm % 64u == 0u) store = 64u;
  else if (bm % 32u == 0u) store = 32u;
  const uint32_t epi =
      (bm >= 128u ||
       (arch == HardwareArch::B300_SM100 && (bm == 80u || bm == 96u) &&
        pipe >= 175u)) ? 2u : 1u;
  uint32_t scat = 0u;
  if (phase == 2u) {
    scat = mu_dedup != 0u ? 2u : (is_tight_reg_arch != 0u ? 1u : 0u);
  }
  GemmInnerProfile p{};
  p.block_m = static_cast<uint16_t>(bm);
  p.tma_load_m = static_cast<uint16_t>(tma);
  p.store_block_m = static_cast<uint16_t>(store);
  p.wg_layout_mode = static_cast<uint8_t>(wg);
  p.pipe_mode = static_cast<uint8_t>(pipe);
  p.l2_swizzle_group = static_cast<uint8_t>(group);
  p.epi_reg_stages = static_cast<uint8_t>(epi);
  p.scatter_path = static_cast<uint8_t>(scat);
  return p;
}

// Public Tier-1 lookup (bit-exact with calibrate.py select_phase_inner_profile).
WEAVE_HOST_DEVICE constexpr GemmInnerProfile select_phase_inner_profile(
    HardwareArch arch, PrecisionMode precision, uint32_t phase,
    uint32_t n, uint32_t k, uint32_t m_tokens, uint32_t mu_dedup,
    uint32_t is_tight_reg_arch) {
  const PhaseEnvelope env = select_phase_envelope(arch, precision, phase, n, k);
  return select_inner_from_envelope(arch, precision, phase, n, k, env,
                                    m_tokens, mu_dedup, is_tight_reg_arch);
}

// Zero-fill an OnlineRoutingPolicy word-wise (the struct is all 4-byte
// fields): avoids materializing a 2 KB value temporary on the device stack.
WEAVE_HOST_DEVICE void zero_online_policy(OnlineRoutingPolicy* p) {
  uint32_t* w = reinterpret_cast<uint32_t*>(p);
  for (uint32_t i = 0u; i < sizeof(OnlineRoutingPolicy) / 4u; ++i) w[i] = 0u;
}

// Tier-3 online policy from the winning Eq.(1)-(3) breakdown.  The gate
// arithmetic is O(1) (no second histogram traversal for the equation terms);
// every decision boundary is an integer comparison or a single binary32
// compare so the Python mirror is bit-exact.  Explicit knob pins always
// override; the routing-adaptive rules only run when WEAVE_ADAPT_BM=1.
// Writes through p_out (caller storage -- typically the GLOBAL control block)
// so the extended per-expert LUT never lives on the device stack.
WEAVE_HOST_DEVICE void cta0_online_policy(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count,
    const EquationBreakdown& br, uint32_t bm_envelope,
    OnlineRoutingPolicy* p_out, uint32_t unique_tokens = 0u) {
  (void)scatter_count;
  if (p_out == nullptr) return;
  const HardwareCalibration cal = get_hardware_calibration(arch);
  OnlineRoutingPolicy& p = *p_out;
  zero_online_policy(p_out);
  uint32_t total = 0u, active = 0u, max_n = 0u;
  histogram_stats(m_e, wl.local_experts, &total, &active, &max_n);
  p.active_experts = active;
  p.max_n = max_n;
  p.total_routes = total;
  p.bm_envelope = bm_envelope;
  p.mean_n = active > 0u
      ? static_cast<float>(total) / static_cast<float>(active) : 0.0f;
  // Rank-hit multiplicity mu = M / unique_tokens.  When the backend cannot
  // supply the unique-token count, the dedup pull count is the live proxy;
  // with neither, mu = 1 (conservative: no dedup recommendation).
  if (unique_tokens > 0u) {
    p.rank_hit_multiplicity_mu =
        static_cast<float>(total) / static_cast<float>(unique_tokens);
  } else if (dedup_pulls > 0u && dedup_pulls < total) {
    p.rank_hit_multiplicity_mu =
        static_cast<float>(total) / static_cast<float>(dedup_pulls);
  } else {
    p.rank_hit_multiplicity_mu = 1.0f;
  }
  // Eq.(2) Middle HBM utilization and Eq.(3) tail Combine fraction of the
  // selected policy (binary32, one rounding per operation).
  const float denom = (br.t_comp_us * 1000.0f) * hw.b_hbm_gbps;
  p.middle_hbm_util = br.t_comp_us > 0.0f ? br.v_mem_bytes / denom : 0.0f;
  p.tail_combine_frac = br.t_total_us > 0.0f
      ? br.t_tail_reduce_us / br.t_total_us : 0.0f;

  const bool adaptive = knobs.adapt_bm != 0u;
  p.adaptive_active = adaptive ? 1u : 0u;

  // (KEY) Dedicated Middle Reduce lanes: keep the compiled R_max only when
  // Middle leaves HBM headroom AND the tail is large enough to amortize them.
  // An explicit pin (WEAVE_ADAPT_BM=0) stays deterministic and bit-identical.
  const uint32_t rw = resolve_reduce_warps(knobs);
  p.reduce_gate_hbm_ok = p.middle_hbm_util <= cal.hbm_headroom_thresh ? 1u : 0u;
  p.reduce_gate_tail_ok = p.tail_combine_frac >= cal.tail_ratio_thresh ? 1u : 0u;
  p.active_reduce_warps = (!adaptive || rw == 0u) ? rw
      : ((p.reduce_gate_hbm_ok != 0u && p.reduce_gate_tail_ok != 0u) ? rw : 0u);

  // Auto V_HEAD: under hot-expert skew (n_max >= 2 x mean_n, integer
  // cross-multiply) shrink the head budget to the calibrated hot budget so
  // helping warps return to GEMM while the comm warps drain the hot expert.
  // Balanced routing keeps the existing Eq. window bit-identical.
  uint32_t vh = effective_v_head(knobs, br.v_head);
  if (adaptive && !has_override(knobs, kOverrideVHead) && active > 0u) {
    const bool hot = static_cast<uint64_t>(max_n) * static_cast<uint64_t>(active)
        >= 2ull * static_cast<uint64_t>(total);
    if (hot) {
      uint32_t capped = vh < cal.v_head_hot_budget ? vh : cal.v_head_hot_budget;
      if (capped == 0u) capped = 1u;
      vh = align_up_u32(capped, 32u);
    }
  }
  p.active_v_head = vh;

  // Auto ready group: mean_n ladder {32,128,256} -> {1,2,4,8}; when >= 3/4 of
  // the local experts are active (dense prefill) double the group to amortize
  // publication.  Explicit WEAVE_READY_GROUP_SIZE always overrides.
  uint32_t g = knobs.ready_group_size;
  if (g == 0u) {
    if (active == 0u) {
      g = 1u;
    } else {
      const uint64_t t64 = static_cast<uint64_t>(total);
      const uint64_t a64 = static_cast<uint64_t>(active);
      g = 1u;
      if (t64 >= a64 * 32ull) g = 2u;
      if (t64 >= a64 * 128ull) g = 4u;
      if (t64 >= a64 * 256ull) g = 8u;
      if (a64 * 4ull >= static_cast<uint64_t>(wl.local_experts) * 3ull && g < 8u) {
        g *= 2u;
      }
    }
  }
  p.active_ready_group_size = g;

  // Dedup verdict: explicit WEAVE_DEDUP_MODE pins win; auto compares the live
  // rank-hit multiplicity against the calibrated per-hardware threshold.
  if (knobs.dedup_mode == 0u) {
    p.enable_dedup_runtime = 0u;
  } else if (knobs.dedup_mode == 1u) {
    p.enable_dedup_runtime = 1u;
  } else {
    p.enable_dedup_runtime =
        p.rank_hit_multiplicity_mu >= cal.mu_dedup_thresh ? 1u : 0u;
  }

  // Per-expert BM tier for the mean load (same ladder as the scheduler).
  p.bm_tier_mean = select_online_expert_bm(
      static_cast<uint32_t>(p.mean_n), bm_envelope, cal.bm_pad_slack);

  // 3-Tier (Phase,N,K) profile publication.  GEMM1 and GEMM2 envelopes are
  // selected independently from the launch geometry (N1=2I,K1=H / N2=H,K2=I);
  // the 9-bucket per-M LUT is computed ONCE per phase, then each live expert
  // is bucketed into it (one ladder walk per expert, no per-expert search).
  // Only the adaptive tier publishes codes; pinned builds keep zeros so the
  // compiled structure stays deterministic.  The scatter_path verdict uses
  // the live dedup decision (mu vs mu_dedup_thresh, or the explicit pin).
  if (adaptive) {
    const PrecisionMode prec = static_cast<PrecisionMode>(knobs.precision_mode);
    const uint32_t tight = arch == HardwareArch::H20_SM90 ? 1u : 0u;
    const uint32_t n1 = 2u * wl.intermediate, k1 = wl.hidden;
    const uint32_t n2 = wl.hidden, k2 = wl.intermediate;
    const PhaseEnvelope env1 = select_phase_envelope(arch, prec, 1u, n1, k1);
    const PhaseEnvelope env2 = select_phase_envelope(arch, prec, 2u, n2, k2);
    uint32_t lut1[9], lut2[9];
    for (uint32_t b = 0u; b < 9u; ++b) {
      lut1[b] = encode_inner_profile(arch, select_inner_from_envelope(
          arch, prec, 1u, n1, k1, env1, m_inner_ladder(b),
          p.enable_dedup_runtime, tight));
      lut2[b] = encode_inner_profile(arch, select_inner_from_envelope(
          arch, prec, 2u, n2, k2, env2, m_inner_ladder(b),
          p.enable_dedup_runtime, tight));
    }
    p.phase_profile_active = 1u;
    p.l1_envelope_code = encode_phase_envelope(env1);
    p.l2_envelope_code = encode_phase_envelope(env2);
    if (m_e != nullptr) {
      for (uint32_t e = 0u; e < wl.local_experts; ++e) {
        const uint32_t m = m_e[e];
        if (m == 0u) continue;
        const uint32_t b = m_inner_bucket_index(m);
        p.l1_profile_code[e] = lut1[b];
        p.l2_profile_code[e] = lut2[b];
      }
    }
  }
}

// Eq. (1)-(3) for one spatial candidate and one scatter path.
//
// reduce_warps carries WEAVE_REDUCE_WARPS.  The default 0 evaluates every
// expression exactly as published in Paper Eq. (1)-(3), bit-for-bit, so all
// golden predictions are unchanged.  A non-zero value carves that many
// dedicated streaming Reduce warps out of the CTA communication warp group:
// they overlap the fine-grained Combine reduction with T_mid on the Middle
// HBM bandwidth the GEMM/epilogue leaves spare, and a CommQueueAsync middle
// scatter is drained by max(1, N_c - reduce_warps) communication units.
// Values above kMaxReduceWarps clamp here too, not only in resolve_reduce_warps,
// so this entry point matches the Python mirror when called directly.
WEAVE_HOST_DEVICE EquationBreakdown evaluate_spatial_candidate(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const SpatialCandidate& candidate, uint32_t scatter_mode,
    const uint32_t* m_e, uint32_t dedup_pulls, uint32_t scatter_count,
    uint32_t reduce_warps = 0u) {
  reduce_warps = reduce_warps > kMaxReduceWarps ? kMaxReduceWarps : reduce_warps;
  EquationBreakdown out{};
  out.r_mask = candidate.r_mask;
  out.scatter_mode = scatter_mode;
  out.v_head = choose_v_head(hw, candidate, dedup_pulls);

  const uint32_t total_tokens = sum_histogram(m_e, wl.local_experts);
  out.total_tokens = total_tokens;

  // Eq. (1): F_gemm(r) and V_mem(r).
  const float h = static_cast<float>(wl.hidden);
  const float i = static_cast<float>(wl.intermediate);
  const float bm = static_cast<float>(candidate.block_m);
  float gemm_flops = 0.0f;
  float weight_bytes = 0.0f;
  if (m_e != nullptr) {
    for (uint32_t e = 0u; e < wl.local_experts; ++e) {
      const uint32_t k_e = ceil_div_u32(m_e[e], candidate.block_m);
      gemm_flops += static_cast<float>(k_e) * bm * 6.0f * h * i;
      weight_bytes += static_cast<float>(k_e) * 3.0f * h * i * hw.weight_bytes;
    }
  }
  const float activation_bytes =
      static_cast<float>(total_tokens) * hw.act_in_bytes * (h + i);
  out.f_gemm_flops = gemm_flops;
  out.v_mem_bytes = weight_bytes + activation_bytes;

  // Eq. (2): T_mma, T_epi, T_comp.  B300 overlaps TMEM drain with MMA;
  // H20 and MI308X serialize shared-register accumulator writeback.
  const float t_mem_us =
      out.v_mem_bytes / (hw.b_hbm_gbps * 1000.0f);  // bytes / (GB/s -> B/us)
  const float t_mma_us =
      out.f_gemm_flops / (candidate.p_tc_tflops * 1000000.0f);  // FLOP / (TFLOP/s -> FLOP/us)
  out.t_mma_us = fmaxf(t_mem_us, t_mma_us);

  out.v_epi_bytes = static_cast<float>(total_tokens) *
      (hw.act_in_bytes * i + hw.out_bytes * h);
  const float total_epi_units =
      static_cast<float>(hw.num_sms_or_cus) * static_cast<float>(candidate.comp_units);
  const float b_epi_gbps = fminf(
      hw.b_hbm_gbps,
      fmaxf(1.0f, total_epi_units * hw.b_epi_write_per_unit_gbps));
  out.t_epi_us = out.v_epi_bytes / (b_epi_gbps * 1000.0f);
  out.t_comp_us = arch == HardwareArch::B300_SM100
      ? fmaxf(out.t_mma_us, out.t_epi_us)
      : out.t_mma_us + out.t_epi_us;

  // Eq. (3): bandwidth-limited pull and scatter.
  // r=(N_c, N_e) is GPU-wide in the paper's Eq. (3).  Candidate fields are
  // per-SM/CU ratios (Table I), so expand them before applying the link cap.
  const uint32_t effective_comm_units_per_sm = candidate.comm_units == 0u
      ? candidate.comp_units : candidate.comm_units;
  // WEAVE_REDUCE_WARPS: the Reduce lanes are carved out of the CTA
  // communication group, so the middle CommQueueAsync scatter keeps only
  // max(1, N_c - reduce_warps) units (unsigned-safe subtraction).  The
  // EpilogueInline scatter is issued by the epilogue warps and keeps its full
  // bandwidth; the dispatch pull is owned by the dispatch warps in both paths.
  const uint32_t scatter_comm_units_per_sm =
      (reduce_warps > 0u &&
       scatter_mode != static_cast<uint32_t>(ScatterPath::EpilogueInline))
          ? (effective_comm_units_per_sm > reduce_warps
                 ? effective_comm_units_per_sm - reduce_warps
                 : 1u)
          : effective_comm_units_per_sm;
  const float gpu_comm_units =
      static_cast<float>(hw.num_sms_or_cus * effective_comm_units_per_sm);
  const float gpu_scat_comm_units =
      static_cast<float>(hw.num_sms_or_cus * scatter_comm_units_per_sm);
  const float gpu_epi_units =
      static_cast<float>(hw.num_sms_or_cus * candidate.comp_units);
  const float b_pull_gbps = fminf(
      hw.b_link_gbps,
      fmaxf(1.0f, gpu_comm_units * hw.b_pull_per_unit_gbps));
  const float b_scat_gbps = scatter_mode ==
          static_cast<uint32_t>(ScatterPath::EpilogueInline)
      ? fminf(hw.b_link_gbps,
              fmaxf(1.0f, gpu_epi_units * hw.b_scat_epi_per_unit_gbps))
      : fminf(hw.b_link_gbps,
              fmaxf(1.0f, gpu_scat_comm_units * hw.b_scat_comm_per_unit_gbps));

  out.t_pull_us = dedup_pulls == 0u ? 0.0f :
      static_cast<float>(dedup_pulls) * h * hw.act_in_bytes /
          (b_pull_gbps * 1000.0f);
  out.t_scat_us = scatter_count == 0u ? 0.0f :
      static_cast<float>(scatter_count) * h * hw.out_bytes /
          (b_scat_gbps * 1000.0f);

  if (scatter_mode == static_cast<uint32_t>(ScatterPath::EpilogueInline)) {
    out.t_mid_us = fmaxf(out.t_pull_us, out.t_comp_us + out.t_scat_us);
  } else {
    out.t_mid_us = fmaxf(out.t_pull_us + out.t_scat_us,
                         out.t_comp_us + hw.t_handoff_us);
  }

  // Fine-grained tail reduce is bandwidth-bound and overlaps with the queue.
  const float b_tail_gbps = fminf(
      hw.b_hbm_gbps,
      fmaxf(1.0f, static_cast<float>(hw.num_sms_or_cus) *
                      static_cast<float>(hw.max_units_per_sm) *
                      kReduceGbpsPerWarp));
  const float reduce_bytes =
      static_cast<float>(total_tokens) * h * hw.out_bytes * 2.0f;
  out.t_tail_reduce_us = reduce_bytes / (b_tail_gbps * 1000.0f);
  if (reduce_warps > 0u) {
    // Middle already streams V_mem through HBM inside T_comp, so only the
    // spare bandwidth is left for the Reduce lanes.  An empty Middle
    // (T_comp == 0) offers no overlap window and therefore no spare bandwidth.
    const float b_mid_used_gbps = out.t_comp_us > 0.0f
        ? out.v_mem_bytes / (out.t_comp_us * 1000.0f)
        : hw.b_hbm_gbps;
    const float b_mid_spare_gbps =
        fmaxf(0.0f, hw.b_hbm_gbps - b_mid_used_gbps);
    const float b_reduce_gbps = fminf(
        b_mid_spare_gbps,
        static_cast<float>(hw.num_sms_or_cus) *
            static_cast<float>(reduce_warps) * kReduceGbpsPerWarp);
    // The bytes the Reduce lanes drain inside T_mid leave the tail; the
    // remainder still pays the full tail bandwidth.
    const float overlapped_bytes = b_reduce_gbps * 1000.0f * out.t_mid_us;
    out.t_tail_reduce_us =
        fmaxf(0.0f, reduce_bytes - overlapped_bytes) / (b_tail_gbps * 1000.0f);
  }
  out.t_total_us = hw.t_head_us + out.t_mid_us +
      (WEAVE_TAIL_FINE_GRAINED ? out.t_tail_reduce_us : hw.t_tail_us);
  return out;
}

WEAVE_HOST_DEVICE PolicyDecision make_override_decision(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count,
    OnlineRoutingPolicy* online_out = nullptr, uint32_t unique_tokens = 0u) {
  const CandidateSet candidates = get_candidate_set(
      arch, static_cast<PrecisionMode>(knobs.precision_mode));
  const uint32_t requested_r = has_override(knobs, kOverrideSpatial)
      ? knobs.override_r_mask
      : default_r_mask_for_arch(arch);

  // Manual / ablation overrides may name a schedule outside Table I.  Match a
  // paper candidate when possible; otherwise synthesise the same candidate
  // descriptor from the packed r_mask so the equations stay well-defined.
  uint32_t selected = 0xffffffffu;
  for (uint32_t c = 0u; c < candidates.count; ++c) {
    if (candidates.items[c].r_mask == requested_r) {
      selected = c;
      break;
    }
  }
  SpatialCandidate candidate{};
  if (selected < candidates.count) {
    candidate = candidates.items[selected];
  } else {
    const uint32_t comm = r_comm_units(requested_r);
    const uint32_t comp = r_comp_units(requested_r);
    const uint32_t groups = r_groups(requested_r);
    const uint32_t block_m = knobs.block_m_fixed != 0u
        ? knobs.block_m_fixed : 128u;
    const float p_tc = comp >= 8u ? hw.p_tc_full_tflops
                                  : hw.p_tc_half_tflops;
    candidate = SpatialCandidate{
        requested_r, comm, comp, block_m,
        groups == 0u ? 4u : groups, p_tc, r_heterogeneous(requested_r)};
  }

  const uint32_t scatter_mode = effective_scatter_mode(
      knobs, candidate.comm_units == 0u ? 0u : 1u);
  const EquationBreakdown br = evaluate_spatial_candidate(
      arch, hw, wl, candidate, scatter_mode,
      m_e, dedup_pulls, scatter_count, resolve_reduce_warps(knobs));

  PolicyDecision d{};
  d.magic = kPolicyMagic;
  d.version = kPolicyVersion;
  d.arch = static_cast<uint32_t>(arch);
  d.valid = 1u;
  d.selected_candidate = selected;
  d.selected_r_mask = br.r_mask;
  d.selected_scatter_mode = scatter_mode;
  d.selected_v_head = effective_v_head(knobs, br.v_head);
  d.selected_comm_units = hw.num_sms_or_cus * r_comm_units(br.r_mask);
  d.selected_comp_units = hw.num_sms_or_cus * r_comp_units(br.r_mask);
  d.selected_block_m = candidate.block_m;
  d.selected_role_groups = r_groups(br.r_mask);
  d.predicted_mid_us = br.t_mid_us;
  d.predicted_total_us = br.t_total_us;
  d.predicted_comp_us = br.t_comp_us;
  d.predicted_pull_us = br.t_pull_us;
  d.predicted_scat_us = br.t_scat_us;
  d.input_total_tokens = br.total_tokens;
  d.input_dedup_pulls = dedup_pulls;
  d.input_scatter_rows = scatter_count;
  if (online_out != nullptr) {
    cta0_online_policy(
        arch, hw, wl, knobs, m_e, dedup_pulls, scatter_count,
        br, candidate.block_m, online_out, unique_tokens);
  }
  return d;
}

// CTA-0 entry point.  One thread calls this function; the histogram and all
// routing counters must be visible before invocation.  Explicit WeaveKnobs
// overrides bypass the search but still pass through the same equation path.
// When online_out is non-null the Tier-3 OnlineRoutingPolicy is computed from
// the winning breakdown (one extra histogram stats pass, O(1) gate arithmetic);
// the PolicyDecision itself is unchanged, bit-for-bit.
WEAVE_HOST_DEVICE PolicyDecision cta0_select_policy(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count,
    OnlineRoutingPolicy* online_out = nullptr, uint32_t unique_tokens = 0u) {
  if (!knobs.enable_runtime_cost_model ||
      has_override(knobs, kOverrideSpatial | kOverrideScatter |
                                kOverrideVHead | kOverrideTemporal)) {
    return make_override_decision(
        arch, hw, wl, knobs, m_e, dedup_pulls, scatter_count,
        online_out, unique_tokens);
  }

  const CandidateSet candidates = get_candidate_set(
      arch, static_cast<PrecisionMode>(knobs.precision_mode));
  PolicyDecision best{};
  best.magic = kPolicyMagic;
  best.version = kPolicyVersion;
  best.arch = static_cast<uint32_t>(arch);
  best.valid = 0u;
  best.predicted_total_us = 3.402823466e+38f;
  EquationBreakdown best_br{};

  const auto consider = [&](uint32_t c, uint32_t s,
                            const EquationBreakdown& br) {
    if (!best.valid || br.t_total_us < best.predicted_total_us) {
      best.valid = 1u;
      best_br = br;
      best.selected_candidate = c;
      best.selected_r_mask = br.r_mask;
      best.selected_scatter_mode = s;
      best.selected_v_head = effective_v_head(knobs, br.v_head);
      best.selected_comm_units = hw.num_sms_or_cus * candidates.items[c].comm_units;
      best.selected_comp_units = hw.num_sms_or_cus * candidates.items[c].comp_units;
      best.selected_block_m = candidates.items[c].block_m;
      best.selected_role_groups = candidates.items[c].role_groups;
      best.predicted_mid_us = br.t_mid_us;
      best.predicted_total_us = br.t_total_us;
      best.predicted_comp_us = br.t_comp_us;
      best.predicted_pull_us = br.t_pull_us;
      best.predicted_scat_us = br.t_scat_us;
      best.input_total_tokens = br.total_tokens;
      best.input_dedup_pulls = dedup_pulls;
      best.input_scatter_rows = scatter_count;
    }
  };

  // Resolved once: 0 keeps the paper path, including the in-place s=comm
  // derivation below, bit-identical to Eq. (1)-(3).
  const uint32_t reduce_warps = resolve_reduce_warps(knobs);

  for (uint32_t c = 0u; c < candidates.count; ++c) {
    // Eq.(1) and the GEMM/epilogue part of Eq.(2) are independent of s.
    // Evaluate them once, then derive the s=comm mid/total terms in place.
    // This halves CTA-0's histogram traversal on the critical path.
    const EquationBreakdown br_epi = evaluate_spatial_candidate(
        arch, hw, wl, candidates.items[c], 0u,
        m_e, dedup_pulls, scatter_count, reduce_warps);
    consider(c, 0u, br_epi);

    if (reduce_warps > 0u) {
      // With dedicated Reduce warps the s=comm branch changes both the scatter
      // bandwidth (carved-out units) and the tail overlap window, so evaluate
      // it in full instead of deriving it from the s=epi breakdown.
      const EquationBreakdown br_comm = evaluate_spatial_candidate(
          arch, hw, wl, candidates.items[c],
          static_cast<uint32_t>(ScatterPath::CommQueueAsync),
          m_e, dedup_pulls, scatter_count, reduce_warps);
      consider(c, 1u, br_comm);
      continue;
    }

    EquationBreakdown br_comm = br_epi;
    br_comm.scatter_mode = static_cast<uint32_t>(ScatterPath::CommQueueAsync);
    const uint32_t eff_comm_per_sm = candidates.items[c].comm_units == 0u
        ? candidates.items[c].comp_units : candidates.items[c].comm_units;
    const float gpu_comm_units = static_cast<float>(
        hw.num_sms_or_cus * eff_comm_per_sm);
    const float b_scat_comm = fminf(
        hw.b_link_gbps,
        fmaxf(1.0f, gpu_comm_units * hw.b_scat_comm_per_unit_gbps));
    br_comm.t_scat_us = scatter_count == 0u ? 0.0f :
        static_cast<float>(scatter_count) * static_cast<float>(wl.hidden) *
            hw.out_bytes / (b_scat_comm * 1000.0f);
    br_comm.t_mid_us = fmaxf(br_comm.t_pull_us + br_comm.t_scat_us,
                             br_comm.t_comp_us + hw.t_handoff_us);
    br_comm.t_total_us = hw.t_head_us + br_comm.t_mid_us +
        (WEAVE_TAIL_FINE_GRAINED ? br_comm.t_tail_reduce_us : hw.t_tail_us);
    consider(c, 1u, br_comm);
  }
  if (online_out != nullptr) {
    if (best.valid) {
      cta0_online_policy(arch, hw, wl, knobs, m_e, dedup_pulls,
                         scatter_count, best_br, best.selected_block_m,
                         online_out, unique_tokens);
    } else {
      zero_online_policy(online_out);
    }
  }
  return best;
}

WEAVE_HOST_DEVICE PolicyDecision cta0_select_policy(
    HardwareArch arch, const WorkloadSpec& wl, const WeaveKnobs& knobs,
    const uint32_t* m_e, uint32_t dedup_pulls, uint32_t scatter_count,
    OnlineRoutingPolicy* online_out = nullptr, uint32_t unique_tokens = 0u) {
  return cta0_select_policy(
      arch, get_hardware_spec(
                arch, static_cast<PrecisionMode>(knobs.precision_mode)),
      wl, knobs, m_e, dedup_pulls, scatter_count, online_out, unique_tokens);
}

}  // namespace weave




