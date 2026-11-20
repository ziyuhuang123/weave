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
static constexpr uint32_t kMaxRuntimeExperts = 128u;

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

WEAVE_HOST_DEVICE EquationBreakdown evaluate_spatial_candidate(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const SpatialCandidate& candidate, uint32_t scatter_mode,
    const uint32_t* m_e, uint32_t dedup_pulls, uint32_t scatter_count) {
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
  const float gpu_comm_units =
      static_cast<float>(hw.num_sms_or_cus * effective_comm_units_per_sm);
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
              fmaxf(1.0f, gpu_comm_units * hw.b_scat_comm_per_unit_gbps));

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
                      0.55f));
  out.t_tail_reduce_us =
      static_cast<float>(total_tokens) * h * hw.out_bytes * 2.0f /
      (b_tail_gbps * 1000.0f);
  out.t_total_us = hw.t_head_us + out.t_mid_us +
      (WEAVE_TAIL_FINE_GRAINED ? out.t_tail_reduce_us : hw.t_tail_us);
  return out;
}

WEAVE_HOST_DEVICE PolicyDecision make_override_decision(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count) {
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
      m_e, dedup_pulls, scatter_count);

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
  return d;
}

// CTA-0 entry point.  One thread calls this function; the histogram and all
// routing counters must be visible before invocation.  Explicit WeaveKnobs
// overrides bypass the search but still pass through the same equation path.
WEAVE_HOST_DEVICE PolicyDecision cta0_select_policy(
    HardwareArch arch, const HardwareSpec& hw, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count) {
  if (!knobs.enable_runtime_cost_model ||
      has_override(knobs, kOverrideSpatial | kOverrideScatter |
                                kOverrideVHead | kOverrideTemporal)) {
    return make_override_decision(
        arch, hw, wl, knobs, m_e, dedup_pulls, scatter_count);
  }

  const CandidateSet candidates = get_candidate_set(
      arch, static_cast<PrecisionMode>(knobs.precision_mode));
  PolicyDecision best{};
  best.magic = kPolicyMagic;
  best.version = kPolicyVersion;
  best.arch = static_cast<uint32_t>(arch);
  best.valid = 0u;
  best.predicted_total_us = 3.402823466e+38f;

  const auto consider = [&](uint32_t c, uint32_t s,
                            const EquationBreakdown& br) {
    if (!best.valid || br.t_total_us < best.predicted_total_us) {
      best.valid = 1u;
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

  for (uint32_t c = 0u; c < candidates.count; ++c) {
    // Eq.(1) and the GEMM/epilogue part of Eq.(2) are independent of s.
    // Evaluate them once, then derive the s=comm mid/total terms in place.
    // This halves CTA-0's histogram traversal on the critical path.
    const EquationBreakdown br_epi = evaluate_spatial_candidate(
        arch, hw, wl, candidates.items[c], 0u,
        m_e, dedup_pulls, scatter_count);
    consider(c, 0u, br_epi);

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
  return best;
}

WEAVE_HOST_DEVICE PolicyDecision cta0_select_policy(
    HardwareArch arch, const WorkloadSpec& wl, const WeaveKnobs& knobs,
    const uint32_t* m_e, uint32_t dedup_pulls, uint32_t scatter_count) {
  return cta0_select_policy(
      arch, get_hardware_spec(
                arch, static_cast<PrecisionMode>(knobs.precision_mode)),
      wl, knobs, m_e, dedup_pulls, scatter_count);
}

}  // namespace weave




