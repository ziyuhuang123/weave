// Standalone C++17 host/device-model unit test for Weave Eq. (1)-(3).
// Build with either:
//   nvcc -std=c++17 -x cu tests/test_cost_model_eq123.cpp -o test_cost_model
//   cl /std:c++17 /EHsc tests\test_cost_model_eq123.cpp
//
// The test intentionally recomputes every equation independently from the
// production evaluator and compares all fields.  It also brute-forces the
// CTA-0 argmin over the same candidate set.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include <deep_gemm/scheduler/weave_cost_model.cuh>
#include <deep_gemm/scheduler/weave_policy_runtime.cuh>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void expect_near(float got, float expected, float rel_tol, const char* what) {
  const float scale = std::max(1.0f, std::fabs(expected));
  const float diff = std::fabs(got - expected);
  if (diff > rel_tol * scale) {
    std::fprintf(stderr,
                 "FAIL: %s got=%.9g expected=%.9g rel=%.6g\n",
                 what, got, expected, diff / scale);
    ++failures;
  }
}

uint32_t cdiv(uint32_t n, uint32_t d) { return (n + d - 1u) / d; }

struct Expected {
  float f_gemm;
  float v_mem;
  float v_epi;
  float t_mma;
  float t_epi;
  float t_comp;
  float t_pull;
  float t_scat;
  float t_mid;
};

Expected recompute(const weave::HardwareSpec& hw,
                   const weave::WorkloadSpec& wl,
                   const weave::SpatialCandidate& c,
                   uint32_t scatter_mode,
                   const std::vector<uint32_t>& m_e,
                   uint32_t d_in, uint32_t m_out) {
  uint32_t m = 0u;
  float f_gemm = 0.0f;
  float weight_bytes = 0.0f;
  for (uint32_t e = 0u; e < wl.local_experts; ++e) {
    m += m_e[e];
    const uint32_t k = cdiv(m_e[e], c.block_m);
    f_gemm += static_cast<float>(k * c.block_m) * 6.0f *
              static_cast<float>(wl.hidden) * static_cast<float>(wl.intermediate);
    weight_bytes += static_cast<float>(k) * 3.0f *
                    static_cast<float>(wl.hidden * wl.intermediate) * hw.weight_bytes;
  }
  Expected out{};
  out.f_gemm = f_gemm;
  out.v_mem = weight_bytes + static_cast<float>(m) * hw.act_in_bytes *
                                 (static_cast<float>(wl.hidden) +
                                  static_cast<float>(wl.intermediate));
  out.t_mma = std::max(out.v_mem / (hw.b_hbm_gbps * 1000.0f),
                       f_gemm / (c.p_tc_tflops * 1000000.0f));
  out.v_epi = static_cast<float>(m) *
              (hw.act_in_bytes * static_cast<float>(wl.intermediate) +
               hw.out_bytes * static_cast<float>(wl.hidden));
  const float b_epi = std::min(
      hw.b_hbm_gbps,
      std::max(1.0f, static_cast<float>(hw.num_sms_or_cus) *
                         static_cast<float>(c.comp_units) *
                         hw.b_epi_write_per_unit_gbps));
  out.t_epi = out.v_epi / (b_epi * 1000.0f);
  out.t_comp = hw.arch == weave::HardwareArch::B300_SM100
                   ? std::max(out.t_mma, out.t_epi)
                   : out.t_mma + out.t_epi;
  const uint32_t eff_comm_per_sm = c.comm_units == 0u ? c.comp_units : c.comm_units;
  const float gpu_comm = static_cast<float>(hw.num_sms_or_cus * eff_comm_per_sm);
  const float gpu_epi = static_cast<float>(hw.num_sms_or_cus * c.comp_units);
  const float b_pull = std::min(
      hw.b_link_gbps, std::max(1.0f, gpu_comm * hw.b_pull_per_unit_gbps));
  const float b_scat =
      scatter_mode == static_cast<uint32_t>(weave::ScatterPath::EpilogueInline)
          ? std::min(hw.b_link_gbps,
                     std::max(1.0f, gpu_epi * hw.b_scat_epi_per_unit_gbps))
          : std::min(hw.b_link_gbps,
                     std::max(1.0f, gpu_comm * hw.b_scat_comm_per_unit_gbps));
  out.t_pull = static_cast<float>(d_in) * static_cast<float>(wl.hidden) *
               hw.act_in_bytes / (b_pull * 1000.0f);
  out.t_scat = static_cast<float>(m_out) * static_cast<float>(wl.hidden) *
               hw.out_bytes / (b_scat * 1000.0f);
  if (scatter_mode == static_cast<uint32_t>(weave::ScatterPath::EpilogueInline)) {
    out.t_mid = std::max(out.t_pull, out.t_comp + out.t_scat);
  } else {
    out.t_mid = std::max(out.t_pull + out.t_scat, out.t_comp + hw.t_handoff_us);
  }
  return out;
}

void test_arch(weave::HardwareArch arch, const char* name) {
  const auto hw = weave::get_hardware_spec(arch);
  const auto candidates = weave::get_candidate_set(arch);
  const weave::WorkloadSpec wl{4u, 128u, 64u};
  const std::vector<uint32_t> m_e = {0u, 3u, 5u, 9u};
  const uint32_t d_in = 23u;
  const uint32_t m_out = 17u;

  check(candidates.count >= 2u, "candidate count");
  for (uint32_t ci = 0u; ci < candidates.count; ++ci) {
    const auto& c = candidates.items[ci];
    for (uint32_t s = 0u; s < 2u; ++s) {
      const auto br = weave::evaluate_spatial_candidate(
          arch, hw, wl, c, s, m_e.data(), d_in, m_out);
      const auto ex = recompute(hw, wl, c, s, m_e, d_in, m_out);
      expect_near(br.f_gemm_flops, ex.f_gemm, 2e-6f, "Eq1 F_gemm");
      expect_near(br.v_mem_bytes, ex.v_mem, 2e-6f, "Eq1 V_mem");
      expect_near(br.v_epi_bytes, ex.v_epi, 2e-6f, "Eq2 V_epi");
      expect_near(br.t_mma_us, ex.t_mma, 2e-6f, "Eq2 T_mma");
      expect_near(br.t_epi_us, ex.t_epi, 2e-6f, "Eq2 T_epi");
      expect_near(br.t_comp_us, ex.t_comp, 2e-6f, "Eq2 T_comp");
      expect_near(br.t_pull_us, ex.t_pull, 2e-6f, "Eq3 T_pull");
      expect_near(br.t_scat_us, ex.t_scat, 2e-6f, "Eq3 T_scat");
      expect_near(br.t_mid_us, ex.t_mid, 2e-6f, "Eq3 T_mid");
      check(br.total_tokens == 17u, "M = sum m_e");
      if (arch == weave::HardwareArch::B300_SM100) {
        check(std::fabs(br.t_comp_us - std::max(br.t_mma_us, br.t_epi_us)) < 1e-4f,
              "B300 T_comp=max");
      } else {
        check(std::fabs(br.t_comp_us - (br.t_mma_us + br.t_epi_us)) < 1e-4f,
              "H20/MI308X T_comp=sum");
      }
    }
  }

  weave::WeaveKnobs knobs = weave::default_weave_knobs(arch);
  // The brute-force oracle below assumes the runtime search, so pin it on:
  // ablation presets compile WEAVE_ENABLE_COST_MODEL=0 and would otherwise
  // route this call through the fixed override path.
  knobs.enable_runtime_cost_model = 1u;
  knobs.override_mask = 0u;
  const auto decision = weave::cta0_select_policy(
      arch, wl, knobs, m_e.data(), d_in, m_out);
  check(weave::policy_is_valid(decision), "auto decision valid");
  check(decision.input_total_tokens == 17u, "auto M snapshot");
  check(decision.selected_v_head > 0u, "auto V_head selected");
  check(decision.selected_r_mask == candidates.items[decision.selected_candidate].r_mask,
        "selected r_mask consistent");

  float best_total = 1.0e30f;
  uint32_t best_c = 0u, best_s = 0u;
  for (uint32_t ci = 0u; ci < candidates.count; ++ci) {
    for (uint32_t s = 0u; s < 2u; ++s) {
      const auto br = weave::evaluate_spatial_candidate(
          arch, hw, wl, candidates.items[ci], s, m_e.data(), d_in, m_out,
          weave::resolve_reduce_warps(knobs));
      if (br.t_total_us < best_total) {
        best_total = br.t_total_us;
        best_c = ci;
        best_s = s;
      }
    }
  }
  check(decision.selected_candidate == best_c, "argmin candidate");
  check(decision.selected_scatter_mode == best_s, "argmin scatter");
  expect_near(decision.predicted_total_us, best_total, 2e-6f, "argmin total");

  // Explicit override path: same equations, forced policy fields.
  knobs.override_mask = weave::kOverrideSpatial | weave::kOverrideScatter |
                        weave::kOverrideVHead;
  knobs.override_r_mask = candidates.items[1].r_mask;
  knobs.override_scatter_mode = 0u;
  knobs.override_v_head = 32u;
  const auto forced = weave::cta0_select_policy(
      arch, wl, knobs, m_e.data(), d_in, m_out);
  check(weave::policy_is_valid(forced), "override valid");
  check(forced.selected_r_mask == candidates.items[1].r_mask, "override r_mask");
  check(forced.selected_scatter_mode == 0u, "override scatter");
  check(forced.selected_v_head == 32u, "override V_head");
  std::printf("[PASS] %s: candidates=%u auto=(r=%u,s=%u,v_head=%u,total=%.3fus)\n",
              name, candidates.count, decision.selected_candidate,
              decision.selected_scatter_mode, decision.selected_v_head,
              decision.predicted_total_us);
}

void test_precision_modes() {
  using weave::PrecisionMode;
  const auto base = weave::get_hardware_spec(weave::HardwareArch::B300_SM100);
  const auto nv = weave::get_hardware_spec(
      weave::HardwareArch::B300_SM100, PrecisionMode::NVFP4);
  const auto fp8 = weave::get_hardware_spec(
      weave::HardwareArch::B300_SM100, PrecisionMode::FP8xFP8);
  const auto bf16 = weave::get_hardware_spec(
      weave::HardwareArch::B300_SM100, PrecisionMode::BF16);

  // Paper Table-I default must remain bit-identical.
  check(base.weight_bytes == 0.5f && base.act_in_bytes == 0.5625f,
        "FP8xFP4 Table-I byte widths");
  check(nv.weight_bytes == 0.5625f && nv.act_in_bytes == 0.5625f,
        "NVFP4 byte widths");
  check(fp8.weight_bytes == 1.03125f && fp8.act_in_bytes == 1.03125f,
        "FP8xFP8 byte widths");
  check(bf16.weight_bytes == 2.0f && bf16.act_in_bytes == 2.03125f,
        "BF16 byte widths");
  check(nv.p_tc_full_tflops == 2.0f * base.p_tc_full_tflops,
        "NVFP4 tensor-core rate");
  check(bf16.p_tc_full_tflops == 0.5f * base.p_tc_full_tflops,
        "BF16 tensor-core rate");

  check(std::strcmp(weave::precision_mode_name(PrecisionMode::FP8xFP4),
                    "fp8xfp4") == 0, "precision name fp8xfp4");
  check(std::strcmp(weave::precision_mode_name(PrecisionMode::NVFP4),
                    "fp4xfp4") == 0, "precision name fp4xfp4");
  check(std::strcmp(weave::precision_mode_name(PrecisionMode::FP8xFP8),
                    "fp8") == 0, "precision name fp8");
  check(std::strcmp(weave::precision_mode_name(PrecisionMode::BF16),
                    "bf16") == 0, "precision name bf16");

  const weave::WorkloadSpec wl{4u, 128u, 64u};
  const std::vector<uint32_t> m_e = {0u, 3u, 5u, 9u};
  for (PrecisionMode mode : {PrecisionMode::FP8xFP4, PrecisionMode::NVFP4,
                             PrecisionMode::FP8xFP8, PrecisionMode::BF16}) {
    const auto hw = weave::get_hardware_spec(
        weave::HardwareArch::B300_SM100, mode);
    const auto candidates = weave::get_candidate_set(
        weave::HardwareArch::B300_SM100, mode);
    const auto knobs = weave::default_weave_knobs(weave::HardwareArch::B300_SM100, mode);
    check(knobs.precision_mode == static_cast<uint32_t>(mode),
          "precision knob propagation");
    const auto decision = weave::cta0_select_policy(
        weave::HardwareArch::B300_SM100, wl, knobs, m_e.data(), 23u, 17u);
    check(weave::policy_is_valid(decision), "precision decision valid");
    check(decision.predicted_total_us > 0.0f, "precision prediction positive");
    // Every candidate must use the precision-scaled tensor-core rate.
    for (uint32_t c = 0u; c < candidates.count; ++c) {
      check(candidates.items[c].p_tc_tflops ==
                (mode == PrecisionMode::NVFP4 ? 2.0f :
                 mode == PrecisionMode::BF16 ? 0.5f : 1.0f) *
                    weave::get_candidate_set(
                        weave::HardwareArch::B300_SM100).items[c].p_tc_tflops,
            "candidate p_tc precision scaling");
    }
    const auto br = weave::evaluate_spatial_candidate(
        weave::HardwareArch::B300_SM100, hw, wl, candidates.items[0], 0u,
        m_e.data(), 23u, 17u);
    check(br.t_total_us > 0.0f, "precision equation total positive");
  }
  std::printf("[PASS] B300 precision modes: fp8xfp4/nvfp4/fp8/bf16\n");
}


// ---------------------------------------------------------------------------
// Independent WEAVE_REDUCE_WARPS / Eq.(3)-tail references.  These are written
// in double precision straight from the knob contract, never by calling the
// production evaluator, so a wiring mistake cannot hide behind shared code.
// ---------------------------------------------------------------------------
double reference_reduce_tail_us(const weave::HardwareSpec& hw,
                                const weave::WorkloadSpec& wl,
                                uint32_t total_tokens, double v_mem_bytes,
                                double t_comp_us, double t_mid_us,
                                uint32_t reduce_warps) {
  const double b_tail = std::min(
      static_cast<double>(hw.b_hbm_gbps),
      std::max(1.0, static_cast<double>(hw.num_sms_or_cus) *
                        static_cast<double>(hw.max_units_per_sm) * 0.55));
  const double bytes = static_cast<double>(total_tokens) *
      static_cast<double>(wl.hidden) * static_cast<double>(hw.out_bytes) * 2.0;
  if (reduce_warps == 0u) {
    return bytes / (b_tail * 1000.0);
  }
  const double used = t_comp_us > 0.0
      ? v_mem_bytes / (t_comp_us * 1000.0)
      : static_cast<double>(hw.b_hbm_gbps);
  const double spare = std::max(0.0, static_cast<double>(hw.b_hbm_gbps) - used);
  const double b_reduce = std::min(
      spare, static_cast<double>(hw.num_sms_or_cus) *
                 static_cast<double>(reduce_warps) * 0.55);
  return std::max(0.0, bytes - b_reduce * 1000.0 * t_mid_us) / (b_tail * 1000.0);
}

double reference_scat_us(const weave::HardwareSpec& hw,
                         const weave::WorkloadSpec& wl,
                         const weave::SpatialCandidate& c,
                         uint32_t scatter_mode, uint32_t m_out,
                         uint32_t reduce_warps) {
  const uint32_t eff_comm = c.comm_units == 0u ? c.comp_units : c.comm_units;
  uint32_t comm = eff_comm;
  if (reduce_warps > 0u && scatter_mode != 0u) {
    comm = eff_comm > reduce_warps ? eff_comm - reduce_warps : 1u;
  }
  const bool epi =
      scatter_mode == static_cast<uint32_t>(weave::ScatterPath::EpilogueInline);
  const double units = static_cast<double>(hw.num_sms_or_cus) *
      static_cast<double>(epi ? c.comp_units : comm);
  const double per_unit =
      epi ? hw.b_scat_epi_per_unit_gbps : hw.b_scat_comm_per_unit_gbps;
  const double b_scat = std::min(static_cast<double>(hw.b_link_gbps),
                                 std::max(1.0, units * per_unit));
  return m_out == 0u ? 0.0
      : static_cast<double>(m_out) * static_cast<double>(wl.hidden) *
            static_cast<double>(hw.out_bytes) / (b_scat * 1000.0);
}

float paper_total_us(const weave::HardwareSpec& hw, float t_mid_us,
                     float t_tail_reduce_us) {
  return hw.t_head_us + t_mid_us +
      (WEAVE_TAIL_FINE_GRAINED ? t_tail_reduce_us : hw.t_tail_us);
}

struct Argmin {
  uint32_t c;
  uint32_t s;
  float total;
};

Argmin brute_force_argmin(weave::HardwareArch arch,
                          const weave::HardwareSpec& hw,
                          const weave::WorkloadSpec& wl,
                          const weave::CandidateSet& candidates,
                          const std::vector<uint32_t>& m_e,
                          uint32_t d_in, uint32_t m_out,
                          uint32_t reduce_warps) {
  Argmin best{0u, 0u, 3.402823466e+38f};
  for (uint32_t c = 0u; c < candidates.count; ++c) {
    for (uint32_t s = 0u; s < 2u; ++s) {
      const auto br = weave::evaluate_spatial_candidate(
          arch, hw, wl, candidates.items[c], s, m_e.data(), d_in, m_out,
          reduce_warps);
      if (br.t_total_us < best.total) {
        best = Argmin{c, s, br.t_total_us};
      }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// kMaxRuntimeExperts = 256: every 128 < E_local <= 256 workload must produce a
// valid PolicyDecision, and PolicyControl must be able to hold the histogram.
// ---------------------------------------------------------------------------
void test_max_runtime_experts() {
  static_assert(weave::kMaxRuntimeExperts == 256u,
                "runtime expert capacity must cover E_local up to 256");
  check(weave::kMaxRuntimeExperts == 256u, "kMaxRuntimeExperts == 256");

  weave::PolicyControl control{};
  check(sizeof(control.histogram) / sizeof(control.histogram[0]) ==
            weave::kMaxRuntimeExperts, "PolicyControl histogram capacity");
  check(sizeof(weave::PolicyControl) >=
            sizeof(weave::PolicyDecision) + 8u * sizeof(uint32_t) +
                weave::kMaxRuntimeExperts * sizeof(uint32_t),
        "PolicyControl sized for 256 histogram entries");
  weave::reset_policy_control(&control);
  check(control.magic == weave::kPolicyMagic &&
            control.version == weave::kPolicyVersion,
        "reset_policy_control header");
  bool zeroed = true;
  for (uint32_t e = 0u; e < weave::kMaxRuntimeExperts; ++e) {
    zeroed = zeroed && control.histogram[e] == 0u;
  }
  check(zeroed, "reset_policy_control clears all 256 entries");
  uint32_t expected_sum = 0u;
  for (uint32_t e = 0u; e < weave::kMaxRuntimeExperts; ++e) {
    control.histogram[e] = e;
    expected_sum += e;
  }
  check(control.histogram[weave::kMaxRuntimeExperts - 1u] ==
            weave::kMaxRuntimeExperts - 1u, "last histogram slot writable");
  check(weave::sum_histogram(control.histogram, weave::kMaxRuntimeExperts) ==
            expected_sum, "256-entry histogram round trip");
  check(control.decision.magic == 0u,
        "histogram writes never reach the decision fields");
  // Over-capacity rejection lives in the device-only packed-histogram rail, so
  // it is exercised on a GPU by test_device_experts256_rail() below rather than
  // asserted here against a host-inaccessible function.

  for (weave::HardwareArch arch : {weave::HardwareArch::B300_SM100,
                                   weave::HardwareArch::H20_SM90,
                                   weave::HardwareArch::MI308X_GFX942}) {
    const auto hw = weave::get_hardware_spec(arch);
    const auto candidates = weave::get_candidate_set(arch);
    for (uint32_t experts : {128u, 129u, 192u, 256u}) {
      const weave::WorkloadSpec wl{experts, 2048u, 512u};
      std::vector<uint32_t> m_e(experts);
      uint32_t total = 0u;
      for (uint32_t e = 0u; e < experts; ++e) {
        m_e[e] = e % 5u;  // includes empty experts
        total += m_e[e];
      }
      const uint32_t d_in = total + 13u;
      for (uint32_t rw : {0u, 4u}) {
        weave::WeaveKnobs knobs = weave::default_weave_knobs(arch);
        knobs.enable_runtime_cost_model = 1u;
        knobs.override_mask = 0u;
        knobs.reduce_warps = rw;
        const auto d = weave::cta0_select_policy(
            arch, wl, knobs, m_e.data(), d_in, total);
        check(weave::policy_is_valid(d), "E_local<=256 decision valid");
        check(d.valid == 1u, "E_local<=256 valid flag");
        check(d.input_total_tokens == total, "E_local<=256 M snapshot");
        check(d.input_dedup_pulls == d_in, "E_local<=256 D_in snapshot");
        check(d.input_scatter_rows == total, "E_local<=256 M_out snapshot");
        check(d.selected_v_head > 0u, "E_local<=256 V_head selected");
        check(d.predicted_total_us > 0.0f && std::isfinite(d.predicted_total_us),
              "E_local<=256 finite positive prediction");
        const auto best = brute_force_argmin(
            arch, hw, wl, candidates, m_e, d_in, total, rw);
        check(d.selected_candidate == best.c, "E_local<=256 argmin candidate");
        check(d.selected_scatter_mode == best.s, "E_local<=256 argmin scatter");
        expect_near(d.predicted_total_us, best.total, 2e-6f,
                    "E_local<=256 argmin total");
      }
    }
  }
  std::printf("[PASS] kMaxRuntimeExperts=256: E_local in {128,129,192,256} valid "
              "on B300/H20/MI308X (reduce_warps 0 and 4)\n");
}

// ---------------------------------------------------------------------------
// WEAVE_V_HEAD / knobs.override_v_head: the override is published verbatim and
// never perturbs Eq. (1)-(3).
// ---------------------------------------------------------------------------
void test_v_head_override() {
  check(weave::kCompiledVHeadDefault == WEAVE_V_HEAD,
        "compiled WEAVE_V_HEAD default exposed");
  const bool compiled_override =
      (WEAVE_V_HEAD != 0u && WEAVE_V_HEAD != weave::kAutoU32);

  for (weave::HardwareArch arch : {weave::HardwareArch::B300_SM100,
                                   weave::HardwareArch::H20_SM90,
                                   weave::HardwareArch::MI308X_GFX942}) {
    const auto hw = weave::get_hardware_spec(arch);
    const auto candidates = weave::get_candidate_set(arch);
    const weave::WorkloadSpec wl{4u, 128u, 64u};
    const std::vector<uint32_t> m_e = {0u, 3u, 5u, 9u};
    const uint32_t d_in = 23u, m_out = 17u;

    weave::WeaveKnobs knobs = weave::default_weave_knobs(arch);
    knobs.enable_runtime_cost_model = 1u;
    knobs.override_mask = 0u;
    knobs.reduce_warps = 0u;
    const auto auto_d = weave::cta0_select_policy(
        arch, wl, knobs, m_e.data(), d_in, m_out);
    check(weave::policy_is_valid(auto_d), "V_head auto decision valid");
    check(auto_d.selected_v_head ==
              weave::choose_v_head(hw, candidates.items[auto_d.selected_candidate],
                                   d_in),
          "auto V_head from head window");
    check(auto_d.selected_v_head > 0u && auto_d.selected_v_head % 32u == 0u,
          "auto V_head 32-aligned");
    check(weave::effective_v_head(knobs, 77u) == 77u,
          "effective_v_head passthrough without override bit");

    const auto zero_d = weave::cta0_select_policy(
        arch, wl, knobs, m_e.data(), 0u, 0u);
    check(weave::policy_is_valid(zero_d), "zero-pull decision valid");
    check(zero_d.selected_v_head == 0u, "V_head=0 when D_in=0");

    // Default-r_mask candidate and its auto scatter mode, as resolved by the
    // override path when only kOverrideVHead is set.
    const auto& c0 = candidates.items[0];
    check(c0.r_mask == weave::default_r_mask_for_arch(arch),
          "default r_mask is candidate 0");
    const uint32_t expect_s = c0.comm_units == 0u ? 0u : 1u;
    const auto br0 = weave::evaluate_spatial_candidate(
        arch, hw, wl, c0, expect_s, m_e.data(), d_in, m_out);

    for (uint32_t v : {32u, 64u, 256u, 1024u}) {
      knobs.override_mask = weave::kOverrideVHead;
      knobs.override_v_head = v;
      check(weave::effective_v_head(knobs, 77u) == v,
            "effective_v_head override");
      const auto d = weave::cta0_select_policy(
          arch, wl, knobs, m_e.data(), d_in, m_out);
      check(weave::policy_is_valid(d), "V_head override decision valid");
      check(d.selected_v_head == v, "override V_head published verbatim");
      check(d.selected_r_mask == c0.r_mask, "V_head override keeps default r_mask");
      check(d.selected_scatter_mode == expect_s,
            "V_head override keeps auto scatter mode");
      // V_head is a published head budget: it must not move any equation.
      check(d.predicted_total_us == br0.t_total_us,
            "V_head override leaves T_total bit-identical");
      check(d.predicted_mid_us == br0.t_mid_us,
            "V_head override leaves T_mid bit-identical");
      check(d.predicted_comp_us == br0.t_comp_us,
            "V_head override leaves T_comp bit-identical");
      check(d.predicted_pull_us == br0.t_pull_us,
            "V_head override leaves T_pull bit-identical");
      check(d.predicted_scat_us == br0.t_scat_us,
            "V_head override leaves T_scat bit-identical");
    }

    // Compile-time WEAVE_V_HEAD plumbing through default_weave_knobs().
    const auto def = weave::default_weave_knobs(arch);
    check(compiled_override ==
              ((def.override_mask & weave::kOverrideVHead) != 0u),
          "compiled WEAVE_V_HEAD sets kOverrideVHead only when non-auto");
    if (compiled_override) {
      check(def.override_v_head == WEAVE_V_HEAD,
            "compiled WEAVE_V_HEAD carried into knobs");
      const auto d = weave::cta0_select_policy(
          arch, wl, def, m_e.data(), d_in, m_out);
      check(weave::policy_is_valid(d), "compiled V_head decision valid");
      check(d.selected_v_head == WEAVE_V_HEAD,
            "compiled WEAVE_V_HEAD published");
    } else {
      check(def.override_v_head == weave::kAutoU32,
            "auto WEAVE_V_HEAD keeps kAutoU32 sentinel");
    }
  }
  std::printf("[PASS] WEAVE_V_HEAD: auto head window, overrides {32,64,256,1024} "
              "verbatim, Eq.(1)-(3) untouched (compiled=%u)\n",
              compiled_override ? WEAVE_V_HEAD : 0u);
}

// ---------------------------------------------------------------------------
// WEAVE_REDUCE_WARPS = 0 / 2 / 4.
//   * 0 is bit-identical to Paper Eq. (1)-(3).
//   * >0 overlaps the Combine reduction with T_mid on spare Middle HBM
//     bandwidth and carves the lanes out of the middle comm scatter group.
// ---------------------------------------------------------------------------
void test_reduce_warps() {
  check(weave::kMaxReduceWarps == 8u, "kMaxReduceWarps == 8");
  check(WEAVE_REDUCE_WARPS <= 8u, "compiled WEAVE_REDUCE_WARPS in range");

  for (weave::HardwareArch arch : {weave::HardwareArch::B300_SM100,
                                   weave::HardwareArch::H20_SM90,
                                   weave::HardwareArch::MI308X_GFX942}) {
    const auto hw = weave::get_hardware_spec(arch);
    const auto candidates = weave::get_candidate_set(arch);
    const weave::WorkloadSpec wl{4u, 128u, 64u};
    const std::vector<uint32_t> m_e = {0u, 3u, 5u, 9u};
    const uint32_t d_in = 23u, m_out = 17u, total = 17u;

    for (uint32_t ci = 0u; ci < candidates.count; ++ci) {
      const auto& c = candidates.items[ci];
      for (uint32_t s = 0u; s < 2u; ++s) {
        const auto base = weave::evaluate_spatial_candidate(
            arch, hw, wl, c, s, m_e.data(), d_in, m_out, 0u);
        const auto dflt = weave::evaluate_spatial_candidate(
            arch, hw, wl, c, s, m_e.data(), d_in, m_out);
        check(base.t_total_us == dflt.t_total_us &&
                  base.t_mid_us == dflt.t_mid_us &&
                  base.t_scat_us == dflt.t_scat_us &&
                  base.t_tail_reduce_us == dflt.t_tail_reduce_us,
              "reduce_warps defaults to the paper path");

        // Paper Eq.(1)-(3) reference for reduce_warps == 0.
        const auto ex = recompute(hw, wl, c, s, m_e, d_in, m_out);
        const double tail0 = reference_reduce_tail_us(
            hw, wl, total, base.v_mem_bytes, base.t_comp_us, base.t_mid_us, 0u);
        expect_near(base.t_tail_reduce_us, static_cast<float>(tail0), 2e-6f,
                    "rw=0 Eq3 tail");
        expect_near(base.t_scat_us,
                    static_cast<float>(reference_scat_us(hw, wl, c, s, m_out, 0u)),
                    2e-6f, "rw=0 Eq3 scatter");
        expect_near(base.t_total_us,
                    paper_total_us(hw, ex.t_mid, static_cast<float>(tail0)),
                    2e-6f, "rw=0 paper total");

        float prev_tail = base.t_tail_reduce_us;
        for (uint32_t rw : {2u, 4u}) {
          const auto br = weave::evaluate_spatial_candidate(
              arch, hw, wl, c, s, m_e.data(), d_in, m_out, rw);
          // Eq.(1), Eq.(2), the pull and the head budget are untouched.
          check(br.f_gemm_flops == base.f_gemm_flops &&
                    br.v_mem_bytes == base.v_mem_bytes &&
                    br.v_epi_bytes == base.v_epi_bytes,
                "reduce warps leave Eq1/Eq2 bytes bit-identical");
          check(br.t_mma_us == base.t_mma_us && br.t_epi_us == base.t_epi_us &&
                    br.t_comp_us == base.t_comp_us,
                "reduce warps leave Eq2 times bit-identical");
          check(br.t_pull_us == base.t_pull_us,
                "reduce warps leave T_pull bit-identical");
          check(br.v_head == base.v_head, "reduce warps leave V_head unchanged");

          // Middle scatter: carved-out comm units, epilogue path untouched.
          expect_near(br.t_scat_us,
                      static_cast<float>(reference_scat_us(hw, wl, c, s, m_out, rw)),
                      2e-5f, "reduce-warps middle scatter bandwidth");
          if (s == static_cast<uint32_t>(weave::ScatterPath::EpilogueInline)) {
            check(br.t_scat_us == base.t_scat_us,
                  "epilogue-inline scatter keeps its units");
            check(br.t_mid_us == base.t_mid_us,
                  "epilogue-inline T_mid unchanged");
          } else {
            check(br.t_scat_us >= base.t_scat_us,
                  "carved-out comm units never speed up the queue");
          }
          const float expect_mid =
              s == static_cast<uint32_t>(weave::ScatterPath::EpilogueInline)
                  ? std::max(br.t_pull_us, br.t_comp_us + br.t_scat_us)
                  : std::max(br.t_pull_us + br.t_scat_us,
                             br.t_comp_us + hw.t_handoff_us);
          expect_near(br.t_mid_us, expect_mid, 2e-6f, "reduce-warps T_mid");

          // Tail: spare Middle HBM bandwidth drained by the dedicated lanes.
          expect_near(br.t_tail_reduce_us,
                      static_cast<float>(reference_reduce_tail_us(
                          hw, wl, total, br.v_mem_bytes, br.t_comp_us,
                          br.t_mid_us, rw)),
                      2e-5f, "reduce-warps overlapped tail");
          check(br.t_tail_reduce_us <= prev_tail + 1e-9f,
                "more reduce warps never grow the tail");
          prev_tail = br.t_tail_reduce_us;
          expect_near(br.t_total_us,
                      paper_total_us(hw, br.t_mid_us, br.t_tail_reduce_us),
                      2e-6f, "reduce-warps total identity");
          if (s == static_cast<uint32_t>(weave::ScatterPath::EpilogueInline) &&
              WEAVE_TAIL_FINE_GRAINED) {
            check(br.t_total_us <= base.t_total_us + 1e-9f,
                  "epilogue-inline total never regresses with reduce warps");
          }
        }
      }
    }

    // Policy level: valid decisions and the reduce-aware argmin for 0/2/4.
    float totals[3] = {0.0f, 0.0f, 0.0f};
    const uint32_t rw_set[3] = {0u, 2u, 4u};
    for (uint32_t k = 0u; k < 3u; ++k) {
      weave::WeaveKnobs knobs = weave::default_weave_knobs(arch);
      knobs.enable_runtime_cost_model = 1u;
      knobs.override_mask = 0u;
      knobs.reduce_warps = rw_set[k];
      check(weave::resolve_reduce_warps(knobs) == rw_set[k],
            "resolve_reduce_warps passthrough");
      const auto d = weave::cta0_select_policy(
          arch, wl, knobs, m_e.data(), d_in, m_out);
      check(weave::policy_is_valid(d), "reduce-warps decision valid");
      const auto best = brute_force_argmin(
          arch, hw, wl, candidates, m_e, d_in, m_out, rw_set[k]);
      check(d.selected_candidate == best.c, "reduce-warps argmin candidate");
      check(d.selected_scatter_mode == best.s, "reduce-warps argmin scatter");
      expect_near(d.predicted_total_us, best.total, 2e-6f,
                  "reduce-warps argmin total");
      totals[k] = d.predicted_total_us;
    }
    // reduce_warps=0 must reproduce the independent paper argmin built from
    // recompute() + the double-precision Eq.(3) tail, with no production call.
    float paper_best = 3.402823466e+38f;
    for (uint32_t ci = 0u; ci < candidates.count; ++ci) {
      for (uint32_t s = 0u; s < 2u; ++s) {
        const auto ex = recompute(hw, wl, candidates.items[ci], s, m_e, d_in, m_out);
        const float tail = static_cast<float>(reference_reduce_tail_us(
            hw, wl, total, ex.v_mem, ex.t_comp, ex.t_mid, 0u));
        paper_best = std::min(paper_best, paper_total_us(hw, ex.t_mid, tail));
      }
    }
    expect_near(totals[0], paper_best, 2e-6f,
                "rw=0 policy total equals the independent paper argmin");

    // Out-of-range runtime values clamp to the validated maximum.
    weave::WeaveKnobs over = weave::default_weave_knobs(arch);
    over.enable_runtime_cost_model = 1u;
    over.override_mask = 0u;
    over.reduce_warps = 16u;
    check(weave::resolve_reduce_warps(over) == weave::kMaxReduceWarps,
          "resolve_reduce_warps clamps to 8");
    weave::WeaveKnobs eight = over;
    eight.reduce_warps = 8u;
    const auto d_over = weave::cta0_select_policy(
        arch, wl, over, m_e.data(), d_in, m_out);
    const auto d_eight = weave::cta0_select_policy(
        arch, wl, eight, m_e.data(), d_in, m_out);
    check(d_over.predicted_total_us == d_eight.predicted_total_us &&
              d_over.selected_candidate == d_eight.selected_candidate &&
              d_over.selected_scatter_mode == d_eight.selected_scatter_mode,
          "clamped reduce_warps matches 8 lanes");

    // The low-level evaluator clamps as well, so a direct call with an
    // out-of-range lane count can neither over-subscribe the comm group nor
    // invent tail bandwidth.  H20's 10-unit candidate is the sensitive case.
    for (uint32_t ci = 0u; ci < candidates.count; ++ci) {
      for (uint32_t s = 0u; s < 2u; ++s) {
        const auto clamped = weave::evaluate_spatial_candidate(
            arch, hw, wl, candidates.items[ci], s, m_e.data(), d_in, m_out, 16u);
        const auto eight_br = weave::evaluate_spatial_candidate(
            arch, hw, wl, candidates.items[ci], s, m_e.data(), d_in, m_out,
            weave::kMaxReduceWarps);
        check(clamped.t_scat_us == eight_br.t_scat_us &&
                  clamped.t_tail_reduce_us == eight_br.t_tail_reduce_us &&
                  clamped.t_total_us == eight_br.t_total_us,
              "evaluate_spatial_candidate clamps out-of-range reduce_warps");
      }
    }

    // Explicit spatial/scatter overrides still model the reduce lanes.
    weave::WeaveKnobs forced = weave::default_weave_knobs(arch);
    forced.override_mask = weave::kOverrideSpatial | weave::kOverrideScatter;
    forced.override_r_mask = candidates.items[1].r_mask;
    forced.override_scatter_mode =
        static_cast<uint32_t>(weave::ScatterPath::CommQueueAsync);
    forced.reduce_warps = 4u;
    const auto fd = weave::cta0_select_policy(
        arch, wl, forced, m_e.data(), d_in, m_out);
    const auto fbr = weave::evaluate_spatial_candidate(
        arch, hw, wl, candidates.items[1],
        static_cast<uint32_t>(weave::ScatterPath::CommQueueAsync),
        m_e.data(), d_in, m_out, 4u);
    check(weave::policy_is_valid(fd), "override + reduce warps valid");
    check(fd.predicted_total_us == fbr.t_total_us,
          "override path models reduce warps");
    check(fd.predicted_scat_us == fbr.t_scat_us,
          "override path carves comm units");

    std::printf("[PASS] %s WEAVE_REDUCE_WARPS: total_us 0=%.3f 2=%.3f 4=%.3f\n",
                arch == weave::HardwareArch::B300_SM100 ? "B300_SM100" :
                arch == weave::HardwareArch::H20_SM90 ? "H20_SM90" : "MI308X_GFX942",
                totals[0], totals[1], totals[2]);
  }
}

// ---------------------------------------------------------------------------
// Two-Level Tier-3: OnlineRoutingPolicy parity fixtures.
//
// 12 realistic routing distributions x 3 hardware x 3 knob modes.  Every
// field is printed as a machine-readable ONLINE line (floats as binary32 hex
// bits) so tests/test_two_level_adaptive_policy.py can compare bit-exactly
// against the Python mirror.  CALIB lines expose the offline calibration
// constants for the same cross-language check.
// ---------------------------------------------------------------------------

uint32_t float_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

const char* arch_name(weave::HardwareArch arch) {
  return arch == weave::HardwareArch::B300_SM100 ? "b300" :
         arch == weave::HardwareArch::H20_SM90 ? "h20" : "mi308x";
}

// Deterministic max-anchored proxy: one expert at max_n, the remaining
// nonzero-1 active experts share total-max_n evenly (floor, +1 to the first
// remainder).  Consistent with the measured pure21 aggregates.
std::vector<uint32_t> max_anchored(uint32_t total, uint32_t max_n,
                                   uint32_t nonzero, uint32_t experts) {
  std::vector<uint32_t> h(experts, 0u);
  if (nonzero == 0u || total == 0u) return h;
  if (nonzero == 1u) { h[0] = total; return h; }
  h[0] = max_n;
  const uint32_t rest = total - max_n;
  const uint32_t base = rest / (nonzero - 1u);
  const uint32_t rem = rest % (nonzero - 1u);
  for (uint32_t i = 1u; i < nonzero; ++i)
    h[i] = base + (i <= rem ? 1u : 0u);
  return h;
}

struct OnlineFixture {
  const char* name;
  uint32_t experts;
  uint32_t hidden;
  uint32_t inter;
  std::vector<uint32_t> m_e;
  uint32_t dedup_pulls;
  uint32_t scatter;
  uint32_t unique_tokens;
};

std::vector<OnlineFixture> online_fixtures() {
  std::vector<OnlineFixture> f;
  f.push_back({"uniform_decode_s128", 64u, 2048u, 512u,
               std::vector<uint32_t>(64u, 2u), 128u, 128u, 128u});
  f.push_back({"uniform_balanced_s4096", 128u, 2048u, 512u,
               std::vector<uint32_t>(128u, 32u), 4096u, 4096u, 4096u});
  // Real H20 pure21 rank aggregates (total/max/nonzero/dedup_rows/scatter).
  f.push_back({"h20_rank0_real", 128u, 8192u, 2048u,
               max_anchored(2482u, 131u, 123u, 128u), 964u, 1849u, 964u});
  f.push_back({"h20_rank2_hot", 128u, 8192u, 2048u,
               max_anchored(2937u, 346u, 122u, 128u), 1002u, 2181u, 1002u});
  {
    std::vector<uint32_t> h(128u, 8u); h[0] = 4096u;
    f.push_back({"skew_hot_1expert", 128u, 2048u, 512u, h, 5112u, 5112u, 5112u});
  }
  f.push_back({"ultra_dense_32k", 256u, 2048u, 512u,
               std::vector<uint32_t>(256u, 128u), 32768u, 32768u, 32768u});
  f.push_back({"ultra_dense_64k", 256u, 2048u, 512u,
               std::vector<uint32_t>(256u, 256u), 65536u, 65536u, 65536u});
  {
    std::vector<uint32_t> h(128u, 0u);
    for (uint32_t i = 0u; i < 16u; ++i) h[i] = 64u;
    f.push_back({"sparse_low_mu", 128u, 2048u, 512u, h, 1024u, 1024u, 1024u});
  }
  f.push_back({"high_mu_dup", 64u, 2048u, 512u,
               std::vector<uint32_t>(64u, 64u), 512u, 4096u, 512u});
  f.push_back({"mid_mu_thresh", 64u, 2048u, 512u,
               std::vector<uint32_t>(64u, 32u), 1024u, 2048u, 1024u});
  f.push_back({"empty", 128u, 2048u, 512u,
               std::vector<uint32_t>(128u, 0u), 0u, 0u, 0u});
  {
    std::vector<uint32_t> h(256u, 0u); h[0] = 65535u;
    f.push_back({"single_expert_max", 256u, 2048u, 512u, h, 1024u, 65535u, 1024u});
  }
  return f;
}

void emit_online(const char* arch, const char* fix, const char* mode,
                 const weave::OnlineRoutingPolicy& p) {
  std::printf("ONLINE arch=%s fix=%s mode=%s arw=%u avh=%u arg=%u edr=%u "
              "bmt=%u bme=%u ae=%u mx=%u tot=%u ada=%u hbm=%u tail=%u "
              "mean=%08x mu=%08x util=%08x frac=%08x\n",
              arch, fix, mode,
              p.active_reduce_warps, p.active_v_head, p.active_ready_group_size,
              p.enable_dedup_runtime, p.bm_tier_mean, p.bm_envelope,
              p.active_experts, p.max_n, p.total_routes, p.adaptive_active,
              p.reduce_gate_hbm_ok, p.reduce_gate_tail_ok,
              float_bits(p.mean_n), float_bits(p.rank_hit_multiplicity_mu),
              float_bits(p.middle_hbm_util), float_bits(p.tail_combine_frac));
}

void test_online_routing_policy() {
  const weave::HardwareArch archs[3] = {
      weave::HardwareArch::B300_SM100, weave::HardwareArch::H20_SM90,
      weave::HardwareArch::MI308X_GFX942};
  const auto fixtures = online_fixtures();
  check(fixtures.size() == 12u, "12 online routing fixtures");

  for (auto arch : archs) {
    const weave::HardwareCalibration cal = weave::get_hardware_calibration(arch);
    std::printf("CALIB arch=%s feff=%08x bhbm=%08x bcomm=%08x tau=%08x "
                "mu=%08x hbm=%08x tail=%08x vhb=%u slack=%u\n",
                arch_name(arch), float_bits(cal.f_eff_tflops),
                float_bits(cal.b_hbm_gbps), float_bits(cal.b_comm_gbps),
                float_bits(cal.tau_sync_us), float_bits(cal.mu_dedup_thresh),
                float_bits(cal.hbm_headroom_thresh),
                float_bits(cal.tail_ratio_thresh),
                cal.v_head_hot_budget, cal.bm_pad_slack);

    for (const auto& fx : fixtures) {
      const weave::WorkloadSpec wl{fx.experts, fx.hidden, fx.inter};
      const uint32_t total = weave::sum_histogram(fx.m_e.data(), fx.experts);

      // Mode auto: WEAVE_ADAPT_BM=1 (online tier on), R_max=4 compiled,
      // V_HEAD / READY_GROUP_SIZE / DEDUP_MODE all auto.
      weave::WeaveKnobs autok = weave::default_weave_knobs(arch);
      autok.enable_runtime_cost_model = 1u;
      autok.override_mask = 0u;
      autok.reduce_warps = 4u;
      autok.adapt_bm = 1u;
      autok.ready_group_size = 0u;
      autok.dedup_mode = weave::kAutoU32;
      weave::OnlineRoutingPolicy pa{};
      const auto da = weave::cta0_select_policy(
          arch, wl, autok, fx.m_e.data(), fx.dedup_pulls, fx.scatter,
          &pa, fx.unique_tokens);
      check(weave::policy_is_valid(da), "online auto decision valid");
      emit_online(arch_name(arch), fx.name, "auto", pa);

      // The online tier must never change the Eq.(1)-(3) decision itself.
      weave::WeaveKnobs basek = autok;
      basek.adapt_bm = 0u;
      const auto db = weave::cta0_select_policy(
          arch, wl, basek, fx.m_e.data(), fx.dedup_pulls, fx.scatter);
      check(da.predicted_total_us == db.predicted_total_us &&
                da.selected_r_mask == db.selected_r_mask &&
                da.selected_scatter_mode == db.selected_scatter_mode &&
                da.selected_v_head == db.selected_v_head &&
                da.predicted_comp_us == db.predicted_comp_us,
            "online tier leaves Eq.(1)-(3) decision bit-identical");

      // Gate semantics: adaptive R_max survives iff both gates pass.
      check(pa.active_reduce_warps == 0u || pa.active_reduce_warps == 4u,
            "active_reduce_warps in {0, R_max}");
      const bool hbm_ok = pa.middle_hbm_util <= cal.hbm_headroom_thresh;
      const bool tail_ok = pa.tail_combine_frac >= cal.tail_ratio_thresh;
      check(pa.reduce_gate_hbm_ok == (hbm_ok ? 1u : 0u) &&
                pa.reduce_gate_tail_ok == (tail_ok ? 1u : 0u),
            "reduce gate bits match the calibrated comparisons");
      check(pa.active_reduce_warps == ((hbm_ok && tail_ok) ? 4u : 0u),
            "adaptive reduce gate = hbm_ok && tail_ok");
      check(pa.total_routes == total, "online total = sum m_e");
      check(pa.adaptive_active == 1u, "adaptive_active tracks WEAVE_ADAPT_BM");
      check(pa.active_ready_group_size >= 1u && pa.active_ready_group_size <= 8u,
            "ready group in 1..8");
      check(pa.active_v_head % 32u == 0u, "auto v_head stays 32-aligned");

      // Mode pinned: explicit knobs, online tier OFF (adapt_bm=0).  Everything
      // deterministic: R_max stays, V_HEAD=64, group=2, dedup=1.
      weave::WeaveKnobs pink = weave::default_weave_knobs(arch);
      pink.enable_runtime_cost_model = 1u;
      pink.override_mask = weave::kOverrideVHead;
      pink.override_v_head = 64u;
      pink.reduce_warps = 4u;
      pink.adapt_bm = 0u;
      pink.ready_group_size = 2u;
      pink.dedup_mode = 1u;
      weave::OnlineRoutingPolicy pp{};
      const auto dp = weave::cta0_select_policy(
          arch, wl, pink, fx.m_e.data(), fx.dedup_pulls, fx.scatter,
          &pp, fx.unique_tokens);
      check(weave::policy_is_valid(dp), "online pinned decision valid");
      emit_online(arch_name(arch), fx.name, "pinned", pp);
      check(pp.active_reduce_warps == 4u, "pinned R_max deterministic");
      check(pp.active_v_head == 64u, "explicit V_HEAD overrides");
      check(pp.active_ready_group_size == 2u, "explicit READY_GROUP overrides");
      check(pp.enable_dedup_runtime == 1u, "explicit DEDUP_MODE=1 overrides");
      check(pp.adaptive_active == 0u, "pinned mode keeps online tier off");
      const auto dp_plain = weave::cta0_select_policy(
          arch, wl, pink, fx.m_e.data(), fx.dedup_pulls, fx.scatter);
      check(dp.predicted_total_us == dp_plain.predicted_total_us &&
                dp.selected_r_mask == dp_plain.selected_r_mask &&
                dp.selected_v_head == dp_plain.selected_v_head,
            "online output leaves the pinned decision bit-identical");

      // Mode pinned_adaptive: explicit pins must win even with adapt_bm=1.
      weave::WeaveKnobs pak = pink;
      pak.adapt_bm = 1u;
      pak.dedup_mode = 0u;
      weave::OnlineRoutingPolicy ppa{};
      const auto dpa = weave::cta0_select_policy(
          arch, wl, pak, fx.m_e.data(), fx.dedup_pulls, fx.scatter,
          &ppa, fx.unique_tokens);
      check(weave::policy_is_valid(dpa), "pinned_adaptive decision valid");
      emit_online(arch_name(arch), fx.name, "pinned_adaptive", ppa);
      check(ppa.active_v_head == 64u, "V_HEAD pin wins in adaptive mode");
      check(ppa.active_ready_group_size == 2u, "group pin wins in adaptive mode");
      check(ppa.enable_dedup_runtime == 0u, "DEDUP_MODE=0 pin wins in adaptive mode");
      // The reduce gate still applies to the override-path breakdown (the
      // v_head/group/dedup pins do not enter Eq.(1)-(3), so the gate inputs
      // are identical to the pinned mode's; only adapt_bm flips the gate on).
      check(ppa.reduce_gate_hbm_ok == pp.reduce_gate_hbm_ok &&
                ppa.reduce_gate_tail_ok == pp.reduce_gate_tail_ok &&
                ppa.middle_hbm_util == pp.middle_hbm_util &&
                ppa.tail_combine_frac == pp.tail_combine_frac,
            "explicit v_head/group/dedup pins leave the reduce gate inputs unchanged");
      check(ppa.active_reduce_warps ==
                ((ppa.reduce_gate_hbm_ok != 0u && ppa.reduce_gate_tail_ok != 0u)
                     ? 4u : 0u),
            "pinned_adaptive reduce gate = hbm_ok && tail_ok");
      check(dpa.predicted_total_us == dp.predicted_total_us,
            "pinned_adaptive Eq. total matches pinned");
    }
  }
  std::printf("[PASS] online routing policy: 12 fixtures x 3 hardware x 3 modes, "
              "gate semantics + explicit-pin overrides + Eq. bit-identity\n");
}

// ---------------------------------------------------------------------------
// 3-Tier Tier-1: (Hardware, Precision, Phase, N, K) envelope + per-M inner
// tuple parity grid.  ENV / INNER lines are compared field-by-field (all
// integer) against calibrate.py by tests/test_two_level_adaptive_policy.py.
// ---------------------------------------------------------------------------

struct PnkGeometry {
  uint32_t phase;
  uint32_t n;
  uint32_t k;
};

// The 6 standard models (both phases) + 4 non-standard geometries that stress
// the pruning rules (odd N/K multiples, short K, small N).
std::vector<PnkGeometry> pnk_geometries() {
  return {
      {1u, 4096u, 8192u}, {2u, 8192u, 2048u},   // qwen38max
      {1u, 4096u, 4096u}, {2u, 4096u, 2048u},   // dskv4_flash
      {1u, 6144u, 6144u}, {2u, 6144u, 3072u},   // minimax_m3
      {1u, 1024u, 2048u}, {2u, 2048u, 512u},    // qwen36_35b (Qwen3.5-35B-A3B)
      {1u, 2048u, 3072u}, {2u, 3072u, 1024u},   // qwen35_122b
      {1u, 2048u, 4096u}, {2u, 4096u, 1024u},   // qwen35_397b
      {1u, 1536u, 1536u}, {2u, 1536u, 768u},    // non-standard multiples
      {1u, 512u, 512u},   {2u, 1024u, 256u},    // small N / very short K
  };
}

void test_phase_nk_profiles() {
  static_assert(sizeof(weave::GemmInnerProfile) <= 16u,
                "GemmInnerProfile <= 16 bytes");
  static_assert(std::is_trivially_copyable<weave::GemmInnerProfile>::value,
                "GemmInnerProfile trivially copyable");
  static_assert(std::is_trivially_copyable<weave::PhaseEnvelope>::value,
                "PhaseEnvelope trivially copyable");
  const weave::HardwareArch archs[3] = {
      weave::HardwareArch::B300_SM100, weave::HardwareArch::H20_SM90,
      weave::HardwareArch::MI308X_GFX942};
  const weave::PrecisionMode precs[4] = {
      weave::PrecisionMode::FP8xFP4, weave::PrecisionMode::NVFP4,
      weave::PrecisionMode::FP8xFP8, weave::PrecisionMode::BF16};
  const auto geos = pnk_geometries();
  uint32_t env_lines = 0u, inner_lines = 0u;
  uint32_t distinct_codes = 0u;
  std::vector<uint32_t> seen_codes;
  for (auto arch : archs) {
    for (auto prec : precs) {
      for (const auto& g : geos) {
        const auto env = weave::select_phase_envelope(
            arch, prec, g.phase, g.n, g.k);
        check(env.bn != 0u, "legal envelope exists for every grid geometry");
        check(g.n % env.bn == 0u && g.n % (2u * env.bn) == 0u,
              "BN divides N and the 2-CTA cluster pair");
        check(g.k % env.bk == 0u, "BK divides K");
        check(env.num_stages >= 2u && env.num_stages <= 8u, "stages in 2..8");
        check(env.max_bm * env.bn <=
                  weave::get_pnk_hardware(arch).acc_tile_limit,
              "zero-spill accumulator budget");
        const uint32_t stage_bytes = weave::pnk_stage_bytes(
            env.max_bm, env.bn, env.bk,
            weave::pnk_act_bytes32(prec), weave::pnk_weight_bytes32(prec));
        const auto hw = weave::get_pnk_hardware(arch);
        check(env.num_stages * stage_bytes <= hw.smem_bytes - hw.smem_reserve,
              "stages fit the SMEM/LDS budget");
        if (g.k <= 1024u) check(env.num_stages <= 4u, "short-K stage cap");
        check(env.num_stages <= g.k / env.bk || env.num_stages == 2u,
              "stages bounded by K-loop length");
        // Backend tile legality: the envelope must be a configuration the
        // kernel's own static asserts accept (SM100 BK 128/256, BN 64/128/256,
        // BM%16; SM90 BN<=512 and BK%128; MI308X BM 16/32/64/128).
        bool bn_legal = false, bk_legal = false, bm_legal = false;
        for (uint32_t i = 0u; i < weave::pnk_bn_count(arch); ++i) {
          if (weave::pnk_bn_candidate(arch, i) == env.bn) bn_legal = true;
        }
        for (uint32_t i = 0u; i < weave::pnk_bk_count(arch); ++i) {
          if (weave::pnk_bk_candidate(arch, i) == env.bk) bk_legal = true;
        }
        for (uint32_t i = 0u; i < weave::pnk_bm_count(arch); ++i) {
          if (weave::pnk_bm_candidate(arch, i) == env.max_bm) bm_legal = true;
        }
        check(bn_legal, "BN is a legal backend BLOCK_N");
        check(bk_legal, "BK is a legal backend BLOCK_K");
        check(bm_legal, "max_BM is a legal backend BLOCK_M");
        if (arch == weave::HardwareArch::B300_SM100) {
          check(env.max_bm % 16u == 0u, "SM100 BLOCK_M % 16 == 0");
        } else if (arch == weave::HardwareArch::H20_SM90) {
          check(env.max_bm % 64u == 0u, "SM90 BLOCK_M % 64 == 0 (WGMMA::M)");
          check(env.bk % 128u == 0u, "SM90 BLOCK_K % 128 == 0 (TMA tile)");
        }
        // Round-trip the compact envelope code (lossless for structural fields).
        const uint32_t ecode = weave::encode_phase_envelope(env);
        const auto edec = weave::decode_phase_envelope(ecode);
        check(edec.bn == env.bn && edec.bk == env.bk &&
                  edec.max_bm == env.max_bm &&
                  edec.num_stages == env.num_stages &&
                  edec.l2_swizzle_group == env.l2_swizzle_group &&
                  edec.interleave_mode == env.interleave_mode &&
                  edec.wg_layout_mode == env.wg_layout_mode &&
                  edec.epw == env.epw,
              "PhaseEnvelope code round trip");
        std::printf("ENV arch=%s prec=%u phase=%u n=%u k=%u code=%08x "
                    "bn=%u bk=%u bm=%u st=%u grp=%u mode=%u epw=%u score=%u "
                    "wg=%u pen=%u\n",
                    arch_name(arch), static_cast<uint32_t>(prec), g.phase,
                    g.n, g.k, ecode, env.bn, env.bk, env.max_bm,
                    env.num_stages, env.l2_swizzle_group, env.interleave_mode,
                    env.epw, env.score_ns, env.wg_layout_mode, env.penalty);
        ++env_lines;
        if (std::find(seen_codes.begin(), seen_codes.end(), ecode) ==
            seen_codes.end()) {
          seen_codes.push_back(ecode);
          ++distinct_codes;
        }
        // Per-M inner tuples: GEMM1 has no scatter choice; GEMM2 exercises
        // (dedup, tight-reg) in {(0,0), (0,1), (1,0)}.
        for (uint32_t mi = 0u; mi < 9u; ++mi) {
          const uint32_t m = weave::m_inner_ladder(mi);
          const uint32_t combos = g.phase == 1u ? 1u : 3u;
          for (uint32_t c = 0u; c < combos; ++c) {
            const uint32_t dedup = c == 2u ? 1u : 0u;
            const uint32_t tight = c == 1u ? 1u : 0u;
            const auto p = weave::select_inner_from_envelope(
                arch, prec, g.phase, g.n, g.k, env, m, dedup, tight);
            check(p.block_m <= env.max_bm, "inner BM shares the envelope");
            check(p.block_m % 16u == 0u, "inner BM 16-row aligned");
            if (arch == weave::HardwareArch::H20_SM90) {
              // WGMMA::M = 64; 16/32 are only reachable through kFP8SwapAB.
              check(p.block_m % 64u == 0u || p.block_m == 32u ||
                        p.block_m == 16u,
                    "SM90 inner BM is WGMMA::M-aligned or a swapAB height");
            } else if (arch == weave::HardwareArch::MI308X_GFX942) {
              check(p.block_m == 16u || p.block_m == 32u || p.block_m == 64u ||
                        p.block_m == 128u,
                    "MI308X inner BM is a legal CK tile height");
            }
            check(p.tma_load_m <= p.block_m, "tma_load_m <= block_m");
            check(p.store_block_m <= 64u && p.block_m % p.store_block_m == 0u,
                  "store height divides block_m");
            check(p.epi_reg_stages == 1u || p.epi_reg_stages == 2u,
                  "epi reg stages 1/2");
            check(p.l2_swizzle_group >= 1u && p.l2_swizzle_group <= 8u,
                  "swizzle group 1..8");
            if (g.phase == 1u) check(p.scatter_path == 0u, "GEMM1 no scatter");
            else check(p.scatter_path == (dedup ? 2u : (tight ? 1u : 0u)),
                       "GEMM2 scatter path verdict");
            // L1 L2-swizzle preservation: multi-block M with the weight in
            // half of L2 must keep group >= 2 (never tight mode-5 eviction).
            const uint32_t m_blocks = cdiv(m, p.block_m);
            const uint64_t wbytes = static_cast<uint64_t>(g.n) * g.k *
                weave::pnk_weight_bytes32(prec) / 32ull;
            if (m_blocks >= 2u && wbytes * 2ull <= hw.l2_bytes) {
              check(p.l2_swizzle_group >= 2u,
                    "L1 intra-expert 2D L2 swizzle preserved");
            }
            const uint32_t code = weave::encode_inner_profile(arch, p);
            const auto dec = weave::decode_inner_profile(arch, code);
            check(dec.block_m == p.block_m && dec.tma_load_m == p.tma_load_m &&
                      dec.store_block_m == p.store_block_m &&
                      dec.wg_layout_mode == p.wg_layout_mode &&
                      dec.pipe_mode == p.pipe_mode &&
                      dec.l2_swizzle_group == p.l2_swizzle_group &&
                      dec.epi_reg_stages == p.epi_reg_stages &&
                      dec.scatter_path == p.scatter_path,
                  "GemmInnerProfile code round trip");
            // Public lookup == envelope + inner composition.
            const auto pub = weave::select_phase_inner_profile(
                arch, prec, g.phase, g.n, g.k, m, dedup, tight);
            check(weave::encode_inner_profile(arch, pub) == code,
                  "select_phase_inner_profile composition");
            std::printf("INNER arch=%s prec=%u phase=%u n=%u k=%u m=%u "
                        "dedup=%u tight=%u code=%08x bm=%u tma=%u wg=%u "
                        "pipe=%u grp=%u store=%u epi=%u scat=%u\n",
                        arch_name(arch), static_cast<uint32_t>(prec), g.phase,
                        g.n, g.k, m, dedup, tight, code, p.block_m,
                        p.tma_load_m, p.wg_layout_mode, p.pipe_mode,
                        p.l2_swizzle_group, p.store_block_m, p.epi_reg_stages,
                        p.scatter_path);
            ++inner_lines;
          }
        }
      }
    }
  }
  check(distinct_codes >= 8u,
        "different (N,K) models select genuinely different envelopes");
  std::printf("[PASS] phase-nk profiles: %u ENV + %u INNER lines, "
              "%u distinct envelope codes, legality + L1-swizzle + round trips\n",
              env_lines, inner_lines, distinct_codes);
}

// ---------------------------------------------------------------------------
// 3-Tier Tier-3a: SM0 dedup-aware ordering + hot/cold wave pairing.  ORDER
// lines carry the exact permutation for bit-exact comparison against the
// Python mirror; the C++ side additionally proves span reduction, wave
// variance and the no-pure-cold-wave property on realistic histograms.
// ---------------------------------------------------------------------------

// Frozen zipf(0.9) proxy histograms (calibrate.py zipf_histogram convention;
// transcribed verbatim so no libm pow divergence can enter the fixtures).
static const uint32_t kOrder_zipf64_4096[64] = {
    713, 382, 265, 204, 167, 142, 123, 109, 98, 89, 82, 76, 70, 66, 62, 58,
    55, 52, 50, 48, 46, 44, 42, 40, 39, 38, 36, 35, 34, 33, 32, 31,
    30, 29, 29, 28, 27, 27, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21,
    21, 21, 20, 20, 20, 19, 19, 19, 18, 18, 18, 17, 17, 17, 17, 49,
};
static const uint32_t kOrder_zipf128_32768[128] = {
    4803, 2574, 1787, 1379, 1128, 957, 833, 739, 664, 604, 555, 513, 477, 446, 419, 396,
    375, 356, 339, 324, 310, 297, 285, 275, 265, 255, 247, 239, 231, 224, 218, 212,
    206, 201, 195, 190, 186, 181, 177, 173, 169, 166, 162, 159, 156, 153, 150, 147,
    144, 142, 139, 137, 134, 132, 130, 128, 126, 124, 122, 120, 118, 117, 115, 113,
    112, 110, 109, 107, 106, 104, 103, 102, 101, 99, 98, 97, 96, 95, 94, 93,
    92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 82, 81, 80, 79, 78,
    78, 77, 76, 76, 75, 74, 74, 73, 72, 72, 71, 71, 70, 69, 69, 68,
    68, 67, 67, 66, 66, 65, 65, 64, 64, 63, 63, 62, 62, 61, 61, 118,
};
static const uint32_t kOrder_decode128_320[128] = {
    46, 25, 17, 13, 11, 9, 8, 7, 6, 5, 5, 5, 4, 4, 4, 3,
    3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71,
};

struct OrderFixture {
  const char* name;
  std::vector<uint32_t> counts;
  uint32_t wave;
  uint32_t partner_shift;   // partner[e] = (e + shift) % E
  uint32_t dedup;
  bool expect_variance_reduction;
  bool expect_no_cold_wave;
};

std::vector<OrderFixture> order_fixtures() {
  std::vector<OrderFixture> f;
  f.push_back({"zipf64_4096_w16",
               std::vector<uint32_t>(kOrder_zipf64_4096, kOrder_zipf64_4096 + 64),
               16u, 32u, 1u, true, true});
  f.push_back({"zipf64_4096_w16_nd",
               std::vector<uint32_t>(kOrder_zipf64_4096, kOrder_zipf64_4096 + 64),
               16u, 32u, 0u, true, true});
  f.push_back({"h20_rank0_w78", max_anchored(2482u, 131u, 123u, 128u),
               78u, 64u, 1u, false, false});
  f.push_back({"h20_rank2_w78", max_anchored(2937u, 346u, 122u, 128u),
               78u, 64u, 1u, true, false});
  f.push_back({"zipf128_32768_w16",
               std::vector<uint32_t>(kOrder_zipf128_32768, kOrder_zipf128_32768 + 128),
               16u, 64u, 1u, true, true});
  f.push_back({"decode128_w78",
               std::vector<uint32_t>(kOrder_decode128_320, kOrder_decode128_320 + 128),
               78u, 64u, 1u, false, false});
  {
    std::vector<uint32_t> c(128u, 5u);
    for (uint32_t i = 0u; i < 10u; ++i) c[i] = 100u;
    f.push_back({"ten_hot_w78", c, 78u, 64u, 0u, true, true});
  }
  f.push_back({"empty_128", std::vector<uint32_t>(128u, 0u), 78u, 64u, 1u,
               false, false});
  {
    std::vector<uint32_t> c(256u, 0u); c[0] = 65535u;
    f.push_back({"single_256", c, 16u, 128u, 1u, false, false});
  }
  f.push_back({"dense256_w16", std::vector<uint32_t>(256u, 128u), 16u, 128u, 1u,
               false, false});
  f.push_back({"uniform64_w8", std::vector<uint32_t>(64u, 32u), 8u, 32u, 1u,
               false, false});
  {
    std::vector<uint32_t> c(128u, 8u); c[0] = 4096u;
    f.push_back({"skew_hot_1expert", c, 16u, 64u, 1u, false, false});
  }
  // Wave size 1 with 2-member mutual clusters: exercises the capacity-
  // overflow path (cluster larger than every wave's remaining capacity).
  f.push_back({"tiny_waves_w1", std::vector<uint32_t>{50u, 3u, 80u, 2u},
               1u, 2u, 1u, false, false});
  return f;
}

uint64_t order_span(const std::vector<uint32_t>& order,
                    const std::vector<uint32_t>& counts,
                    const std::vector<uint32_t>& partner) {
  const uint32_t e_count = static_cast<uint32_t>(order.size());
  std::vector<uint32_t> pos(e_count);
  for (uint32_t i = 0u; i < e_count; ++i) pos[order[i]] = i;
  uint64_t span = 0u;
  for (uint32_t e = 0u; e < e_count; ++e) {
    const uint32_t p = partner[e];
    if (counts[e] > 0u && counts[p] > 0u && p != e) {
      const uint32_t a = pos[e], b = pos[p];
      span += a > b ? a - b : b - a;
    }
  }
  return span;   // each mutual pair counted twice (symmetric, both sides)
}

double order_wave_variance(const std::vector<uint32_t>& order,
                           const std::vector<uint32_t>& counts,
                           uint32_t wave) {
  std::vector<double> sums;
  for (uint32_t i = 0u; i < order.size(); i += wave) {
    double s = 0.0;
    for (uint32_t j = i; j < std::min<uint32_t>(i + wave, order.size()); ++j) {
      s += counts[order[j]];
    }
    sums.push_back(s);
  }
  if (sums.empty()) return 0.0;
  const double mean = std::accumulate(sums.begin(), sums.end(), 0.0) / sums.size();
  double var = 0.0;
  for (double s : sums) var += (s - mean) * (s - mean);
  return var / sums.size();
}

void test_sm0_expert_order() {
  auto fixtures = order_fixtures();
  std::vector<uint32_t> order(weave::kMaxRuntimeExperts);
  std::vector<uint32_t> multi(weave::kMaxRuntimeExperts);
  std::vector<uint32_t> partner(weave::kMaxRuntimeExperts);
  auto* scratch = new weave::Sm0OrderScratch();
  double worst_us = 0.0;
  for (const auto& fx : fixtures) {
    const uint32_t e_count = static_cast<uint32_t>(fx.counts.size());
    for (uint32_t e = 0u; e < e_count; ++e) {
      multi[e] = fx.counts[e] / 2u;
      partner[e] = (e + fx.partner_shift) % e_count;
    }
    uint32_t mode = 0xffffffffu;
    const uint32_t idx = weave::build_sm0_expert_order(
        fx.counts.data(), multi.data(), partner.data(), e_count, fx.wave,
        fx.dedup, order.data(), scratch, &mode);
    check(idx == e_count, "order covers every expert");
    check(mode <= weave::kSm0ModeNoPair, "order mode verdict in range");
    // The hot/cold pairing boundary is explicit: interleaving is only legal
    // in kSm0ModeHotCold (dedup inactive AND experts_per_wave >= 2).
    if (fx.dedup == 0u) {
      check(mode == (fx.wave >= 2u ? weave::kSm0ModeHotCold
                                   : weave::kSm0ModeNoPair),
            "non-dedup mode follows the experts_per_wave >= 2 boundary");
    }
    std::vector<uint32_t> seen(e_count, 0u);
    for (uint32_t i = 0u; i < idx; ++i) {
      check(order[i] < e_count, "order entry in range");
      ++seen[order[i]];
    }
    bool perm = true;
    for (uint32_t e = 0u; e < e_count; ++e) perm = perm && seen[e] == 1u;
    check(perm, "order is a permutation");

    // Determinism: a second call reproduces the permutation bit-for-bit.
    std::vector<uint32_t> order2(e_count);
    weave::build_sm0_expert_order(fx.counts.data(), multi.data(), partner.data(),
                                  e_count, fx.wave, fx.dedup, order2.data(),
                                  scratch);
    check(order2 == std::vector<uint32_t>(order.begin(), order.begin() + e_count),
          "order deterministic");

    // Host latency proxy for the < 2 us device budget (one CTA-0 thread).
    // Min of 5 x 1000-rep batches: robust against turbo/throttle interference
    // on shared laptops.  The device budget itself is evidenced by the
    // clock64() measurement in the GPU rail (weave_sm0_order_device_smoke).
    double us = 1e30;
    for (uint32_t batch = 0u; batch < 5u; ++batch) {
      const auto t0 = std::chrono::steady_clock::now();
      for (uint32_t r = 0u; r < 1000u; ++r) {
        weave::build_sm0_expert_order(fx.counts.data(), multi.data(),
                                      partner.data(), e_count, fx.wave,
                                      fx.dedup, order2.data(), scratch);
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double b =
          std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000u;
      us = std::min(us, b);
    }
    worst_us = std::max(worst_us, us);

    const uint32_t active = static_cast<uint32_t>(
        std::count_if(fx.counts.begin(), fx.counts.end(),
                      [](uint32_t c) { return c > 0u; }));
    uint32_t total = 0u;
    for (uint32_t c : fx.counts) total += c;
    const uint32_t hot_thresh =
        active > 0u ? std::max(13u, 2u * (total / active)) : 13u;

    // Dedup span: mutual multi-hit pairs must collapse vs identity order.
    if (fx.dedup != 0u && active > 1u) {
      std::vector<uint32_t> identity(e_count);
      std::iota(identity.begin(), identity.end(), 0u);
      const uint64_t base_span = order_span(identity, fx.counts, partner);
      const uint64_t new_span =
          order_span(std::vector<uint32_t>(order.begin(), order.begin() + e_count),
                     fx.counts, partner);
      if (base_span > 0u) {
        check(new_span * 100u <= base_span * 70u,
              "dedup completion span reduced >= 30%");
      }
      // Cluster adjacency: every mutual active pair sits in adjacent slots.
      std::vector<uint32_t> pos(e_count);
      for (uint32_t i = 0u; i < e_count; ++i) pos[order[i]] = i;
      for (uint32_t e = 0u; e < e_count; ++e) {
        const uint32_t p = partner[e];
        if (multi[e] > 0u && fx.counts[e] > 0u && fx.counts[p] > 0u &&
            partner[p] == e) {
          const uint32_t d = pos[e] > pos[p] ? pos[e] - pos[p] : pos[p] - pos[e];
          check(d == 1u, "multi-hit partner cluster adjacent");
        }
      }
    }

    // Hot/cold wave balance vs descending packing.
    std::vector<uint32_t> desc(e_count);
    std::iota(desc.begin(), desc.end(), 0u);
    std::stable_sort(desc.begin(), desc.end(), [&](uint32_t a, uint32_t b) {
      return fx.counts[a] > fx.counts[b];
    });
    const double var_desc = order_wave_variance(desc, fx.counts, fx.wave);
    const double var_new = order_wave_variance(
        std::vector<uint32_t>(order.begin(), order.begin() + e_count),
        fx.counts, fx.wave);
    check(var_new <= var_desc + 1e-9, "wave M variance never grows");
    if (fx.expect_variance_reduction) {
      check(var_new < var_desc, "wave M variance strictly lower");
    }
    if (fx.expect_no_cold_wave && active > 0u) {
      const uint32_t num_waves = (active + fx.wave - 1u) / fx.wave;
      uint32_t hot_experts = 0u;
      for (uint32_t c : fx.counts) if (c >= hot_thresh) ++hot_experts;
      if (hot_experts >= num_waves) {
        uint32_t cold_waves = 0u;
        for (uint32_t i = 0u; i < e_count; i += fx.wave) {
          bool any_hot = false;
          for (uint32_t j = i; j < std::min<uint32_t>(i + fx.wave, e_count); ++j) {
            if (fx.counts[order[j]] >= hot_thresh) any_hot = true;
          }
          if (!any_hot) ++cold_waves;
        }
        check(cold_waves == 0u, "no wave packed purely with cold experts");
      }
    }

    std::printf("ORDER fix=%s dedup=%u wave=%u E=%u us=%.3f mode=%u order=",
                fx.name, fx.dedup, fx.wave, e_count, us, mode);
    for (uint32_t i = 0u; i < e_count; ++i) {
      std::printf("%s%u", i == 0u ? "" : ",", order[i]);
    }
    std::printf("\n");
  }
  delete scratch;
  check(worst_us < 10.0, "host order latency proxy inside budget");
  std::printf("[PASS] sm0 expert order: %zu fixtures, permutation + determinism "
              "+ span >= 30%% + wave variance + hot/cold balance "
              "(worst host %.3f us/call)\n", fixtures.size(), worst_us);
}

// ---------------------------------------------------------------------------
// 3-Tier Tier-3b: Sm0ReadyTokenQueue lock-free correctness (host threads).
// ---------------------------------------------------------------------------

void test_sm0_ready_queue() {
  using weave::Sm0ReadyCell;
  using weave::Sm0ReadyTokenQueue;
  // POD / layout contract.
  static_assert(sizeof(Sm0ReadyTokenQueue) == 64u, "queue header one cache line");
  static_assert(std::is_trivially_copyable<Sm0ReadyCell>::value, "cell POD");

  // Single-threaded FIFO + epoch semantics.
  {
    Sm0ReadyTokenQueue q{};
    std::vector<Sm0ReadyCell> cells(8u, Sm0ReadyCell{0u, 0u});
    weave::rtq_init(&q, 8u, 1u);
    uint32_t tok = 0u;
    check(!weave::rtq_pop(&q, cells.data(), &tok), "empty queue pops nothing");
    for (uint32_t i = 0u; i < 5u; ++i) {
      check(weave::rtq_push(&q, cells.data(), 100u + i), "push ok");
    }
    for (uint32_t i = 0u; i < 5u; ++i) {
      check(weave::rtq_pop(&q, cells.data(), &tok) && tok == 100u + i,
            "FIFO pop order");
    }
    check(!weave::rtq_pop(&q, cells.data(), &tok), "drained queue empty");
    // Epoch bump: stale cells must never pop.
    weave::rtq_begin_epoch(&q, 2u);
    check(!weave::rtq_pop(&q, cells.data(), &tok), "stale epoch not poppable");
    check(weave::rtq_push(&q, cells.data(), 7u), "new epoch push");
    check(weave::rtq_pop(&q, cells.data(), &tok) && tok == 7u, "new epoch pop");
    // Overflow: capacity 8, push 10 -> 8 ok, 2 rejected, sticky flag.
    weave::rtq_begin_epoch(&q, 3u);
    uint32_t ok = 0u;
    for (uint32_t i = 0u; i < 10u; ++i) {
      if (weave::rtq_push(&q, cells.data(), i)) ++ok;
    }
    check(ok == 8u && q.overflow == 1u, "overflow bounded + sticky");
    uint32_t popped = 0u;
    while (weave::rtq_pop(&q, cells.data(), &tok)) {
      check(tok == popped, "overflow epoch FIFO intact");
      ++popped;
    }
    check(popped == 8u, "exactly capacity tokens poppable after overflow");
  }

  // Row encoding.
  check(weave::rtq_encode_row(3u, 0x123456u) == 0x03123456u, "row encode");
  check(weave::rtq_decode_rank(0x03123456u) == 3u, "row rank decode");
  check(weave::rtq_decode_token(0x03123456u) == 0x123456u, "row token decode");

  // Sender-side local pre-reduce: only the LAST hit pushes the row.
  {
    Sm0ReadyTokenQueue q{};
    std::vector<Sm0ReadyCell> cells(4u, Sm0ReadyCell{0u, 0u});
    weave::rtq_init(&q, 4u, 1u);
    uint32_t remaining = 3u;
    check(!weave::rtq_notify_local_prereduce(&q, cells.data(), &remaining, 2u, 77u),
          "hit 1/3 not ready");
    check(!weave::rtq_notify_local_prereduce(&q, cells.data(), &remaining, 2u, 77u),
          "hit 2/3 not ready");
    check(weave::rtq_notify_local_prereduce(&q, cells.data(), &remaining, 2u, 77u),
          "hit 3/3 ready");
    uint32_t row = 0u;
    check(weave::rtq_pop(&q, cells.data(), &row) &&
              weave::rtq_decode_rank(row) == 2u &&
              weave::rtq_decode_token(row) == 77u,
          "pre-reduce row popped ready");
    check(!weave::rtq_pop(&q, cells.data(), &row), "one row per token");
  }

  // Receiver-side top-k combine: only the expected-th arrival pushes.
  {
    Sm0ReadyTokenQueue q{};
    std::vector<Sm0ReadyCell> cells(4u, Sm0ReadyCell{0u, 0u});
    weave::rtq_init(&q, 4u, 1u);
    uint32_t arrived = 0u;
    for (uint32_t i = 0u; i < 3u; ++i) {
      check(!weave::rtq_notify_remote_arrival(&q, cells.data(), &arrived, 4u, 9u),
            "arrival < top-k not ready");
    }
    check(weave::rtq_notify_remote_arrival(&q, cells.data(), &arrived, 4u, 9u),
          "top-k arrival ready");
    uint32_t tok = 0u;
    check(weave::rtq_pop(&q, cells.data(), &tok) && tok == 9u,
          "combine token popped ready");
  }

  // MPSC lock-free stress: 4 producer threads x 2500 pushes, one concurrent
  // consumer; every token popped exactly once, in per-producer FIFO order.
  {
    constexpr uint32_t kProducers = 4u;
    constexpr uint32_t kPerProducer = 2500u;
    constexpr uint32_t kCapacity = kProducers * kPerProducer;
    Sm0ReadyTokenQueue q{};
    std::vector<Sm0ReadyCell> cells(kCapacity, Sm0ReadyCell{0u, 0u});
    weave::rtq_init(&q, kCapacity, 1u);
    std::atomic<uint32_t> consumed{0u};
    std::vector<uint32_t> popped;
    popped.reserve(kCapacity);
    std::atomic<bool> producers_done{false};
    std::thread consumer([&]() {
      uint32_t tok = 0u;
      while (consumed.load(std::memory_order_relaxed) < kCapacity) {
        if (weave::rtq_pop(&q, cells.data(), &tok)) {
          popped.push_back(tok);
          consumed.fetch_add(1u, std::memory_order_relaxed);
        } else if (producers_done.load(std::memory_order_acquire) &&
                   q.head == q.tail) {
          break;
        }
      }
    });
    std::vector<std::thread> producers;
    for (uint32_t p = 0u; p < kProducers; ++p) {
      producers.emplace_back([&, p]() {
        for (uint32_t i = 0u; i < kPerProducer; ++i) {
          const uint32_t token = (p << 24) | i;
          while (!weave::rtq_push(&q, cells.data(), token)) {
            std::this_thread::yield();
          }
        }
      });
    }
    for (auto& t : producers) t.join();
    producers_done.store(true, std::memory_order_release);
    consumer.join();
    check(popped.size() == kCapacity, "MPSC: every token popped");
    check(q.overflow == 0u, "MPSC: no overflow");
    std::vector<uint32_t> sorted = popped;
    std::sort(sorted.begin(), sorted.end());
    bool unique = true;
    for (uint32_t i = 1u; i < sorted.size(); ++i) {
      if (sorted[i] == sorted[i - 1u]) unique = false;
    }
    check(unique, "MPSC: no duplicates");
    // Per-producer FIFO: tokens of each producer appear in push order.
    std::vector<uint32_t> last(kProducers, 0u);
    std::vector<bool> started(kProducers, false);
    bool fifo = true;
    for (uint32_t tok : popped) {
      const uint32_t p = tok >> 24, i = tok & 0xffffffu;
      if (started[p] && i != last[p] + 1u) fifo = false;
      started[p] = true;
      last[p] = i;
    }
    check(fifo, "MPSC: per-producer FIFO preserved");
  }
  std::printf("[PASS] sm0 ready queue: FIFO + epoch + overflow + pre-reduce / "
              "top-k notify + 4-producer MPSC stress (10000 tokens)\n");
}

// ---------------------------------------------------------------------------
// 3-Tier online publication: per-expert l1/l2 profile codes (LUT lines for
// the auto mode; pinned modes must publish zeros).
// ---------------------------------------------------------------------------

void test_online_profile_lut() {
  const weave::HardwareArch archs[3] = {
      weave::HardwareArch::B300_SM100, weave::HardwareArch::H20_SM90,
      weave::HardwareArch::MI308X_GFX942};
  const auto fixtures = online_fixtures();
  uint32_t lines = 0u;
  for (auto arch : archs) {
    const uint32_t tight = arch == weave::HardwareArch::H20_SM90 ? 1u : 0u;
    for (const auto& fx : fixtures) {
      const weave::WorkloadSpec wl{fx.experts, fx.hidden, fx.inter};
      weave::WeaveKnobs knobs = weave::default_weave_knobs(arch);
      knobs.enable_runtime_cost_model = 1u;
      knobs.override_mask = 0u;
      knobs.reduce_warps = 4u;
      knobs.adapt_bm = 1u;
      knobs.ready_group_size = 0u;
      knobs.dedup_mode = weave::kAutoU32;
      weave::OnlineRoutingPolicy p{};
      (void)weave::cta0_select_policy(arch, wl, knobs, fx.m_e.data(),
                                      fx.dedup_pulls, fx.scatter, &p,
                                      fx.unique_tokens);
      check(p.phase_profile_active == 1u, "adaptive publishes profile LUT");
      const uint32_t n1 = 2u * fx.inter, k1 = fx.hidden;
      const uint32_t n2 = fx.hidden, k2 = fx.inter;
      const auto env1 = weave::select_phase_envelope(
          arch, weave::PrecisionMode::FP8xFP4, 1u, n1, k1);
      const auto env2 = weave::select_phase_envelope(
          arch, weave::PrecisionMode::FP8xFP4, 2u, n2, k2);
      check(p.l1_envelope_code == weave::encode_phase_envelope(env1),
            "published GEMM1 envelope code");
      check(p.l2_envelope_code == weave::encode_phase_envelope(env2),
            "published GEMM2 envelope code");
      for (uint32_t e = 0u; e < fx.experts; ++e) {
        const uint32_t m = fx.m_e[e];
        const uint32_t expect1 = m == 0u ? 0u :
            weave::encode_inner_profile(arch, weave::select_phase_inner_profile(
                arch, weave::PrecisionMode::FP8xFP4, 1u, n1, k1,
                weave::m_inner_bucket(m), p.enable_dedup_runtime, tight));
        const uint32_t expect2 = m == 0u ? 0u :
            weave::encode_inner_profile(arch, weave::select_phase_inner_profile(
                arch, weave::PrecisionMode::FP8xFP4, 2u, n2, k2,
                weave::m_inner_bucket(m), p.enable_dedup_runtime, tight));
        check(p.l1_profile_code[e] == expect1, "per-expert GEMM1 code");
        check(p.l2_profile_code[e] == expect2, "per-expert GEMM2 code");
      }
      // Pinned / non-adaptive: zeros, compiled structure untouched.
      weave::WeaveKnobs pin = knobs;
      pin.adapt_bm = 0u;
      weave::OnlineRoutingPolicy pp{};
      (void)weave::cta0_select_policy(arch, wl, pin, fx.m_e.data(),
                                      fx.dedup_pulls, fx.scatter, &pp,
                                      fx.unique_tokens);
      check(pp.phase_profile_active == 0u && pp.l1_envelope_code == 0u &&
                pp.l2_envelope_code == 0u,
            "non-adaptive publishes no profiles");
      bool zero = true;
      for (uint32_t e = 0u; e < weave::kMaxRuntimeExperts; ++e) {
        zero = zero && pp.l1_profile_code[e] == 0u && pp.l2_profile_code[e] == 0u;
      }
      check(zero, "non-adaptive LUT all zero");

      std::printf("LUT arch=%s fix=%s active=%u l1env=%08x l2env=%08x l1=",
                  arch_name(arch), fx.name, p.phase_profile_active,
                  p.l1_envelope_code, p.l2_envelope_code);
      for (uint32_t e = 0u; e < fx.experts; ++e) {
        std::printf("%s%08x", e == 0u ? "" : ",", p.l1_profile_code[e]);
      }
      std::printf(" l2=");
      for (uint32_t e = 0u; e < fx.experts; ++e) {
        std::printf("%s%08x", e == 0u ? "" : ",", p.l2_profile_code[e]);
      }
      std::printf("\n");
      ++lines;
    }
  }
  std::printf("[PASS] online profile LUT: %u publications, per-expert codes + "
              "envelope codes + pinned-zero semantics\n", lines);
}

// ---------------------------------------------------------------------------
// Round-2 synthesis: SM0 mode verdicts (pendulum / no-pair) + the warp path's
// bit-identity contract, expressed on the host through the shared passes.
// ---------------------------------------------------------------------------

void test_sm0_order_modes() {
  auto* scratch = new weave::Sm0OrderScratch();
  std::vector<uint32_t> order(weave::kMaxRuntimeExperts);
  uint32_t lines = 0u;

  // Pendulum fixtures: per-expert multi-hit weights are known but NO partner
  // edges exist (partner = E, "none").  The metric's hit sets are real
  // multi-expert sets, so the completion span is measurable even though the
  // ordering algorithm only sees the per-expert counters.
  struct PendFx {
    const char* name;
    std::vector<uint32_t> counts;
    std::vector<uint32_t> hits;
    uint32_t wave;
  };
  std::vector<PendFx> pend;
  {
    // Zipf-ish loads; multi-hit weight concentrated on a few experts.
    for (uint32_t E : {32u, 64u, 128u}) {
      PendFx fx{"pend", std::vector<uint32_t>(E), std::vector<uint32_t>(E), 16u};
      for (uint32_t e = 0u; e < E; ++e) {
        fx.counts[e] = 400u / (e + 1u) + 4u;
        // Every 3rd expert carries multi-hit tokens; weight decays with id.
        fx.hits[e] = (e % 3u == 0u) ? (E - e) : 0u;
      }
      fx.name = E == 32u ? "pend32" : (E == 64u ? "pend64" : "pend128");
      pend.push_back(fx);
    }
  }
  for (const auto& fx : pend) {
    const uint32_t E = static_cast<uint32_t>(fx.counts.size());
    const std::vector<uint32_t> none(E, E);   // no co-occurrence edges
    uint32_t mode = 0xffffffffu;
    const uint32_t idx = weave::build_sm0_expert_order(
        fx.counts.data(), fx.hits.data(), none.data(), E, fx.wave, 1u,
        order.data(), scratch, &mode);
    check(idx == E, "pendulum order covers every expert");
    check(mode == weave::kSm0ModePendulum,
          "multi-hit-only dedup selects the pendulum mode");
    std::vector<uint32_t> seen(E, 0u);
    for (uint32_t i = 0u; i < E; ++i) ++seen[order[i]];
    bool perm = true;
    for (uint32_t e = 0u; e < E; ++e) perm = perm && seen[e] == 1u;
    check(perm, "pendulum order is a permutation");
    // The most-shared expert must sit at the centre slot.
    uint32_t best = 0u;
    for (uint32_t e = 1u; e < E; ++e) {
      if (fx.hits[e] > fx.hits[best] ||
          (fx.hits[e] == fx.hits[best] && fx.counts[e] > fx.counts[best])) {
        best = e;
      }
    }
    const uint32_t active = E;   // every expert is loaded in these fixtures
    std::vector<uint32_t> pos(E);
    for (uint32_t i = 0u; i < E; ++i) pos[order[i]] = i;
    check(pos[best] == (active - 1u) / 2u, "most-shared expert is centred");

    // Completion span over real multi-expert hit sets (each token hits the
    // multi-hit experts of one residue class): pendulum vs identity order.
    std::vector<std::vector<uint32_t>> hit_sets;
    for (uint32_t t = 0u; t < 24u; ++t) {
      std::vector<uint32_t> hs;
      for (uint32_t e = t % 3u; e < E; e += 3u) {
        if (fx.hits[e] > 0u && (e / 3u) % 4u == t % 4u) hs.push_back(e);
      }
      if (hs.size() >= 2u) hit_sets.push_back(hs);
    }
    const auto span_of = [&](const std::vector<uint32_t>& o) {
      std::vector<uint32_t> p(E);
      for (uint32_t i = 0u; i < E; ++i) p[o[i]] = i;
      uint64_t span = 0u;
      for (const auto& hs : hit_sets) {
        uint32_t lo = E, hi = 0u;
        for (uint32_t e : hs) { lo = std::min(lo, p[e]); hi = std::max(hi, p[e]); }
        span += hi - lo;
      }
      return span;
    };
    std::vector<uint32_t> identity(E);
    std::iota(identity.begin(), identity.end(), 0u);
    const uint64_t base = span_of(identity);
    const uint64_t now = span_of(std::vector<uint32_t>(order.begin(),
                                                       order.begin() + E));
    check(base > 0u && now < base, "pendulum shortens the completion span");
    std::printf("PEND fix=%s E=%u mode=%u base=%llu new=%llu order=",
                fx.name, E, mode, static_cast<unsigned long long>(base),
                static_cast<unsigned long long>(now));
    for (uint32_t i = 0u; i < E; ++i) {
      std::printf("%s%u", i == 0u ? "" : ",", order[i]);
    }
    std::printf("\n");
    ++lines;
  }

  // W = 1 boundary: one expert fills the whole card -> pairing auto-exits.
  {
    const uint32_t E = 16u;
    std::vector<uint32_t> counts(E), hits(E, 0u), none(E, E);
    for (uint32_t e = 0u; e < E; ++e) counts[e] = 100u - 3u * e;
    uint32_t mode = 0xffffffffu;
    weave::build_sm0_expert_order(counts.data(), hits.data(), none.data(), E,
                                  1u, 0u, order.data(), scratch, &mode);
    check(mode == weave::kSm0ModeNoPair, "W=1 auto-exits hot/cold pairing");
    for (uint32_t i = 0u; i + 1u < E; ++i) {
      check(counts[order[i]] >= counts[order[i + 1u]],
            "no-pair mode is plain load-descending");
    }
    std::printf("PEND fix=w1_nopair E=%u mode=%u base=0 new=0 order=", E, mode);
    for (uint32_t i = 0u; i < E; ++i) {
      std::printf("%s%u", i == 0u ? "" : ",", order[i]);
    }
    std::printf("\n");
    ++lines;
  }
  delete scratch;
  std::printf("[PASS] sm0 order modes: %u pendulum / no-pair fixtures, "
              "centre placement + span reduction + W=1 boundary\n", lines);
}

// ---------------------------------------------------------------------------
// Round-2 synthesis: lock-free local pre-reduce.  Host semantics + a threaded
// stress proving EXACTLY ONE last contributor and a complete sum.
// ---------------------------------------------------------------------------

void test_lockfree_prereduce() {
  static_assert(std::is_trivially_copyable<weave::ReadyRingSlot>::value,
                "ReadyRingSlot POD");
  // Single contributor: accumulate then read back.
  {
    alignas(16) float scratch[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    alignas(16) const float frag[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t done = 0u;
    check(weave::lockfree_local_prereduce_accumulate(scratch, frag, 8u, &done,
                                                     1u),
          "single contributor is the last one");
    alignas(16) float out[8] = {};
    weave::lockfree_local_prereduce_load(scratch, 8u, out);
    for (uint32_t i = 0u; i < 8u; ++i) check(out[i] == frag[i], "sum complete");
    check(done == 1u, "done counter advanced once");
  }
  // Unaligned / non-multiple-of-4 element count takes the scalar tail path.
  {
    alignas(16) float buf[10] = {};
    float* scratch = buf + 1;      // deliberately misaligned
    const float frag[5] = {1, 1, 1, 1, 1};
    uint32_t done = 0u;
    check(weave::lockfree_local_prereduce_accumulate(scratch, frag, 5u, &done,
                                                     1u),
          "scalar tail path last contributor");
    for (uint32_t i = 0u; i < 5u; ++i) check(scratch[i] == 1.0f, "scalar sum");
  }
  // Invalid inputs never touch memory.
  {
    uint32_t done = 0u;
    float v = 0.0f;
    check(!weave::lockfree_local_prereduce_accumulate(nullptr, &v, 1u, &done, 1u),
          "null scratch rejected");
    check(!weave::lockfree_local_prereduce_accumulate(&v, &v, 1u, &done, 0u),
          "zero expected contributors rejected");
    check(done == 0u, "rejected calls leave the counter untouched");
  }
  // Threaded stress: 8 contributors x 2000 rounds, exactly one last per round.
  {
    const uint32_t kContrib = 8u, kRounds = 2000u, kElems = 16u;
    std::vector<std::atomic<uint32_t>> last_counts(kRounds);
    for (auto& a : last_counts) a.store(0u);
    std::vector<std::atomic<uint32_t>> done(kRounds);
    for (auto& a : done) a.store(0u);
    std::vector<float> accum(static_cast<size_t>(kRounds) * kElems, 0.0f);
    std::vector<std::thread> workers;
    for (uint32_t c = 0u; c < kContrib; ++c) {
      workers.emplace_back([&, c]() {
        alignas(16) float frag[kElems];
        for (uint32_t i = 0u; i < kElems; ++i) frag[i] = 1.0f;
        for (uint32_t r = 0u; r < kRounds; ++r) {
          float* sc = accum.data() + static_cast<size_t>(r) * kElems;
          uint32_t* dc = reinterpret_cast<uint32_t*>(&done[r]);
          if (weave::lockfree_local_prereduce_accumulate(sc, frag, kElems, dc,
                                                         kContrib)) {
            last_counts[r].fetch_add(1u);
          }
        }
      });
    }
    for (auto& t : workers) t.join();
    uint32_t bad_last = 0u;
    for (uint32_t r = 0u; r < kRounds; ++r) {
      if (last_counts[r].load() != 1u) ++bad_last;
    }
    check(bad_last == 0u, "exactly one last contributor per round");
    std::printf("PRERED rounds=%u contributors=%u elems=%u bad_last=%u\n",
                kRounds, kContrib, kElems, bad_last);
  }
  std::printf("[PASS] lock-free pre-reduce: vector + scalar paths, invalid "
              "input guards, exactly-once last contributor under 8 threads\n");
}

// ---------------------------------------------------------------------------
// Round-2 synthesis: sharded ready-token rings, all three modes.  The critical
// regression is EXACTLY-ONCE consumption when several SMs pop concurrently.
// ---------------------------------------------------------------------------

void test_sharded_ready_ring() {
  using weave::ShardedReadyTokenRing;
  static_assert(sizeof(weave::ReadyRingShard) == 2112u,
                "shard is ~2 KiB (64 B header + 256 x 8 B slots)");
  static_assert(std::is_trivially_copyable<weave::ReadyRingShard>::value,
                "shard POD");
  for (uint32_t d : {1u, 2u, 4u, 8u}) {
    check(weave::ready_ring_shard_count_ok(d), "D in {1,2,4,8}");
  }
  check(!weave::ready_ring_shard_count_ok(3u), "D=3 rejected");

  auto* rings = new ShardedReadyTokenRing();
  uint32_t token = 0u;

  // Mode 0: cluster-exclusive SPSC.  Owner-only, FIFO, wrap-around, and a
  // non-owner pop is REJECTED (no cross-SM tail competition is possible).
  weave::ready_ring_init(rings, 4u, weave::kRingModeClusterSpsc);
  check(weave::ready_ring_push(rings, 0u, 0u, 11u), "spsc owner push");
  check(!weave::ready_ring_push(rings, 0u, 1u, 12u), "spsc non-owner push denied");
  check(!weave::ready_ring_pop_preferred_or_steal(rings, 1u, &token),
        "spsc non-owner pop denied");
  check(weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token) &&
            token == 11u, "spsc owner pop");
  check(!weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token),
        "spsc empty pop");
  for (uint32_t round = 0u; round < 3u; ++round) {
    for (uint32_t i = 0u; i < weave::kReadyRingSlots; ++i) {
      check(weave::ready_ring_push(rings, 1u, 1u, round * 1000u + i),
            "spsc fill push");
    }
    check(!weave::ready_ring_push(rings, 1u, 1u, 999999u), "spsc full rejected");
    for (uint32_t i = 0u; i < weave::kReadyRingSlots; ++i) {
      check(weave::ready_ring_pop_preferred_or_steal(rings, 1u, &token) &&
                token == round * 1000u + i, "spsc FIFO across wrap");
    }
  }

  // Mode 1: no-CAS MPSC epoch enqueue, single owner consumer, sticky overflow.
  weave::ready_ring_init(rings, 2u, weave::kRingModeMpscEpoch);
  weave::ready_ring_begin_epoch(rings, 7u);
  for (uint32_t i = 0u; i < weave::kReadyRingSlots; ++i) {
    check(weave::ready_ring_push(rings, 0u, 5u, i), "mpsc push");
  }
  check(!weave::ready_ring_push(rings, 0u, 5u, 999u), "mpsc overflow rejected");
  check(rings->shards[0].overflow == 1u, "mpsc overflow is sticky");
  for (uint32_t i = 0u; i < weave::kReadyRingSlots; ++i) {
    check(weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token) &&
              token == i, "mpsc FIFO");
  }
  check(!weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token),
        "mpsc drained");
  // A fresh epoch clears the cursors and the sticky overflow.
  weave::ready_ring_begin_epoch(rings, 8u);
  check(rings->shards[0].overflow == 0u, "new epoch clears overflow");
  check(weave::ready_ring_push(rings, 0u, 5u, 42u), "push after new epoch");
  check(weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token) &&
            token == 42u, "pop after new epoch");

  // Mode 2 single-threaded: unified claim, own shard first then one steal hop.
  weave::ready_ring_init(rings, 2u, weave::kRingModeMpmcSteal);
  check(weave::ready_ring_claim_push(&rings->shards[1], 77u), "claim push shard1");
  check(weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token) &&
            token == 77u, "steal from the neighbour shard when home is empty");
  check(!weave::ready_ring_pop_preferred_or_steal(rings, 0u, &token),
        "both shards empty");

  // Mode 2 THREADED exactly-once stress: 4 producer threads push distinct
  // tokens into 2 shards while 4 consumer threads pop (home + steal).  Every
  // token must be consumed exactly once -- this is the regression that a
  // "plain store tail+1 + neighbour CAS steal" design fails.
  {
    const uint32_t kProducers = 4u, kPerProducer = 2000u, kConsumers = 4u;
    const uint32_t kTotal = kProducers * kPerProducer;
    weave::ready_ring_init(rings, 2u, weave::kRingModeMpmcSteal);
    std::vector<std::atomic<uint32_t>> seen(kTotal);
    for (auto& a : seen) a.store(0u);
    std::atomic<uint32_t> pushed{0u}, popped{0u};
    std::vector<std::thread> workers;
    for (uint32_t p = 0u; p < kProducers; ++p) {
      workers.emplace_back([&, p]() {
        for (uint32_t i = 0u; i < kPerProducer; ++i) {
          const uint32_t tok = p * kPerProducer + i;
          const uint32_t d = tok % 2u;
          while (!weave::ready_ring_claim_push(&rings->shards[d], tok)) {
            std::this_thread::yield();   // ring full: drain and retry
          }
          pushed.fetch_add(1u);
        }
      });
    }
    for (uint32_t c = 0u; c < kConsumers; ++c) {
      workers.emplace_back([&, c]() {
        uint32_t got = 0u;
        while (popped.load() < kTotal) {
          if (weave::ready_ring_pop_preferred_or_steal(rings, c, &got)) {
            seen[got].fetch_add(1u);
            popped.fetch_add(1u);
          } else {
            std::this_thread::yield();
          }
        }
      });
    }
    for (auto& t : workers) t.join();
    uint32_t dup = 0u, missing = 0u;
    for (uint32_t i = 0u; i < kTotal; ++i) {
      const uint32_t n = seen[i].load();
      if (n > 1u) ++dup;
      if (n == 0u) ++missing;
    }
    check(pushed.load() == kTotal, "every token pushed");
    check(popped.load() == kTotal, "every token popped");
    check(dup == 0u, "NO duplicate consumption under concurrent steal");
    check(missing == 0u, "no token lost under concurrent steal");
    std::printf("RING mode=mpmc_steal D=2 producers=%u consumers=%u total=%u "
                "dup=%u missing=%u\n", kProducers, kConsumers, kTotal, dup,
                missing);
  }

  // Shard-local readiness tracker.
  {
    weave::ShardTokenTracker tr{};
    const uint32_t expected[4] = {3u, 1u, 2u, 4u};
    weave::shard_tracker_reset(&tr, expected, 4u);
    check(tr.num_tracked == 4u, "tracker sized");
    check(!weave::shard_tracker_note_hit(&tr, 0u, 1u), "hit 1/3 not ready");
    check(!weave::shard_tracker_note_hit(&tr, 0u, 1u), "hit 2/3 not ready");
    check(weave::shard_tracker_note_hit(&tr, 0u, 1u), "hit 3/3 ready");
    check(weave::shard_tracker_note_hit(&tr, 1u, 1u), "single-hit token ready");
    check(!weave::shard_tracker_note_hit(&tr, 9u, 1u), "out-of-range slot");
    std::printf("RING mode=tracker slots=4 ready=2\n");
  }
  delete rings;
  std::printf("[PASS] sharded ready rings: cluster-exclusive SPSC (owner-only, "
              "wrap), no-CAS MPSC epochs, unified-claim MPMC steal "
              "exactly-once\n");
}

// ---------------------------------------------------------------------------
// Round-2 synthesis: cluster DSMEM fan-out simulation (isomorphic to the real
// st.async + mbarrier path compiled by tests/compile_sm100_dsmem_sched.cu).
// ---------------------------------------------------------------------------

void test_cluster_dsmem_sim() {
  static_assert(sizeof(weave::ClusterSchedPacket) == 16u, "packet 16 B");
  static_assert(alignof(weave::ClusterSchedPacket) == 16u, "packet 16 B aligned");
  auto* sim = new weave::ClusterDsmemScheduleSim();
  uint32_t lines = 0u;
  for (uint32_t cs : {2u, 4u, 8u}) {
    weave::cluster_sched_sim_init(sim, cs, 0u);
    // Fan-out delivers to every peer except the leader, in issue order.
    for (uint32_t round = 0u; round < 5u; ++round) {
      for (uint32_t i = 0u; i < weave::kClusterSchedSlots; ++i) {
        const auto pkt = weave::make_cluster_sched_packet(
            round, round * 100u + i, i % 8u, 128u, 2u, 2u, 2u);
        check(weave::cluster_sched_sim_fanout(sim, pkt), "fanout accepted");
      }
      const auto overflow = weave::make_cluster_sched_packet(
          round, 999999u, 0u, 64u, 1u, 0u, 1u);
      check(!weave::cluster_sched_sim_fanout(sim, overflow),
            "full inbox rejected without partial delivery");
      for (uint32_t c = 0u; c < cs; ++c) {
        if (c == 0u) {
          check(sim->head[c] == 0u, "leader receives nothing");
          continue;
        }
        check(sim->expected_tx[c] == sim->completed_tx[c],
              "expect_tx == completed_tx (mbarrier balanced)");
        check(sim->completed_tx[c] ==
                  (round + 1u) * weave::kClusterSchedSlots * 16u,
              "16 transaction bytes per packet");
        for (uint32_t i = 0u; i < weave::kClusterSchedSlots; ++i) {
          weave::ClusterSchedPacket got{};
          check(weave::cluster_sched_sim_pop(sim, c, &got), "peer pop");
          check(got.token_id == round * 100u + i, "peer FIFO across wrap");
          check(got.epoch == round, "packet epoch preserved");
        }
        weave::ClusterSchedPacket empty{};
        check(!weave::cluster_sched_sim_pop(sim, c, &empty), "peer drained");
      }
    }
    std::printf("CLUSIM cluster=%u rounds=5 slots=%u tx=%u\n", cs,
                weave::kClusterSchedSlots, sim->completed_tx[cs - 1u]);
    ++lines;
  }
  // Non-zero leader index: the leader is skipped, everyone else receives.
  weave::cluster_sched_sim_init(sim, 4u, 2u);
  const auto pkt = weave::make_cluster_sched_packet(1u, 5u, 3u, 64u, 4u, 1u, 2u);
  check(weave::cluster_sched_sim_fanout(sim, pkt), "fanout with leader=2");
  check(sim->head[2] == 0u, "leader 2 skipped");
  check(sim->head[0] == 1u && sim->head[1] == 1u && sim->head[3] == 1u,
        "all non-leader peers received");
  // Channel parity semantics (simulation path).
  weave::ClusterSchedChannel ch{};
  weave::cluster_sched_barrier_init(&ch);
  check(!weave::cluster_sched_poll(&ch, 0u), "no packet -> parity unchanged");
  weave::cluster_sched_push(&ch, pkt, 1u);
  check(weave::cluster_sched_poll(&ch, 0u), "one packet flips parity");
  check(weave::cluster_sched_load(&ch).token_id == 5u, "slot payload landed");
  weave::cluster_sched_push(&ch, pkt, 1u);
  check(!weave::cluster_sched_poll(&ch, 0u), "two packets restore parity");
  delete sim;
  std::printf("[PASS] cluster DSMEM sim: %u cluster sizes, fan-out delivery + "
              "FIFO + wrap + tx accounting + parity\n", lines);
}

}  // namespace

#if defined(__CUDACC__) || defined(__HIPCC__)
// Minimal workspace surface used to force device instantiation of the policy
// rail helpers without pulling in the full DeepGEMM layout headers.
struct DummyWeaveWorkspace {
  uint64_t* counts;
  __device__ uint64_t* get_expert_send_count_ptr(uint32_t e) const {
    return counts + e;
  }
};

__global__ void weave_policy_runtime_device_smoke(
    uint64_t* counts, const weave::PolicyControl* control,
    weave::PolicyDecision* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const DummyWeaveWorkspace ws{counts};
    out[0] = weave::load_published_policy(control);
    out[1] = {};
    out[1].input_scatter_rows =
        weave::count_remote_expert_sends(ws, 8u, 2u, 0u);
  }
}

// Forces device-side instantiation of the 256-expert packed-histogram rail,
// including the over-capacity rejection path (E_local = 257).
__global__ void weave_policy_runtime_experts256_smoke(
    const uint64_t* packed_counts, weave::PolicyControl* control,
    weave::PolicyDecision* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    weave::WeaveKnobs knobs =
        weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    // Pin the full search so the device result is comparable to the host model
    // even when the file is built with an ablation preset or a V_head override.
    knobs.enable_runtime_cost_model = 1u;
    knobs.override_mask = 0u;
    knobs.reduce_warps = 4u;
    weave::WorkloadSpec wl{weave::kMaxRuntimeExperts, 2048u, 512u};
    out[0] = weave::cta0_select_from_packed_hist(
        control, weave::HardwareArch::B300_SM100, wl, knobs, packed_counts,
        0u, 0u, 1u);
    wl.local_experts = weave::kMaxRuntimeExperts + 1u;
    out[1] = weave::cta0_select_from_packed_hist(
        control, weave::HardwareArch::B300_SM100, wl, knobs, packed_counts,
        0u, 0u, 2u);
  }
}

// Forces device-side instantiation of the same functions used by all kernels.
__global__ void weave_cost_model_device_smoke(
    const uint32_t* m_e, weave::PolicyDecision* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const weave::WorkloadSpec wl{4u, 128u, 64u};
    const weave::WeaveKnobs knobs =
        weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    *out = weave::cta0_select_policy(
        weave::HardwareArch::B300_SM100, wl, knobs, m_e, 23u, 17u);
  }
}

// Forces device-side instantiation of the SM0 dedup-aware ordering helper
// (16 KB scratch in SMEM, one scheduler thread builds the permutation).
__global__ void weave_sm0_order_device_smoke(
    const uint32_t* counts, const uint32_t* multi, const uint32_t* partner,
    uint32_t e_count, uint32_t wave, uint32_t dedup, uint32_t* out) {
  __shared__ weave::Sm0OrderScratch scratch;
  if (threadIdx.x == 0) {
    out[0] = weave::build_sm0_expert_order(
        counts, multi, partner, e_count, wave, dedup, out + 1, &scratch);
    // Device latency evidence: mean clock64() cycles over 10 warm builds.
    const long long t0 = clock64();
    for (uint32_t r = 0u; r < 10u; ++r) {
      weave::build_sm0_expert_order(
          counts, multi, partner, e_count, wave, dedup, out + 1, &scratch);
    }
    const long long t1 = clock64();
    out[e_count + 1u] = static_cast<uint32_t>((t1 - t0) / 10);
  }
}

// Forces device-side instantiation of the lock-free ready-token queue:
// 32 lanes push concurrently (MPSC), then 32 pre-reduce hits on one row
// notify through atomicSub; lane 0 drains the guaranteed-ready tokens.
__global__ void weave_rtq_device_smoke(
    weave::Sm0ReadyTokenQueue* q, weave::Sm0ReadyCell* cells,
    uint32_t* remaining, uint32_t* out) {
  const uint32_t lane = threadIdx.x;
  weave::rtq_push(q, cells, lane);
  __syncwarp();
  weave::rtq_notify_local_prereduce(q, cells, remaining, 5u, 1234u);
  __syncwarp();
  if (lane == 0) {
    uint32_t tok = 0u, n = 0u;
    while (weave::rtq_pop(q, cells, &tok)) {
      out[1u + n] = tok;
      ++n;
    }
    out[0] = n;
  }
}

// Per-expert profile-code consumption: one acquire + one 4-byte load.
__global__ void weave_profile_code_device_smoke(
    const weave::PolicyControl* control, uint32_t* out) {
  if (threadIdx.x == 0) {
    out[0] = weave::load_expert_profile_code(control, 1u, 0u);
    out[1] = weave::load_expert_profile_code(control, 2u, 0u);
    out[2] = weave::load_expert_profile_code(control, 1u, 255u);
    out[3] = weave::load_expert_profile_code(control, 1u, 256u);
  }
}

// Forces device-side instantiation of the Tier-3 online publication and the
// per-CTA consumption helpers (global control block, zero SMEM).
__global__ void weave_online_policy_device_smoke(
    const uint32_t* m_e, weave::PolicyControl* control,
    weave::OnlineRoutingPolicy* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    weave::WeaveKnobs knobs =
        weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    knobs.enable_runtime_cost_model = 1u;
    knobs.override_mask = 0u;
    knobs.reduce_warps = 4u;
    knobs.adapt_bm = 1u;
    knobs.ready_group_size = 0u;
    knobs.dedup_mode = weave::kAutoU32;
    const weave::WorkloadSpec wl{128u, 8192u, 2048u};
    (void)weave::cta0_select_and_publish(
        control, weave::HardwareArch::B300_SM100, wl, knobs, m_e,
        964u, 1849u, 7u, 964u);
    out[0] = weave::load_published_online(control);
    out[1] = {};
    out[1].active_reduce_warps = weave::load_active_reduce_warps(control);
    out[1].active_v_head = weave::load_active_v_head(control);
    out[1].active_ready_group_size =
        weave::stream_reduce_stride(control, 1000u, 78u);
  }
}
#endif

#if defined(__CUDACC__)
#include <cuda_runtime.h>

// Executes the device-only policy rail on a real GPU.  cta0_select_from_packed_hist
// is unreachable from the host build, so E_local == 256 acceptance and the
// E_local == 257 rejection are only observable here.  A build host without a CUDA
// device skips this rather than failing.
static void test_device_policy_rail() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("[SKIP] device policy rail: no CUDA device on this host\n");
    return;
  }

  const uint32_t experts = weave::kMaxRuntimeExperts;
  std::vector<uint64_t> packed(experts + 1u, 0u);
  std::vector<uint32_t> hist(experts, 0u);
  uint32_t total = 0u;
  for (uint32_t e = 0u; e < experts; ++e) {
    hist[e] = e % 5u;
    packed[e] = hist[e];
    total += hist[e];
  }
  packed[experts] = 7u;

  uint64_t* d_counts = nullptr;
  weave::PolicyControl* d_control = nullptr;
  weave::PolicyDecision* d_out = nullptr;
  const bool alloc_ok =
      cudaMalloc(&d_counts, packed.size() * sizeof(uint64_t)) == cudaSuccess &&
      cudaMalloc(&d_control, sizeof(weave::PolicyControl)) == cudaSuccess &&
      cudaMalloc(&d_out, 2u * sizeof(weave::PolicyDecision)) == cudaSuccess;
  check(alloc_ok, "device rail cudaMalloc");
  if (!alloc_ok) {
    cudaFree(d_counts);
    cudaFree(d_control);
    cudaFree(d_out);
    return;
  }

  weave::PolicyControl control{};
  weave::reset_policy_control(&control);
  check(cudaMemcpy(d_counts, packed.data(),
                   packed.size() * sizeof(uint64_t),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_control, &control, sizeof(control),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemset(d_out, 0, 2u * sizeof(weave::PolicyDecision)) ==
                cudaSuccess,
        "device rail upload");

  weave_policy_runtime_experts256_smoke<<<1, 32>>>(d_counts, d_control, d_out);
  check(cudaGetLastError() == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess,
        "experts256 smoke launch");

  weave::PolicyDecision out[2]{};
  weave::PolicyControl after{};
  check(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(&after, d_control, sizeof(after),
                       cudaMemcpyDeviceToHost) == cudaSuccess,
        "device rail download");

  weave::WeaveKnobs knobs =
      weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
  knobs.enable_runtime_cost_model = 1u;
  knobs.override_mask = 0u;
  knobs.reduce_warps = 4u;
  const weave::WorkloadSpec wl{experts, 2048u, 512u};
  const auto host_d = weave::cta0_select_policy(
      weave::HardwareArch::B300_SM100, wl, knobs, hist.data(), total, total);

  check(weave::policy_is_valid(out[0]), "device E_local=256 accepted");
  check(out[0].input_total_tokens == total, "device 256 M snapshot");
  check(out[0].selected_candidate == host_d.selected_candidate &&
            out[0].selected_scatter_mode == host_d.selected_scatter_mode,
        "device 256 argmin matches host");
  check(out[0].predicted_total_us == host_d.predicted_total_us,
        "device 256 prediction bit-identical to host");
  check(after.published_epoch == 1u, "accepted call publishes epoch 1");
  bool hist_ok = true;
  for (uint32_t e = 0u; e < experts; ++e) {
    hist_ok = hist_ok && after.histogram[e] == hist[e];
  }
  check(hist_ok, "device histogram holds all 256 counts");

  check(out[1].magic == weave::kPolicyMagic &&
            out[1].version == weave::kPolicyVersion,
        "device E_local=257 returns a tagged invalid decision");
  check(out[1].valid == 0u && !weave::policy_is_valid(out[1]),
        "device E_local=257 rejected by the packed-histogram guard");
  check(after.published_epoch == 1u,
        "rejected call never publishes epoch 2");

  // The device cost model must agree with the host model bit-for-bit.
  const uint32_t m_e[4] = {0u, 3u, 5u, 9u};
  uint32_t* d_m_e = nullptr;
  check(cudaMalloc(&d_m_e, sizeof(m_e)) == cudaSuccess &&
            cudaMemcpy(d_m_e, m_e, sizeof(m_e),
                       cudaMemcpyHostToDevice) == cudaSuccess,
        "device cost-model upload");
  check(cudaMemset(d_out, 0, 2u * sizeof(weave::PolicyDecision)) == cudaSuccess,
        "device cost-model reset");
  weave_cost_model_device_smoke<<<1, 32>>>(d_m_e, d_out);
  check(cudaGetLastError() == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess,
        "cost-model smoke launch");
  weave::PolicyDecision dev_model{};
  check(cudaMemcpy(&dev_model, d_out, sizeof(dev_model),
                   cudaMemcpyDeviceToHost) == cudaSuccess,
        "device cost-model download");
  const weave::WorkloadSpec wl4{4u, 128u, 64u};
  const auto host_model = weave::cta0_select_policy(
      weave::HardwareArch::B300_SM100, wl4,
      weave::default_weave_knobs(weave::HardwareArch::B300_SM100),
      m_e, 23u, 17u);
  check(dev_model.predicted_total_us == host_model.predicted_total_us &&
            dev_model.predicted_mid_us == host_model.predicted_mid_us &&
            dev_model.predicted_comp_us == host_model.predicted_comp_us &&
            dev_model.predicted_pull_us == host_model.predicted_pull_us &&
            dev_model.predicted_scat_us == host_model.predicted_scat_us &&
            dev_model.selected_candidate == host_model.selected_candidate &&
            dev_model.selected_scatter_mode ==
                host_model.selected_scatter_mode &&
            dev_model.selected_v_head == host_model.selected_v_head,
        "device cost model bit-identical to host");

  // Tier-3 online policy: device publication must match the host bit-for-bit.
  const auto rank0 = max_anchored(2482u, 131u, 123u, 128u);
  uint32_t* d_hist128 = nullptr;
  weave::PolicyControl* d_online_ctrl = nullptr;
  weave::OnlineRoutingPolicy* d_online_out = nullptr;
  check(cudaMalloc(&d_hist128, rank0.size() * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_online_ctrl, sizeof(weave::PolicyControl)) == cudaSuccess &&
            cudaMalloc(&d_online_out,
                       2u * sizeof(weave::OnlineRoutingPolicy)) == cudaSuccess,
        "online rail cudaMalloc");
  weave::PolicyControl online_ctrl{};
  weave::reset_policy_control(&online_ctrl);
  check(cudaMemcpy(d_hist128, rank0.data(),
                   rank0.size() * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_online_ctrl, &online_ctrl, sizeof(online_ctrl),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemset(d_online_out, 0,
                       2u * sizeof(weave::OnlineRoutingPolicy)) == cudaSuccess,
        "online rail upload");
  weave_online_policy_device_smoke<<<1, 32>>>(d_hist128, d_online_ctrl,
                                              d_online_out);
  check(cudaGetLastError() == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess,
        "online smoke launch");
  weave::OnlineRoutingPolicy dev_online[2]{};
  weave::PolicyControl online_after{};
  check(cudaMemcpy(dev_online, d_online_out, sizeof(dev_online),
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(&online_after, d_online_ctrl, sizeof(online_after),
                       cudaMemcpyDeviceToHost) == cudaSuccess,
        "online rail download");
  weave::WeaveKnobs oknobs =
      weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
  oknobs.enable_runtime_cost_model = 1u;
  oknobs.override_mask = 0u;
  oknobs.reduce_warps = 4u;
  oknobs.adapt_bm = 1u;
  oknobs.ready_group_size = 0u;
  oknobs.dedup_mode = weave::kAutoU32;
  const weave::WorkloadSpec wl128{128u, 8192u, 2048u};
  weave::OnlineRoutingPolicy host_online{};
  (void)weave::cta0_select_policy(
      weave::HardwareArch::B300_SM100, wl128, oknobs, rank0.data(),
      964u, 1849u, &host_online, 964u);
  check(dev_online[0].active_reduce_warps == host_online.active_reduce_warps &&
            dev_online[0].active_v_head == host_online.active_v_head &&
            dev_online[0].active_ready_group_size ==
                host_online.active_ready_group_size &&
            dev_online[0].enable_dedup_runtime ==
                host_online.enable_dedup_runtime &&
            dev_online[0].bm_tier_mean == host_online.bm_tier_mean &&
            dev_online[0].active_experts == host_online.active_experts &&
            dev_online[0].max_n == host_online.max_n &&
            dev_online[0].total_routes == host_online.total_routes,
        "device online integer fields match host");
  check(float_bits(dev_online[0].mean_n) == float_bits(host_online.mean_n) &&
            float_bits(dev_online[0].rank_hit_multiplicity_mu) ==
                float_bits(host_online.rank_hit_multiplicity_mu) &&
            float_bits(dev_online[0].middle_hbm_util) ==
                float_bits(host_online.middle_hbm_util) &&
            float_bits(dev_online[0].tail_combine_frac) ==
                float_bits(host_online.tail_combine_frac),
        "device online float fields bit-identical to host");
  check(dev_online[1].active_reduce_warps == host_online.active_reduce_warps &&
            dev_online[1].active_v_head == host_online.active_v_head,
        "device single-field online loads match");
  check(dev_online[1].active_ready_group_size ==
            (host_online.active_reduce_warps > 0u ? 1000u : 78u),
        "device stream_reduce_stride follows the gate");
  check(online_after.published_epoch == 7u &&
            online_after.online.active_reduce_warps ==
                host_online.active_reduce_warps,
        "device online published under epoch 7");

  // Per-expert profile codes: device single-field loads match the host LUT.
  uint32_t* d_codes = nullptr;
  check(cudaMalloc(&d_codes, 4u * sizeof(uint32_t)) == cudaSuccess &&
            cudaMemset(d_codes, 0, 4u * sizeof(uint32_t)) == cudaSuccess,
        "profile-code rail alloc");
  weave_profile_code_device_smoke<<<1, 32>>>(d_online_ctrl, d_codes);
  check(cudaGetLastError() == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess,
        "profile-code smoke launch");
  uint32_t codes[4] = {0u, 0u, 0u, 0u};
  check(cudaMemcpy(codes, d_codes, sizeof(codes), cudaMemcpyDeviceToHost) ==
            cudaSuccess,
        "profile-code download");
  check(codes[0] == host_online.l1_profile_code[0] &&
            codes[1] == host_online.l2_profile_code[0] &&
            codes[2] == host_online.l1_profile_code[255] && codes[3] == 0u,
        "device per-expert profile codes match host");
  cudaFree(d_codes);

  // SM0 ordering on device: same permutation as the host helper.
  {
    const uint32_t e_count = 128u;
    std::vector<uint32_t> multi(e_count), partner(e_count);
    for (uint32_t e = 0u; e < e_count; ++e) {
      multi[e] = rank0[e] / 2u;
      partner[e] = (e + 64u) % e_count;
    }
    uint32_t *d_multi = nullptr, *d_partner = nullptr, *d_order = nullptr;
    check(cudaMalloc(&d_multi, e_count * sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc(&d_partner, e_count * sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc(&d_order, (e_count + 2u) * sizeof(uint32_t)) ==
                  cudaSuccess,
          "order rail alloc");
    check(cudaMemcpy(d_multi, multi.data(), e_count * sizeof(uint32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_partner, partner.data(), e_count * sizeof(uint32_t),
                         cudaMemcpyHostToDevice) == cudaSuccess,
          "order rail upload");
    weave_sm0_order_device_smoke<<<1, 32>>>(d_hist128, d_multi, d_partner,
                                            e_count, 78u, 1u, d_order);
    check(cudaGetLastError() == cudaSuccess &&
              cudaDeviceSynchronize() == cudaSuccess,
          "order smoke launch");
    std::vector<uint32_t> dev_order(e_count + 2u);
    check(cudaMemcpy(dev_order.data(), d_order,
                     (e_count + 2u) * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "order download");
    std::vector<uint32_t> host_order(e_count);
    auto* scratch = new weave::Sm0OrderScratch();
    const uint32_t idx = weave::build_sm0_expert_order(
        rank0.data(), multi.data(), partner.data(), e_count, 78u, 1u,
        host_order.data(), scratch);
    delete scratch;
    check(dev_order[0] == idx && idx == e_count, "device order covers E");
    bool same = true;
    for (uint32_t i = 0u; i < e_count; ++i) {
      same = same && dev_order[1u + i] == host_order[i];
    }
    check(same, "device SM0 order bit-identical to host");
    std::printf("[INFO] device SM0 order: E=%u mean %u cycles/build "
                "(%.3f us at 1.5 GHz)\n", e_count, dev_order[e_count + 1u],
                dev_order[e_count + 1u] / 1500.0);
    cudaFree(d_multi);
    cudaFree(d_partner);
    cudaFree(d_order);
  }

  // Ready-token queue on device: 32 concurrent pushes + 32-hit pre-reduce.
  {
    weave::Sm0ReadyTokenQueue* d_q = nullptr;
    weave::Sm0ReadyCell* d_cells = nullptr;
    uint32_t *d_remaining = nullptr, *d_rqout = nullptr;
    check(cudaMalloc(&d_q, sizeof(weave::Sm0ReadyTokenQueue)) == cudaSuccess &&
              cudaMalloc(&d_cells, 64u * sizeof(weave::Sm0ReadyCell)) ==
                  cudaSuccess &&
              cudaMalloc(&d_remaining, sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc(&d_rqout, 64u * sizeof(uint32_t)) == cudaSuccess,
          "rtq rail alloc");
    weave::Sm0ReadyTokenQueue q{};
    weave::rtq_init(&q, 64u, 1u);
    const uint32_t remaining_init = 32u;
    check(cudaMemcpy(d_q, &q, sizeof(q), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemset(d_cells, 0, 64u * sizeof(weave::Sm0ReadyCell)) ==
                  cudaSuccess &&
              cudaMemcpy(d_remaining, &remaining_init, sizeof(uint32_t),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemset(d_rqout, 0, 64u * sizeof(uint32_t)) == cudaSuccess,
          "rtq rail upload");
    weave_rtq_device_smoke<<<1, 32>>>(d_q, d_cells, d_remaining, d_rqout);
    check(cudaGetLastError() == cudaSuccess &&
              cudaDeviceSynchronize() == cudaSuccess,
          "rtq smoke launch");
    uint32_t rqout[64] = {};
    uint32_t remaining_after = 0u;
    check(cudaMemcpy(rqout, d_rqout, sizeof(rqout), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              cudaMemcpy(&remaining_after, d_remaining, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost) == cudaSuccess,
          "rtq download");
    check(rqout[0] == 33u, "device queue popped 32 pushes + 1 ready row");
    check(remaining_after == 0u, "device pre-reduce counter drained");
    check(rqout[33] == weave::rtq_encode_row(5u, 1234u),
          "device last-hit row published");
    bool lanes[32] = {};
    bool all = true;
    for (uint32_t i = 1u; i <= 32u; ++i) {
      if (rqout[i] >= 32u || lanes[rqout[i]]) all = false;
      else lanes[rqout[i]] = true;
    }
    check(all, "device concurrent pushes popped exactly once each");
    cudaFree(d_q);
    cudaFree(d_cells);
    cudaFree(d_remaining);
    cudaFree(d_rqout);
  }

  cudaFree(d_hist128);
  cudaFree(d_online_ctrl);
  cudaFree(d_online_out);

  cudaFree(d_m_e);
  cudaFree(d_counts);
  cudaFree(d_control);
  cudaFree(d_out);
  std::printf("[PASS] device policy rail: E_local=256 accepted, 257 rejected, "
              "device/host predictions bit-identical, SM0 order + ready queue "
              "+ profile codes device-verified\n");
}
#endif

static int run_all_tests() {
  test_arch(weave::HardwareArch::B300_SM100, "B300_SM100");
  test_arch(weave::HardwareArch::H20_SM90, "H20_SM90");
  test_arch(weave::HardwareArch::MI308X_GFX942, "MI308X_GFX942");
  test_precision_modes();
  test_max_runtime_experts();
  test_v_head_override();
  test_reduce_warps();
  test_online_routing_policy();
  test_phase_nk_profiles();
  test_sm0_expert_order();
  test_sm0_ready_queue();
  test_online_profile_lut();
  test_sm0_order_modes();
  test_lockfree_prereduce();
  test_sharded_ready_ring();
  test_cluster_dsmem_sim();
#if defined(__CUDACC__)
  test_device_policy_rail();
#endif
  if (failures != 0) {
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("ALL EQ(1)-(3), CTA-0 SELECTION, E_local<=256, WEAVE_V_HEAD, WEAVE_REDUCE_WARPS AND TWO-LEVEL ONLINE POLICY TESTS PASSED\n");
  std::printf("ALL PHASE-NK PROFILES, SM0 EXPERT ORDER, READY-TOKEN QUEUE AND PROFILE-LUT TESTS PASSED\n");
  std::printf("ALL SM0 MODE VERDICTS, LOCK-FREE PRE-REDUCE, SHARDED READY RINGS AND CLUSTER DSMEM TESTS PASSED\n");
  return EXIT_SUCCESS;
}

#if defined(_WIN32)
extern "C" __declspec(dllexport) int weave_run_tests() {
  return run_all_tests();
}
#endif

int main() {
  return run_all_tests();
}



