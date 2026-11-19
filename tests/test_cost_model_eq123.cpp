// Standalone C++17 host/device-model unit test for Weave Eq. (1)-(3).
// Build with either:
//   nvcc -std=c++17 -x cu tests/test_cost_model_eq123.cpp -o test_cost_model
//   cl /std:c++17 /EHsc tests\test_cost_model_eq123.cpp
//
// The test intentionally recomputes every equation independently from the
// production evaluator and compares all fields.  It also brute-forces the
// CTA-0 argmin over the same candidate set.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
          arch, hw, wl, candidates.items[ci], s, m_e.data(), d_in, m_out);
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
#endif
static int run_all_tests() {
  test_arch(weave::HardwareArch::B300_SM100, "B300_SM100");
  test_arch(weave::HardwareArch::H20_SM90, "H20_SM90");
  test_arch(weave::HardwareArch::MI308X_GFX942, "MI308X_GFX942");
  if (failures != 0) {
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("ALL EQ(1)-(3) AND CTA-0 SELECTION TESTS PASSED\n");
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



