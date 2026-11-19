// Verifies that WEAVE_ABLATION_PRESET 1/2/3 disable the runtime spatial
// search and fall back to the fixed per-architecture candidate, while preset 0
// enables the full CTA-0 cost model.
#include <cstdio>
#include <cstdlib>

#include <deep_gemm/scheduler/weave_cost_model.cuh>
#include <deep_gemm/scheduler/weave_policy_runtime.cuh>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void test_arch(weave::HardwareArch arch, const char* name) {
  const auto knobs = weave::default_weave_knobs(arch);
#if WEAVE_ABLATION_PRESET == 0
  check(knobs.enable_runtime_cost_model == 1u, "preset0 enables cost model");
  check(knobs.override_mask == 0u, "preset0 has no forced override");
#else
  check(knobs.enable_runtime_cost_model == 0u, "ablation disables cost model");
  check(knobs.override_mask == 3u, "ablation forces spatial+scatter");
  check(knobs.override_r_mask == weave::default_r_mask_for_arch(arch),
        "ablation uses fixed candidate");
#endif
  const uint32_t m_e[4] = {0u, 3u, 5u, 9u};
  const weave::WorkloadSpec wl{4u, 128u, 64u};
  const auto d = weave::cta0_select_policy(arch, wl, knobs, m_e, 23u, 17u);
  check(weave::policy_is_valid(d), "decision valid");
#if WEAVE_ABLATION_PRESET != 0
  check(d.selected_r_mask == weave::default_r_mask_for_arch(arch),
        "selected fixed r mask");
#endif
  check(d.input_total_tokens == 17u, "live M snapshot");
  std::printf("[PASS] %s preset=%d r=0x%08x scatter=%u v_head=%u\n",
              name, WEAVE_ABLATION_PRESET, d.selected_r_mask,
              d.selected_scatter_mode, d.selected_v_head);
}

}  // namespace

int main() {
  test_arch(weave::HardwareArch::B300_SM100, "B300_SM100");
  test_arch(weave::HardwareArch::H20_SM90, "H20_SM90");
  test_arch(weave::HardwareArch::MI308X_GFX942, "MI308X_GFX942");
  if (failures != 0) {
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("ABLATION PRESET %d PASSED\n", WEAVE_ABLATION_PRESET);
  return EXIT_SUCCESS;
}
