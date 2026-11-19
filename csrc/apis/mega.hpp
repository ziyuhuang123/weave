#pragma once

// Unified public C++ surface for the three Weave MegaMoE backends.
//
// NVIDIA B300 and H20 share the DeepGEMM MegaMoE JIT path.  MI308X exposes the
// EP16 HIP launch ABI through gfx942_fp8_mega_moe.hpp.  Python users should
// normally use deep_gemm.mega, which selects the same backend automatically.

#include "mega_moe.hpp"
#include "sm90_mega.hpp"

#include <deep_gemm/scheduler/weave_knobs.cuh>
#include <deep_gemm/scheduler/weave_cost_model.cuh>

namespace deep_gemm::weave_api {

enum class Backend : uint32_t {
  kB300 = 0,
  kH20 = 1,
  kMI308X = 2,
};

inline constexpr weave::HardwareArch to_hardware_arch(Backend backend) {
  return backend == Backend::kB300 ? weave::HardwareArch::B300_SM100 :
         backend == Backend::kH20 ? weave::HardwareArch::H20_SM90 :
                                    weave::HardwareArch::MI308X_GFX942;
}

inline const char* backend_name(Backend backend) {
  return backend == Backend::kB300 ? "b300" :
         backend == Backend::kH20 ? "h20" : "mi308x";
}

struct PolicyRequest {
  Backend backend;
  weave::WorkloadSpec workload;
  weave::WeaveKnobs knobs;
  const uint32_t* m_e;
  uint32_t dedup_pulls;
  uint32_t scatter_count;
};

inline weave::PolicyDecision select_policy(const PolicyRequest& request) {
  return weave::cta0_select_policy(
      to_hardware_arch(request.backend), request.workload, request.knobs,
      request.m_e, request.dedup_pulls, request.scatter_count);
}

}  // namespace deep_gemm::weave_api
