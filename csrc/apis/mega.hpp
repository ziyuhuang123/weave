#pragma once

// Unified public C++ surface for the three Weave MegaMoE backends.
//
// NVIDIA B300 and H20 share the DeepGEMM MegaMoE JIT path.  MI308X exposes the
// EP16 HIP launch ABI through gfx942_fp8_mega_moe.hpp.  Python users should
// normally use deep_gemm.mega.weave_mega_moe(..., mma_type=...), which selects
// the same backend automatically.

#include <string_view>

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

// First-class precision selector.  This is the C++ mirror of the Python
// mma_type argument and of WEAVE_KNOB_PRECISION_MODE.
enum class MmaType : uint32_t {
  kFP8xFP4 = 0,
  kNVFP4 = 1,
  kFP8 = 2,
  kBF16 = 3,
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

inline constexpr const char* mma_type_name(MmaType mma_type) {
  return mma_type == MmaType::kFP8xFP4 ? "fp8xfp4" :
         mma_type == MmaType::kNVFP4 ? "fp4xfp4" :
         mma_type == MmaType::kFP8 ? "fp8" : "bf16";
}

inline constexpr weave::PrecisionMode to_precision_mode(MmaType mma_type) {
  return mma_type == MmaType::kFP8xFP4 ? weave::PrecisionMode::FP8xFP4 :
         mma_type == MmaType::kNVFP4 ? weave::PrecisionMode::NVFP4 :
         mma_type == MmaType::kFP8 ? weave::PrecisionMode::FP8xFP8 :
                                     weave::PrecisionMode::BF16;
}

// Canonical dispatch table used by both the Python entry point and host code.
struct WeaveMegaMoEDispatch {
  MmaType mma_type;
  Backend backend;
  weave::PrecisionMode precision_mode;
  const char* kernel_header;
};

inline WeaveMegaMoEDispatch weave_mega_moe_dispatch(
    std::string_view mma_type) {
  if (mma_type == "fp8xfp4")
    return {MmaType::kFP8xFP4, Backend::kB300,
            weave::PrecisionMode::FP8xFP4,
            "deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"};
  if (mma_type == "fp4xfp4" or mma_type == "nvfp4")
    return {MmaType::kNVFP4, Backend::kB300,
            weave::PrecisionMode::NVFP4,
            "deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh"};
  if (mma_type == "fp8" or mma_type == "fp8xfp8")
    return {MmaType::kFP8, Backend::kH20,
            weave::PrecisionMode::FP8xFP8,
            "deep_gemm/impls/sm90_fp8_mega_moe.cuh"};
  if (mma_type == "bf16")
    return {MmaType::kBF16, Backend::kB300,
            weave::PrecisionMode::BF16,
            "deep_gemm/impls/sm100_bf16_mega_moe.cuh"};
  return {MmaType::kFP8xFP4, Backend::kB300,
          weave::PrecisionMode::FP8xFP4, nullptr};
}

inline weave::WeaveKnobs weave_knobs_for(MmaType mma_type) {
  const auto dispatch = weave_mega_moe_dispatch(mma_type_name(mma_type));
  return weave::default_weave_knobs(
      to_hardware_arch(dispatch.backend), 0u, dispatch.precision_mode);
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
