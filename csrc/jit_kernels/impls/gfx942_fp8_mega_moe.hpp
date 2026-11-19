#pragma once

// MI308X launch wrapper for the single-launch EP16 HIP kernel.
// The 800-byte envelope and 576-byte context ABI are defined by
// deep_gemm/impls/amd/include/ep16_transport.hpp.

#include <cstdint>
#include <stdexcept>

#include <deep_gemm/scheduler/weave_knobs.cuh>

#if defined(__HIPCC__) && defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <deep_gemm/impls/gfx942_fp8_mega_moe.cuh>
#endif

namespace deep_gemm::weave_api {

inline weave::WeaveKnobs get_weave_knobs_for_mi308x() {
  return weave::default_weave_knobs(weave::HardwareArch::MI308X_GFX942);
}

struct Gfx942MegaMoEArgs {
  const void* envelope;  // V983Envelope, 800 bytes
  const void* context;   // Ep16TurboContext, 576 bytes
  int grid_x = 80;
  int block_x = 512;
  void* stream = nullptr;
};

inline void launch_gfx942_fp8_mega_moe(const Gfx942MegaMoEArgs& args) {
#if defined(__HIPCC__) && defined(__HIP_PLATFORM_AMD__)
  hipLaunchKernelGGL(
      stage_ep16_turbo_overlap,
      dim3(static_cast<unsigned>(args.grid_x)),
      dim3(static_cast<unsigned>(args.block_x)),
      0, reinterpret_cast<hipStream_t>(args.stream),
      *reinterpret_cast<const V983Envelope*>(args.envelope),
      reinterpret_cast<const Ep16TurboContext*>(args.context));
#else
  (void)args;
  throw std::runtime_error(
      "gfx942_fp8_mega_moe requires a HIPCC gfx942 build");
#endif
}

}  // namespace deep_gemm::weave_api
