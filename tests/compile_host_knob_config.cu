#include "csrc/jit_kernels/heuristics/mega_moe.hpp"
#include "csrc/jit_kernels/heuristics/weave_knob_config.hpp"

int main() {
    const auto knobs = deep_gemm::weave_knobs::from_env(
        weave::HardwareArch::B300_SM100, weave::PrecisionMode::FP8xFP4);
    return static_cast<int>(knobs.dispatch_warps + knobs.locality_domains +
                            knobs.ring_blocks + knobs.tmem_pipeline_mode);
}
