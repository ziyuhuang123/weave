#pragma once

#include <cstdlib>
#include <optional>
#include <string>

namespace deep_gemm {

// Internal compile-only overrides.  They are honored ONLY when the environment
// variable `DG_JIT_COMPILE_ONLY=1` is set (the kernel is built but never loaded
// or launched), and are silently ignored otherwise:
//   DG_JIT_TARGET_ARCH     e.g. `100a` / `103a` / `100f` / `103f`: the nvcc
//                          `--gpu-architecture=sm_<value>` target
//   DG_JIT_TARGET_NUM_SMS  the SM count seen by the host (may exceed the
//                          physical SM count)
// Because the arch is part of the nvcc flags and the SM count is part of the
// generated source / launch config, the JIT cache key differs per target.
inline bool compile_only_enabled() {
    const char* value = std::getenv("DG_JIT_COMPILE_ONLY");
    return value != nullptr and std::atoi(value) != 0;
}

inline std::optional<std::string> compile_only_target_arch() {
    if (not compile_only_enabled())
        return std::nullopt;
    const char* value = std::getenv("DG_JIT_TARGET_ARCH");
    if (value == nullptr or *value == '\0')
        return std::nullopt;
    return std::string(value);
}

inline std::optional<int> compile_only_target_num_sms() {
    if (not compile_only_enabled())
        return std::nullopt;
    const char* value = std::getenv("DG_JIT_TARGET_NUM_SMS");
    if (value == nullptr or *value == '\0')
        return std::nullopt;
    const int num_sms = std::atoi(value);
    return num_sms > 0 ? std::optional<int>(num_sms) : std::nullopt;
}

}  // namespace deep_gemm
