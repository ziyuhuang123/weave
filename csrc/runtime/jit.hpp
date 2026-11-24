#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/utils/env.hpp>

#include "../jit/compile_only.hpp"

namespace deep_gemm {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION),
        {include_dir},
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        // Preserve the existing 12.x family target when DeepJIT derives a different minor target.
        if (runtime->device.get_arch_major() == 12 and runtime->device.get_arch_minor() != 0)
            runtime->default_compiler_options.arch = "120f";
        // Keep the SM100 family target for SM100/SM102, but let SM103 use its
        // own family target: the upstream reference cubin is built as sm_103f,
        // and forcing sm_100f costs a 192 B stack frame plus 180 local loads.
        if (runtime->device.get_arch_major() == 10 and runtime->device.get_arch_minor() != 1)
            runtime->default_compiler_options.arch =
                runtime->device.get_arch_minor() == 3 ? "103f" : "100f";
        // Compile-only target override (DG_JIT_COMPILE_ONLY=1 + DG_JIT_TARGET_ARCH).
        if (const auto target_arch = compile_only_target_arch(); target_arch.has_value())
            runtime->default_compiler_options.arch = *target_arch;
        // Surface ptxas resource usage / local-memory warnings when asked.
        // DG_JIT_PTXAS_CHECK=2: fail the build on register spills only (a non-spill stack frame is allowed).
        if (deep_jit::get_env<int>("DG_JIT_PTXAS_CHECK", 0) == 2) {
            runtime->default_compiler_options.ptxas_verbose = true;
            runtime->default_compiler_options.check_no_spills = true;
        } else if (deep_jit::get_env<int>("DG_JIT_PTXAS_VERBOSE", 0) or deep_jit::get_env<int>("DG_JIT_PTXAS_CHECK", 0)) {
            runtime->default_compiler_options.ptxas_verbose = true;
            runtime->default_compiler_options.extra_nvcc_flags.emplace_back(
                "--ptxas-options=--warn-on-local-memory-usage");
        }
        return runtime;
    });
}

}  // namespace deep_gemm
