#pragma once

// Minimal {fmt}-compatible facade backed by C++20 <format>.
//
// Root cause: the trimmed weave_unified_cpp csrc references utils/format.hpp,
// which upstream DeepGEMM (commit 559d79fb) implements as a wrapper over the
// third-party {fmt} library (<fmt/base.h>, <fmt/format.h>).  {fmt} is NOT
// present in any build include path on the reference build host (deep_jit / cutlass /
// CUDA / cccl all lack it).  Every fmt::format call site in this repository
// (heuristics/mega_moe.hpp:343, impls/sm100_fp8_fp4_mega_moe.hpp:177, and the
// restored jit/* headers) uses a literal format string with std-formattable
// arguments (int / bool / std::string / const char*), so std::vformat is a
// drop-in replacement and removes the external {fmt} dependency.
#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace fmt {
template <typename... Args>
inline std::string format(std::string_view fmt_str, Args&&... args) {
    return std::vformat(fmt_str, std::make_format_args(args...));
}
}  // namespace fmt
