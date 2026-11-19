#pragma once

#include "../jit/compiler.hpp"
#include "../jit/include_parser.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"
#include "../runtime/runtime.hpp"

namespace deep_gemm::config {

static void register_apis(pybind11::module_& m) {
    m.def("init", [&](const std::string& library_root_path,
                      const std::string& cuda_home_path_by_python) {
        init_jit(library_root_path);
        // Legacy JIT kernels (notably FP8xFP4 MegaMoE) still use Compiler and
        // KernelRuntime directly.  Initialize their static paths here as well.
        if (not cuda_home_path_by_python.empty()) {
            Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
            KernelRuntime::prepare_init(cuda_home_path_by_python);
            IncludeParser::prepare_init(library_root_path);
        }
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return runtime->get_tc_util();
    });
    m.def("set_pdl", [](const bool& new_enable_pdl) {
        jit->default_launch_options.enable_pdl = new_enable_pdl;
    });
    m.def("get_pdl", []() {
        return *jit->default_launch_options.enable_pdl;
    });
    m.def("use_deterministic_algorithms", [&](const bool enabled) {
        heuristics_runtime->use_deterministic_algorithms(enabled);
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
}

} // namespace deep_gemm::config
