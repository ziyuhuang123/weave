# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
cmake_minimum_required(VERSION 3.18)
project(deep_gemm LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(deep_gemm/include third-party/deep_jit/include third-party/cutlass/include third-party/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/cccl ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# The main Python API entrance
pybind11_add_module(_C csrc/python_api.cpp)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python)

# Enable kernel code indexing with CMake-based IDEs
cuda_add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)


# Standalone C++17 cost-model test.  It deliberately does not link Torch/CUDA
# and can also be configured from tests/CMakeLists.txt on a CPU-only machine.
option(WEAVE_BUILD_COST_MODEL_TEST "Build the Eq.(1)-(3) CTA-0 unit test" ON)
if (WEAVE_BUILD_COST_MODEL_TEST)
  add_executable(weave_test_cost_model_eq123 tests/test_cost_model_eq123.cpp)
  target_include_directories(weave_test_cost_model_eq123 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/deep_gemm/include)
  set_target_properties(weave_test_cost_model_eq123 PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

  add_executable(weave_test_ablation_presets tests/test_ablation_presets.cpp)
  target_include_directories(weave_test_ablation_presets PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/deep_gemm/include)
  set_target_properties(weave_test_ablation_presets PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endif()
