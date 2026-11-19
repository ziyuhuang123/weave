cmake_minimum_required(VERSION 3.18)
project(weave_cost_model_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test_cost_model_eq123 test_cost_model_eq123.cpp)
target_include_directories(test_cost_model_eq123 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../deep_gemm/include)
if (MSVC)
  target_compile_options(test_cost_model_eq123 PRIVATE /W4 /O2)
else()
  target_compile_options(test_cost_model_eq123 PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_ablation_presets test_ablation_presets.cpp)
target_include_directories(test_ablation_presets PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../deep_gemm/include)
if (MSVC)
  target_compile_options(test_ablation_presets PRIVATE /W4 /O2)
else()
  target_compile_options(test_ablation_presets PRIVATE -O2 -Wall -Wextra)
endif()
