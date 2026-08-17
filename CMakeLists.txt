cmake_minimum_required(VERSION 3.22)
project(MOSAIC_ODE LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WEAVEODE_ENABLE_WARNINGS "Enable strict compiler warnings" ON)

option(WEAVEODE_ENABLE_CUDA "Build optional CUDA device kernels" OFF)
option(WEAVEODE_ENABLE_NVRTC "Build optional NVRTC runtime RHS compiler helper" OFF)
if(WEAVEODE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "WEAVEODE_ENABLE_CUDA=ON but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  add_library(weaveode_cuda OBJECT
    cuda/candidate_score.cu
    cuda/vector_candidate_score.cu
    cuda/topology_h0.cu
    cuda/online_low_rank.cu
    cuda/block_krylov.cu
    cuda/conditional_graph.cu)
  target_include_directories(weaveode_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(weaveode_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
endif()

add_library(weaveode_core STATIC cpp/core.cpp)
target_include_directories(weaveode_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(weaveode_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WEAVEODE_ENABLE_WARNINGS)
  target_compile_options(weaveode_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(weaveode_c_api_smoke tests/c/test_c_api.c)
target_link_libraries(weaveode_c_api_smoke PRIVATE weaveode_core)
target_include_directories(weaveode_c_api_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(weaveode_c_api_smoke PROPERTIES LINKER_LANGUAGE CXX)

enable_testing()
add_test(NAME c_api_smoke COMMAND weaveode_c_api_smoke)

add_executable(weaveode_candidate_bundle_smoke tests/cpp/test_candidate_bundle.cpp)
target_link_libraries(weaveode_candidate_bundle_smoke PRIVATE weaveode_core)
target_include_directories(weaveode_candidate_bundle_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME candidate_bundle_smoke COMMAND weaveode_candidate_bundle_smoke)

if(WEAVEODE_ENABLE_NVRTC)
  find_package(CUDAToolkit REQUIRED)
  add_library(weaveode_nvrtc STATIC cuda/runtime_rhs_nvrtc.cpp)
  target_compile_definitions(weaveode_nvrtc PRIVATE WEAVEODE_ENABLE_NVRTC=1)
  target_link_libraries(weaveode_nvrtc PRIVATE CUDA::nvrtc)
  set_target_properties(weaveode_nvrtc PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
