cmake_minimum_required(VERSION 3.18)
project(meteor_lift_plugin LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()
find_package(CUDAToolkit REQUIRED)

find_path(TRT_INCLUDE NvInferRuntime.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
        /usr/include /usr/local/tensorrt/include)
find_library(TRT_LIB nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib)

add_library(meteor_lift SHARED meteor_lift_plugin.cu)
target_include_directories(meteor_lift PRIVATE ${TRT_INCLUDE})
target_link_libraries(meteor_lift PRIVATE ${TRT_LIB} CUDA::cudart)
target_compile_options(meteor_lift PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
set_target_properties(meteor_lift PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared POSITION_INDEPENDENT_CODE ON)

add_executable(test_kernels test_kernels.cu)
target_link_libraries(test_kernels PRIVATE CUDA::cudart)
target_compile_options(test_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
