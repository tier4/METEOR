cmake_minimum_required(VERSION 3.18)
project(meteor_archv2_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDAToolkit REQUIRED)
add_executable(archv2_bench archv2_bench.cu)
target_link_libraries(archv2_bench PRIVATE CUDA::cudart)
target_compile_options(archv2_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)
