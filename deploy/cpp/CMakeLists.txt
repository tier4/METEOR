cmake_minimum_required(VERSION 3.18)
project(meteor_infer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# TensorRT location: -DTENSORRT_DIR=... or env TENSORRT_DIR
if(NOT TENSORRT_DIR)
  if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
  else()
    set(TENSORRT_DIR /home/umedan/TensorRT-8.6.0.12)
  endif()
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(meteor_infer meteor_infer.cpp)
target_include_directories(meteor_infer PRIVATE
  ${TENSORRT_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_directories(meteor_infer PRIVATE ${TENSORRT_DIR}/lib)
target_link_libraries(meteor_infer PRIVATE
  nvinfer nvonnxparser CUDA::cudart ${OpenCV_LIBS})
