cmake_minimum_required(VERSION 3.18)
project(meteor_infer LANGUAGES CXX)  # no .cu here: CUDA via CUDAToolkit only (Orin has no nvcc in PATH)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# TensorRT location: -DTENSORRT_DIR=... or env TENSORRT_DIR, falling back to
# system paths (JetPack installs the headers under /usr/include/<triple>)
if(NOT TENSORRT_DIR)
  if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
  else()
    set(TENSORRT_DIR /usr)   # or -DTENSORRT_DIR=/path/to/TensorRT
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_DIR}/include
  PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
        /usr/include)
find_library(TENSORRT_NVINFER_LIB nvinfer
  HINTS ${TENSORRT_DIR}/lib
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser
  HINTS ${TENSORRT_DIR}/lib
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(TENSORRT_NVINFER_PLUGIN_LIB nvinfer_plugin
  HINTS ${TENSORRT_DIR}/lib
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
if(NOT TENSORRT_NVINFER_PLUGIN_LIB)
  set(TENSORRT_NVINFER_PLUGIN_LIB "")
endif()
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
  message(FATAL_ERROR "TensorRT not found (set -DTENSORRT_DIR=...)")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

add_executable(meteor_infer meteor_infer.cpp)
target_include_directories(meteor_infer PRIVATE
  ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(meteor_infer PRIVATE
  ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB}
  CUDA::cudart ${OpenCV_LIBS})

# realtime pipelined demo (port of deploy/orin_realtime.py)
add_executable(meteor_realtime
  realtime_main.cpp meteor_rt.cpp render.cpp viz.cpp decode.cpp)
target_include_directories(meteor_realtime PRIVATE
  ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(meteor_realtime PRIVATE
  ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVINFER_PLUGIN_LIB} CUDA::cudart
  ${OpenCV_LIBS} ZLIB::ZLIB Threads::Threads ${CMAKE_DL_LIBS})
