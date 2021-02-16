# /**
# MIT License

# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *
# */

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(yolo-plugin-tesla LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)

set(TRT_SDK_ROOT "" CACHE PATH "NVIDIA TensorRT SDK root path")
set(OPENCV_ROOT "" CACHE PATH "OpenCV SDK root path")


find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.8 gstreamer-video-1.0>=1.8 gstreamer-base-1.0>=1.8)


find_package(CUDA REQUIRED cudart cublas curand)
list(APPEND GPU_ARCHS 30 35 37 50 52 60 61 70 75)

MESSAGE("found CUDA Version:${CUDA_VERSION}")

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# Generate PTX for the last architecture
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

# Find OpenCV 
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui dnn PATHS ${OPENCV_ROOT} ${CMAKE_SYSTEM_PREFIX_PATH} PATH_SUFFIXES build share NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui dnn)


# Find TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES include)
if(${TRT_INCLUDE_DIR} MATCHES "TRT_INCLUDE_DIR-NOTFOUND")
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT headers. Please set path using -DTRT_SDK_ROOT")
else()
  MESSAGE(STATUS "Found TensorRT headers at ${TRT_INCLUDE_DIR}")
endif()

find_library(NVINFER_LIBRARY nvinfer HINTS ${TRT_SDK_ROOT}/../lib)
find_library(NVPARSERS_LIBRARY nvparsers HINTS ${TRT_SDK_ROOT}/../lib)
find_library(NVONNXPARSER_LIBRARY nvonnxparser HINTS ${TRT_SDK_ROOT}/../lib)
find_library(NVINFERPLUGIN_LIBRARY nvinfer_plugin HINTS ${TRT_SDK_ROOT}/../lib)

# Add yolo lib as subdir
add_subdirectory(${PROJECT_SOURCE_DIR}/../../lib ${PROJECT_BINARY_DIR}/lib)

include_directories(${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR} ${GST_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../../lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${GST_LIBRARY_DIRS})
link_directories(${TRT_SDK_ROOT}/../lib)

add_library(gstyoloplugin SHARED gstyoloplugin.cpp)
set_target_properties(gstyoloplugin PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(gstyoloplugin yololib gst_detectionmetas ${GST_LIBRARIES} nvinfer_plugin nvinfer nvparsers)

#Install library
install(TARGETS gstyoloplugin LIBRARY DESTINATION "/usr/lib/${ARCHITECTURE}-linux-gnu/gstreamer-1.0/" CONFIGURATIONS Release Debug)