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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
# file(GLOB CXX_SRCS *.cpp ${PROJECT_SOURCE_DIR}/../common/*.cpp)
file(GLOB CXX_SRCS *.cpp)

EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
MESSAGE("Detected ${ARCHITECTURE} architecture")

find_package(PkgConfig)
pkg_check_modules(GLIB REQUIRED glib-2.0)

find_package(CUDA REQUIRED cudart cublas curand)

if(NOT DEFINED CUDA_RUNTIME_ROOT)
  set(CUDA_RUNTIME_ROOT "/usr/local/cuda-${CUDA_VERSION}/targets/${ARCHITECTURE}-linux/include")
endif()

include_directories(${CUDA_RUNTIME_ROOT} ${OpenCV_INCLUDE_DIRS} ${TRT_SDK_ROOT} ${GLIB_INCLUDE_DIRS} ${GLFAGS_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../common ${PROJECT_SOURCE_DIR}/../lib)
link_directories(${TRT_SDK_ROOT}/../lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_library(caffetrt SHARED ${CXX_SRCS})
set_target_properties(caffetrt PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(caffetrt cudart cudnn cublas ${OpenCV_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser nvparsers cudart glog gflags stdc++fs dl)
install(TARGETS caffetrt LIBRARY DESTINATION "/usr/local/lib" CONFIGURATIONS Release Debug)