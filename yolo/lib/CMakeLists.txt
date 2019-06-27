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
# file(GLOB CXX_SRCS ${CMAKE_SOURCE_DIR}/lib/*.cpp)
# file(GLOB CU_SRCS ${CMAKE_SOURCE_DIR}/lib/*.cu)

file(GLOB CXX_SRCS *.cpp)
file(GLOB CU_SRCS *.cu)

find_package(PkgConfig)
find_package(CUDA)
find_package(OpenCV)

pkg_check_modules(GLIB REQUIRED glib-2.0)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

set_ifndef(TRT_INC_DIR /usr/include/aarch64-linux-gnu)

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 aarch64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 aarch64-linux-gnu)
set_ifndef(NVINFER_PLUGIN_LIB ${_NVINFER_PLUGIN_LIB})

include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TRT_INC_DIR} ${GLIB_INCLUDE_DIRS} ${GLFAGS_INCLUDE_DIRS} )
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

CUDA_COMPILE(CU_OBJS ${CU_SRCS} OPTIONS  "--compiler-options=-fPIC --shared --ptxas-options=-v --use_fast_math -gencode arch=compute_72,code=sm_72")
add_library(yolo-lib SHARED ${CXX_SRCS} ${CU_OBJS})

target_link_libraries(yolo-lib cudart cudnn cublas ${OpenCV_LIBRARIES} ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} gflags stdc++fs dl)