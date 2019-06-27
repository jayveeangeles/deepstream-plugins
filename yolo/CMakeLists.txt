cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(YOLOv3_TRT LANGUAGES CXX C CUDA)
# project(YOLOv3_TRT LANGUAGES CXX C)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
# Set module name here. MUST MATCH the module name specified in the .cpp
set_ifndef(PLUGIN_MODULE_NAME YoloV3)
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)
# pybind11 defaults to c++14.
set(PYBIND11_CPP_STANDARD -std=c++11)

set_ifndef(PYBIND11_DIR $ENV{HOME}/pybind11/)
set_ifndef(CUDA_VERSION 10.0)
set_ifndef(CUDA_ROOT /usr/local/cuda-${CUDA_VERSION})
set_ifndef(CUDNN_ROOT ${CUDA_ROOT})
set_ifndef(PYTHON_ROOT /usr/include)
set_ifndef(TRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
set_ifndef(TRT_INC_DIR /usr/include/aarch64-linux-gnu)

# Find dependencies.
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

find_library(_CUDA_LIBRARIES cudart HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUDA_LIBRARIES ${_CUDA_LIBRARIES})

find_library(_TRT_INC_DIR NvInfer.h HINTS ${TRT_INC_DIR} PATH_SUFFIXES include aarch64-linux-gnu)
set_ifndef(TRT_INC_DIR ${_TRT_INC_DIR})

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 aarch64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVPARSERS_LIB nvparsers HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 aarch64-linux-gnu)
set_ifndef(NVPARSERS_LIB ${_NVPARSERS_LIB})

find_library(_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 aarch64-linux-gnu)
set_ifndef(NVINFER_PLUGIN_LIB ${_NVINFER_PLUGIN_LIB})

find_path(_PYTHON3_INC_DIR Python.h HINTS ${PYTHON_ROOT} PATH_SUFFIXES python3.7 python3.6 python3.5 python3.4)
set_ifndef(PYTHON3_INC_DIR ${_PYTHON3_INC_DIR})

find_package(CUDA)
find_package(PkgConfig)

# -------- BUILDING --------

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add include directories
include_directories(${TRT_INC_DIR} ${CUDA_INC_DIR} ${OpenCV_INCLUDE_DIRS} ${PYBIND11_DIR}/include/ ${CMAKE_SOURCE_DIR}/lib/)

# Add this so we can retrieve pybind11_add_module.
add_subdirectory(${PYBIND11_DIR} ${CMAKE_BINARY_DIR}/pybind11)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib)

file(GLOB_RECURSE PLUGIN_SOURCE_FILES ${CMAKE_SOURCE_DIR}/python/*.cpp)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# file(GLOB LIB_CPP_FILES ${CMAKE_SOURCE_DIR}/lib/*.cpp)
# file(GLOB LIB_CU_FILES ${CMAKE_SOURCE_DIR}/lib/*.cu)

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options=-fPIC --shared --ptxas-options=-v --use_fast_math -gencode arch=compute_72,code=sm_72" )
# CUDA_COMPILE(CU_OBJS ${LIB_CU_FILES})

# add_library(kernel SHARED ${LIB_CPP_FILES} ${CU_OBJS})

# set_property(TARGET kernel PROPERTY COMPILE_FLAGS "--compiler-options=-fPIC --shared --ptxas-options=-v --use_fast_math -gencode arch=compute_72,code=sm_72")

# Bindings library. The module name MUST MATCH the module name specified in the .cpp
if(PYTHON3_INC_DIR AND NOT (${PYTHON3_INC_DIR} STREQUAL "None"))
    # pybind11_add_module(${PLUGIN_MODULE_NAME} SHARED THIN_LTO ${PLUGIN_SOURCE_FILES})
    # target_include_directories(${PLUGIN_MODULE_NAME} BEFORE PUBLIC ${PYTHON3_INC_DIR})
    # target_link_libraries(${PLUGIN_MODULE_NAME} PRIVATE ${CUDA_LIBRARIES} ${NVINFER_LIB} ${NVPARSERS_LIB} ${NVINFER_PLUGIN_LIB} yolo-lib)

    pybind11_add_module(${PLUGIN_MODULE_NAME} SHARED THIN_LTO ${PLUGIN_SOURCE_FILES})
    target_include_directories(${PLUGIN_MODULE_NAME} BEFORE PUBLIC ${PYTHON3_INC_DIR})
    target_link_libraries(${PLUGIN_MODULE_NAME} PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} ${NVINFER_LIB} ${NVPARSERS_LIB} ${NVINFER_PLUGIN_LIB} yolo-lib)

    # pybind11_add_module(${PROCESSORS_MODULE_NAME} SHARED THIN_LTO ${PROCESSORS_SOURCE_FILES})
    # target_include_directories(${PROCESSORS_MODULE_NAME} BEFORE PUBLIC ${PYTHON3_INC_DIR})
    # target_link_libraries(${PROCESSORS_MODULE_NAME})
endif()