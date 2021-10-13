cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(onnx_yolov4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR})
#set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)
#file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

message("The runtime libraries are included in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message("The library files are included in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

find_package(CUDA)
find_package(OpenCV REQUIRED)
message(" -- CUDA and Opencv Found ")
message(" -- opencv_version  "${OpenCV_VERSION})

#include_directories("/home/suhyung/TensorRT/TensorRT-7.2.3.4/samples/common")
#include_directories("/home/suhyung/TensorRT/TensorRT-7.2.3.4/include")
include_directories("/workspace/tensorrt/samples/common")
include_directories("/usr/include/x86_64-linux-gnu/")



set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};--disable-warnings;
        -O3
	    -gencode arch=compute_61,code=sm_61
        )


file(GLOB sources src/*.cu src/*.cpp src/*.hpp util/*.cpp util/cuda/*.cu include/*.h)
file(GLOB includes util/*.h util/cuda/*.h)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_INCLUDE_DIR}/util)
include_directories(${PROJECT_BINARY_DIR}/util)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

cuda_add_library(inferLib SHARED ${sources})

#target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so)
#target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvinfer.so)
#target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so)
#target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvparsers.so)
#target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvonnxparser.so)

#target_link_libraries(inferLib /home/suhyung/TensorRT/TensorRT-5.0.2.6/lib/libnvcaffe_parser.so)
#target_link_libraries(inferLib /home/suhyung/TensorRT/TensorRT-7.2.3.4/lib/libnvinfer.so)
#target_link_libraries(inferLib /home/suhyung/TensorRT/TensorRT-7.2.3.4/lib/libnvinfer_plugin.so)
#target_link_libraries(inferLib /home/suhyung/TensorRT/TensorRT-7.2.3.4/lib/libnvparsers.so)
#target_link_libraries(inferLib /home/suhyung/TensorRT/TensorRT-7.2.3.4/lib/libnvonnxparser.so)



target_link_libraries(inferLib /usr/lib/x86_64-linux-gnu/libnvinfer.so)
target_link_libraries(inferLib /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so)
target_link_libraries(inferLib /usr/lib/x86_64-linux-gnu/libnvparsers.so)
target_link_libraries(inferLib /usr/lib/x86_64-linux-gnu/libnvonnxparser.so)


# transfer all headers to the include directory
foreach(include ${includes})
    message("-- Copying ${include}")
    configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

## install
foreach(include ${includes})
    install(FILES "${include}" DESTINATION include/inferLib)
endforeach()

add_executable(onnx_yolov4 ${sources} )
#target_link_libraries(mobileNet libnvonnxparser.so ${OpenCV_LIBS})
target_link_libraries(onnx_yolov4 inferLib ${OpenCV_LIBS})
