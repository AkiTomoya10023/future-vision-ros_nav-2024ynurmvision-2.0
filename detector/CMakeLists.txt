#cmake_minimum_required(VERSION 2.6)
#
#project(yolox)
#
#add_definitions(-std=c++17)
#
#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE Debug)
#
#find_package(CUDA REQUIRED)
#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIRS})
#
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#find_package(TensorRT)
#if (TensorRT_FOUND)
#    message("TensorRT found: ${TensorRT_INCLUDE_DIRS}")
#    include_directories(${TensorRT_INCLUDE_DIRS})
#    target_link_libraries(yolox ${TensorRT_LIBS})
#else(TensorRT_FOUND)
#    # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
#    # tensorrt
#    include_directories(/home/ft/RMCVpackages/TensorRT-8.5.2.2/include)
#    link_directories(/home/ft/RMCVpackages/TensorRT-8.5.2.2/lib)
#endif(TensorRT_FOUND)
## cuda
#include_directories(/usr/local/cuda/include)
#link_directories(usr/local/cuda/lib64)
## cudnn
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)
#
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
#
#include_directories(${PROJECT_SOURCE_DIR})
#aux_source_directory(${PROJECT_SOURCE_DIR} DETECTOR_SRC)
#
#add_executable(yolox ${PROJECT_SOURCE_DIR}/yolox.cpp ${DETECTOR_SRC})
#target_link_libraries(yolox nvinfer)
#target_link_libraries(yolox cudart)
#target_link_libraries(yolox nvonnxparser)
#target_link_libraries(yolox ${OpenCV_LIBS})
#target_link_libraries(yolox ${Eigen3_LIBRARIES})
#
#add_definitions(-O2 -pthread)
#
