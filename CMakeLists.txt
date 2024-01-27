cmake_minimum_required(VERSION 2.6)

project(autoaim)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(TensorRT)
if (TensorRT_FOUND)
    message("TensorRT found: ${TensorRT_INCLUDE_DIRS}")
    include_directories(${TensorRT_INCLUDE_DIRS})
    target_link_libraries(autoaim ${TensorRT_LIBS})
else(TensorRT_FOUND)
    # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
    # tensorrt
    include_directories(/home/dji/TensorRT-8.6.1.6/include)
    link_directories(/home/dji/TensorRT-8.6.1.6/lib)
endif(TensorRT_FOUND)
# cuda
include_directories(/usr/local/cuda/include)
link_directories(usr/local/cuda/lib64)
# cudnn
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread")

#include_directories(${PROJECT_SOURCE_DIR})
#aux_source_directory(${PROJECT_SOURCE_DIR} DETECTOR_SRC)

include_directories(detector)
aux_source_directory(detector DETECTOR_SRC)

include_directories(tracker/coordsolver tracker/general tracker/predictor)
aux_source_directory(tracker/coordsolver COORD_SRC)
aux_source_directory(tracker/general GENERAL_SRC)
aux_source_directory(tracker/predictor PRE_SRC)

include_directories(thread)
aux_source_directory(thread THREAD_SRC)

include_directories(camera camera/inc)
aux_source_directory(camera CAMERA_SRC)

include_directories(serial)
aux_source_directory(serial SERIAL_SRC)

add_executable(autoaim ${PROJECT_SOURCE_DIR}/autoaim.cpp ${DETECTOR_SRC} ${TRACKER_SRC} ${COORD_SRC} ${GENERAL_SRC} ${PRE_SRC} ${THREAD_SRC} ${SERIAL_SRC} ${CAMERA_SRC})
target_link_libraries(autoaim nvinfer)
target_link_libraries(autoaim cudart)
target_link_libraries(autoaim nvonnxparser)
target_link_libraries(autoaim ${OpenCV_LIBS})
target_link_libraries(autoaim ${Eigen3_LIBRARIES})
target_link_libraries(autoaim yaml-cpp)
target_link_libraries(autoaim fmt::fmt)
target_link_libraries(autoaim -lgxiapi)

add_definitions(-O0)

