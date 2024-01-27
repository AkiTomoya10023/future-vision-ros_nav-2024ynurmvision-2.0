cmake_minimum_required(VERSION 2.6)

project(daheng_camera)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/inc)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(daheng_camera ${PROJECT_SOURCE_DIR}/camera_task.cpp)
target_link_libraries(daheng_camera ${OpenCV_LIBS})
target_link_libraries(daheng_camera ${Eigen3_LIBRARIES})
target_link_libraries(daheng_camera -lgxiapi -lpthread)

add_definitions(-O2 -pthread)

