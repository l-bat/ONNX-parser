project(onnx_parser)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-g -Wall -Werror -std=c++11")

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

add_executable(onnx_parser main.cpp onnx.pb.cc onnx.pb.h)
target_link_libraries(onnx_parser ${PROTOBUF_LIBRARIES})
