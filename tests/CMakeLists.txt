cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_FLAGS "-DDEBUGTEST")

add_executable(cxxjij_test EXCLUDE_FROM_ALL cxxtest.cpp)

target_link_libraries(cxxjij_test graph_static algorithm_static system_static gmock_main gtest pthread)
