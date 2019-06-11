cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_FLAGS "-DDEBUGTEST")

add_executable(cxxjij_test
    EXCLUDE_FROM_ALL
    cxxtest.cpp
)

target_include_directories(cxxjij_test
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src/system
    ${PROJECT_SOURCE_DIR}/src/algorithm
    ${PROJECT_SOURCE_DIR}/src/updater
    ${PROJECT_SOURCE_DIR}/src/graph
    ${PROJECT_SOURCE_DIR}/src/utility
)

target_link_libraries(cxxjij_test
    gmock_main
    gtest
    pthread
)
