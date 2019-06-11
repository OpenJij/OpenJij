cmake_minimum_required(VERSION 3.11)
project(openjij)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long -pedantic")

# add fPIC option to all object files
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
include(googletest)
include(pybind11)

##### Set default behavior #####
set(DEFAULT_USE_OMP No)
set(DEFAULT_USE_CUDA Yes)
set(DEFAULT_USE_TEST No)

# Use OpenMP as default behavior
if(NOT DEFINED USE_OMP)
    set(USE_OMP ${DEFAULT_USE_OMP})
endif()

if(NOT DEFINED USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDA)
        set(USE_CUDA ${DEFAULT_USE_CUDA})
    else()
        message(STATUS "No CUDA support")
        set(USE_CUDA No)
    endif()
endif()

if(NOT DEFINED USE_TEST)
    set(USE_TEST ${DEFAULT_USE_TEST})
endif()

message(STATUS "USE_OMP = ${USE_OMP}")
message(STATUS "USE_CUDA = ${USE_CUDA}")
message(STATUS "USE_TEST = ${USE_TEST}")

#TODO: OMP settings

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

add_subdirectory(src)

# shared libs
# add_custom_target(shared
#     DEPENDS graph 
#     DEPENDS algorithm 
#     DEPENDS system
# )

add_subdirectory(openjij)
add_subdirectory(tests)

