cmake_minimum_required(VERSION 3.12.2)
project(openjij)


set(CMAKE_CXX_STANDARD 17)
# Set default compile flags for GCC
if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long -pedantic")
else(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCXX)


# add fPIC option to all object files
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
 

##### Set default behavior #####
set(DEFAULT_USE_OMP Yes)
set(DEFAULT_USE_CUDA Yes)
set(DEFAULT_USE_TEST No)
option(USE_TEST "Use test code" No)


# Use OpenMP as default behavior
if(NOT DEFINED USE_OMP)
    set(USE_OMP ${DEFAULT_USE_OMP})
endif()

if(NOT DEFINED USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(USE_CUDA ${DEFAULT_USE_CUDA})

        #for cuda-memcheck
        # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
        #set CXX compiler to nvcc host compiler in order to avoid linker error
        # set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
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

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

if(USE_OMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        #disable Eigen parallelization
        add_definitions(-DEIGEN_DONT_PARALLELIZE)
        add_definitions(-DUSE_OMP)
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH external)
if(USE_TEST)
    include(${CMAKE_SOURCE_DIR}/external/googletest.cmake)
else()
	message(STATUS "Skip downloding googletest")
endif()
include(${CMAKE_SOURCE_DIR}/external/pybind11.cmake)
include(${CMAKE_SOURCE_DIR}/external/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/external/json.cmake)
include(${CMAKE_SOURCE_DIR}/external/pybind11-json.cmake)

add_subdirectory(src)
add_subdirectory(openjij)
if(USE_TEST)
    add_subdirectory(tests)
endif()
