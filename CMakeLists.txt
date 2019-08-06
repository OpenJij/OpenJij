cmake_minimum_required(VERSION 3.11)
project(openjij)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long -pedantic")

# add fPIC option to all object files
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# list(APPEND CMAKE_MODULE_PATH external)
# include(${CMAKE_SOURCE_DIR}/external/googletest.cmake)
# include(${CMAKE_SOURCE_DIR}/external/pybind11.cmake)
# include(${CMAKE_SOURCE_DIR}/external/eigen.cmake)
 
#### Fetch external projects ####

include(FetchContent)

#### Google test ####
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         release-1.8.1
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    message(STATUS "Fetch googletest for C++ testing")
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_SOURCE_DIR}/googlemock)
endif()

#### pybind11 ####
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY  https://github.com/pybind/pybind11
    GIT_TAG         v2.2.4
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_fetch_POPULATED)
    message(STATUS "Fetch pybind11 for python-binding")
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR})
endif()

set(PYBIND11_CPP_STANDARD -std=c++11)

#### Eigen ####
FetchContent_Declare(
    eigen
    URL http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
    )

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    message(STATUS "Fetch eigen")
    FetchContent_Populate(eigen)
    message(${eigen_SOURCE_DIR})
    add_subdirectory(${eigen_SOURCE_DIR})
endif()

add_library(eigen_lib INTERFACE)
target_include_directories(eigen_lib INTERFACE ${eigen_SOURCE_DIR})



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

        #for cuda-memcheck
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
        # set CXX compiler to nvcc host compiler in order to avoid linker error
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

#TODO: OMP settings

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()


add_subdirectory(src)
add_subdirectory(openjij)
add_subdirectory(tests)

