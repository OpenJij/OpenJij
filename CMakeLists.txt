cmake_minimum_required(VERSION 3.12.2)
project(openjij)

set(BOOST_ENABLE_CMAKE ON)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(PythonExtensions)
find_package(NumPy)

find_package(BLAS)
find_package(LAPACK)
find_package(MPI)

set(BLA_STATIC ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)

set(CMAKE_VERBOSE_MAKEFILE True)

include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)
  set_property(TARGET ... PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
# Set default compile flags for GCC
if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long -pedantic")
else(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_C_FLAGS_RELEASE -Ofast)
set(CMAKE_CXX_FLAGS_RELEASE  -Ofast)


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
    find_package(CUDAToolkit)
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
    find_package(CUDAToolkit)
    add_definitions(-DUSE_CUDA)
endif()

if(USE_OMP)
    if(Apple)
        find_library(LOMP NAMES libomp.a, libomp.dylib)
        message(STATUS "LOMP  : ${LOMP}")
        link_directories(${LOMP}/lib)
        find_path(OMP NAMES  omp.h)
        message(STATUS "OMP  : ${OMP}")
        set(INCLUDE_DIRECTORIES ${OMP})
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
            set(OpenMP_C_FLAGS " -fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS " -fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES  "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
            set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
            set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
        endif()
    endif()
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
    message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")
    if(OpenMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        #disable Eigen parallelization
        add_definitions(-DEIGEN_DONT_PARALLELIZE)
        add_definitions(-DUSE_OMP)
    endif()
endif()

find_path(CMATH NAMES  math.h)
message(STATUS "CMATH  : ${CMATH}")
set(INCLUDE_DIRECTORIES ${CMATH})

list(APPEND CMAKE_MODULE_PATH external)
if(USE_TEST)
    include(${CMAKE_SOURCE_DIR}/external/googletest.cmake)
else()
	message(STATUS "Skip downloding googletest")
endif()

if(Linux)
	include(${CMAKE_SOURCE_DIR}/external/boost.cmake)
	find_package(Boost)
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
