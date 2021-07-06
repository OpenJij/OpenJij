cmake_minimum_required(VERSION 3.15.7)
project(openjij)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

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

if (APPLE)
    find_package(BLAS)
endif()

if(USE_TEST)
       #include(${CMAKE_SOURCE_DIR}/external/googletest.cmake)
       conan_cmake_configure(REQUIRES gtest/[>=1.10.0]
                      GENERATORS cmake_find_package_multi
                      OPTIONS gtest:shared=True)                
else()
	message(STATUS "Skip downloding googletest")
endif()

conan_cmake_configure(REQUIRES pybind11/[>=2.6.2]
                      GENERATORS cmake_find_package_multi)

conan_cmake_configure(REQUIRES eigen/[>=3.3.9]
                      GENERATORS cmake_find_package_multi
                      OPTIONS eigen:MPL2_only=true)

conan_cmake_configure(REQUIRES nlohmann_json/[>=3.9.1]
                      GENERATORS cmake_find_package_multi)


foreach(TYPE ${CMAKE_CONFIGURATION_TYPES})
    conan_cmake_autodetect(settings BUILD_TYPE ${TYPE})
    conan_cmake_install(PATH_OR_REFERENCE .
                        BUILD all
                        REMOTE conan-center
                        SETTINGS ${settings})
endforeach()

 
find_package(pybind11 REQUIRED) 
find_package(eigen REQUIRED) 
find_package(nlohmann_json REQUIRED) 
  
if(USE_TEST)  
  find_package(gtest REQUIRED) 
endif()
list(APPEND CMAKE_MODULE_PATH external)
#include(${CMAKE_SOURCE_DIR}/external/pybind11.cmake)
#include(${CMAKE_SOURCE_DIR}/external/eigen.cmake)
#include(${CMAKE_SOURCE_DIR}/external/json.cmake)
include(${CMAKE_SOURCE_DIR}/external/pybind11-json.cmake)

add_subdirectory(src)
add_subdirectory(openjij)
if(USE_TEST)
    add_subdirectory(tests)
endif()
