# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(openjij
    DESCRIPTION "Framework for the Ising model and QUBO."
    HOMEPAGE_URL "https://www.openjij.org"
)

# Standard includes
include(CheckLanguage)

set(OPENJIJ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
        set(lines "You are building in-place.") 
        message(AUTHOR_WARNING ${lines})
    endif()
    set(OPENJIJ_MAIN_PROJECT ON)
    message(STATUS "CMake ${CMAKE_VERSION}")
    if(CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_EXTENSIONS OFF)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()
else()
  set(OPENJIJ_MAIN_PROJECT OFF)
endif()

# Options
option(OPENJIJ_INSTALL "Install cimod header files?" ${OPENJIJ_MAIN_PROJECT})
option(OPENJIJ_TEST "Build cimod test suite?" ${OPENJIJ_MAIN_PROJECT})
option(OPENJIJ_DOCS "Build cimod docs?" ${OPENJIJ_MAIN_PROJECT})
option(BUILD_DOCS "Enable Doxygen support." OFF)
option(CMAKE_REQUIRE_FAILE "If CMake could not find dependencies, build will faile." OFF)

list(APPEND CMAKE_MODULE_PATH external)
list(APPEND CMAKE_MODULE_PATH cmake)

check_language(C)
check_language(CXX)

if(CMAKE_C_COMPILER)
  enable_language(C)
endif() 
if(CMAKE_CXX_COMPILER)
  enable_language(CXX)
endif() 

if(UNIX AND NOT APPLE)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
      enable_language(Fortran)
  endif()
endif()

if(CMAKE_REQUIRE_FAILE) 
   set(CMAKE_REQUIRE_FIND_PACKAGE_Eigen3 ON) 
   set(CMAKE_REQUIRE_FIND_PACKAGE_nlohmann_json ON) 
   set(CMAKE_REQUIRE_FIND_PACKAGE_pybind11 ON) 
   set(CMAKE_REQUIRE_FIND_PACKAGE_pybind11_json ON) 
   set(CMAKE_REQUIRE_FIND_PACKAGE_GTest ON)
endif()

# detect operating system and host processor
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
message(STATUS "The host processor is ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_GENERATOR = ${CMAKE_GENERATOR}")
message(STATUS "CIBUILDWHEEL = $ENV{CIBUILDWHEEL}")
message(STATUS "CMAKE_REQUIRE_FIND_PACKAGE_Eigen3 = ${CMAKE_REQUIRE_FIND_PACKAGE_Eigen3}")
message(STATUS "CMAKE_REQUIRE_FIND_PACKAGE_nlohmann_json = ${CMAKE_REQUIRE_FIND_PACKAGE_nlohmann_json}")
message(STATUS "CMAKE_REQUIRE_FIND_PACKAGE_pybind11 = ${CMAKE_REQUIRE_FIND_PACKAGE_pybind11}")
message(STATUS "CMAKE_REQUIRE_FIND_PACKAGE_pybind11_json = ${CMAKE_REQUIRE_FIND_PACKAGE_pybind11_json}")
message(STATUS "CMAKE_REQUIRE_FIND_PACKAGE_GTest = ${CMAKE_REQUIRE_FIND_PACKAGE_GTest}")
message(STATUS "Started CMake for ${PROJECT_NAME} \n")

if (APPLE)
    #set(CMAKE_MACOSX_RPATH TRUE)
    message(STATUS "CMAKE_OSX_DEPLOYMENT_TARGET = ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    add_compile_options(-Wno-unused-command-line-argument)
elseif (MSVC)
    add_compile_options(/bigobj /EHsc)
    set(CMAKE_Fortran_COMPILER OFF CACHE BOOL "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(APPLE)
    message(STATUS "No CUDA support")
elseif(NOT DEFINED USE_CUDA)
    check_language(CUDA)    
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      find_package(CUDAToolkit)
      if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Found")
        set(USE_CUDA Yes)
      else() 
        message(STATUS "No CUDA support")
      endif()
    else()
        message(STATUS "No CUDA support")
    endif()
elseif(USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
      message(STATUS "CUDA Found")
      set(USE_CUDA Yes)
    else()
      message(STATUS "No CUDA support")
    endif()
  else()
    message(STATUS "No CUDA support")
  endif()
endif()

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "35;37;50;52;53;60;61;62;70;72;75;80;86;87")
      set(CUDA_ARCHITECTURES "35;37;50;52;53;60;61;62;70;72;75;80;86;87")
      set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES 
          "35;37;50;52;53;60;61;62;70;72;75;80;86;87"
       )
      #set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES all)
    endif()
    message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
    add_definitions(-DUSE_CUDA)
endif()

if(NOT APPLE)
  if($ENV{CIBUILDWHEEL})
    message(STATUS "ENABLE BLAS STATIC")
    set(BLA_STATIC ON CACHE BOOL "" FORCE)
  endif()
endif()

if(CUDAToolkit_FOUND)
  set(BLA_VENDOR NVHPC)
  find_package(BLAS)
  find_package(LAPACK)
  if(NOT (BLAS_FOUND AND LAPACK_FOUND))
     unset(BLA_VENDOR)
     find_package(BLAS)
     find_package(LAPACK)
  endif()
else()
  find_package(BLAS)
  find_package(LAPACK)
endif()

##### Set default behavior #####
set(DEFAULT_USE_OMP ON)

if(NOT APPLE) 
  if(NOT DEFINED USE_OMP) 
    set(USE_OMP ${DEFAULT_USE_OMP})
  endif() 
elseif($ENV{CIBUILDWHEEL}) 
  message(STATUS "OpenMP is disabled on macos for distribution binaries.") 
else() 
  set(USE_OMP ${DEFAULT_USE_OMP})
endif()

if(USE_OMP)
  find_package(OpenMP)
  if(OpenMP_FOUND) 
    message(STATUS "OMP_FOUND = YES")
    message(STATUS "USE_OMP = ON")
    add_definitions(-DUSE_OMP)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
    if(WIN32)
      set(CMAKE_INSTALL_OPENMP_LIBRARIES ON)
    endif()
  else() 
    message(STATUS "OMP_FOUND = NO")
    message(STATUS "USE_OMP = OFF")
    set(USE_OMP OFF)
  endif()
endif()

find_package(Eigen3 CONFIG)

if(EIGEN3_FOUND)
  message(STATUS "Skip Download eigen3")
else() 
  message(STATUS "Downlod eigen3")
  include(external/eigen.cmake)
endif() 

find_package(nlohmann_json CONFIG)
if(TARGET nlohmann_json::nlohmann_json)
  message(STATUS "Skip Download nlohmann_json")
else()
  message(STATUS "Downlod nlohmann_json")
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  include(external/json.cmake)
endif()

find_package(cxxcimod_header_only CONFIG)
if(TARGET cxxcimod_header_only)
  message(STATUS "Skip Download cimod")
else()
  message(STATUS "Fetch cimod")
  include(external/cimod.cmake)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build Python Extension 
if(SKBUILD AND OPENJIJ_MAIN_PROJECT)
  message(STATUS "Build Python Extension.")
  option(WITH_THREAD "Compile in rudimentary thread support" ON)
  option(BUILD_TESTING "Enable CTest support." OFF)
  find_package(Python3 COMPONENTS Interpreter Development NumPy)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  if(NOT DEFINED PYTHON_EXECUTABLE) 
    message(STATUS "NOT DEFINED PYTHON_EXECUTABLE")
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
  endif()
  message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
  execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c
              "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE _tmp_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
  find_package(pybind11 CONFIG REQUIRED)
  
  find_package(pybind11_json CONFIG) 
  if(TARGET pybind11_json)
    message(STATUS "Skip Download pybind11_json")
  else()
    message(STATUS "Downlod pybind11_json")
    include(external/pybind11-json.cmake)
  endif()
  
  add_subdirectory(openjij)
elseif(OPENJIJ_MAIN_PROJECT AND ((${CMAKE_BUILD_TYPE} MATCHES Debug) OR OPENJIJ_TEST))
  include(CTest)
  if(ENABLE_COVERAGE AND (NOT MSVC))
    message(STATUS "Use Codecov")
    find_package(codecov)
  endif()
  
  message(STATUS "Use Googletest") 
  message(STATUS "Downlod Googletest")
  include(external/googletest.cmake)
  
  enable_testing()
  
  add_subdirectory(tests)
  
  if(ENABLE_COVERAGE AND (NOT MSVC))
     coverage_evaluate()
  endif()
endif()

if (OPENJIJ_MAIN_PROJECT AND BUILD_DOCS AND OPENJIJ_DOCS)
    include(cmake/GenerateDocs.cmake)
endif ()
