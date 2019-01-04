cmake_minimum_required(VERSION 3.11)
project(openjij)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

#### Google test ####
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest
	GIT_TAG release-1.8.1
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
	message(STATUS "Fetch googletest for C++ testing")
	FetchContent_Populate(googletest)
	add_subdirectory(${googletest_SOURCE_DIR} ${googletest_SOURCE_DIR}/googlemock)
endif()

#### for pybind11 ####
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.2.4
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_fetch_POPULATED)
    message(STATUS "Fetch pybind11 for python-binding")
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR})
endif()


add_subdirectory(src)
add_subdirectory(openjij)
add_subdirectory(tests)

