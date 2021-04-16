include(FetchContent)

#### pybind11 ####
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY  https://github.com/pybind/pybind11
    GIT_TAG         v2.6.2
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_fetch_POPULATED)
    message(STATUS "Fetch pybind11 for python-binding")
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR})
endif()

set(CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

