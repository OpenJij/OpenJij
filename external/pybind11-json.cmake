include(FetchContent)

#### pybind11_json ####
FetchContent_Declare(
    pybind11_json
    GIT_REPOSITORY  https://github.com/pybind/pybind11_json
    GIT_TAG         0.2.4
)

FetchContent_GetProperties(pybind11_json)
if(NOT pybind11_json_fetch_POPULATED)
    message(STATUS "Fetch pybind11_json for python-json-binding")
    FetchContent_Populate(pybind11_json)
    add_subdirectory(${pybind11_json_SOURCE_DIR})
endif()
