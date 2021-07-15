include(FetchContent)

#### pybind11_json ####
FetchContent_Declare(
    pybind11_json
    GIT_REPOSITORY  https://github.com/pybind/pybind11_json
    GIT_TAG         0.2.11 
)

FetchContent_MakeAvailable(pybind11_json)
