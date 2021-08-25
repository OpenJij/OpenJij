include(FetchContent)

#### pybind11 ####
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY  https://github.com/pybind/pybind11
    GIT_TAG         v2.7.1
)

FetchContent_MakeAvailable(pybind11)
set(PYBIND11_CPP_STANDARD -std=c++17)
#add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} ) 
