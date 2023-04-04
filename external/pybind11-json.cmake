include(FetchContent)

message(CHECK_START "Fetching Pybind11 Json")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(FETCHCONTENT_QUIET OFF)
#### pybind11_json ####
FetchContent_Declare(
    pybind11_json
    GIT_REPOSITORY  https://github.com/pybind/pybind11_json
    GIT_TAG         0.2.13
    GIT_SHALLOW     TRUE
)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(pybind11_json)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
