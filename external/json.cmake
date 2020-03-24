include(FetchContent)

#### nlohmann_json ####
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY  https://github.com/nlohmann/json
    GIT_TAG         v3.7.3
)

FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_fetch_POPULATED)
    message(STATUS "Fetch nlohmann_json for json parser")
    FetchContent_Populate(nlohmann_json)
    add_subdirectory(${nlohmann_json_SOURCE_DIR})
endif()
