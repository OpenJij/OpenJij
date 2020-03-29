## include(FetchContent)
## 
## #### nlohmann_json ####
## FetchContent_Declare(
##     nlohmann_json
##     GIT_REPOSITORY  https://github.com/nlohmann/json
##     GIT_TAG         v3.7.3
## )
## 
## FetchContent_GetProperties(nlohmann_json)
## if(NOT nlohmann_json_fetch_POPULATED)
##     message(STATUS "Fetch nlohmann_json for json parser")
##     FetchContent_Populate(nlohmann_json)
##     add_subdirectory(${nlohmann_json_SOURCE_DIR})
## endif()

# Since the git repository of nlohmann/json is huge, we store only a single-include file json.hpp in our project.

add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/external/nlohmann_json)
