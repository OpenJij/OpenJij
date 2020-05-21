include(FetchContent)

#### Eigen ####
FetchContent_Declare(
    eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen
    GIT_TAG         3.3.7
    )

FetchContent_GetProperties(eigen)

if(NOT eigen_POPULATED)
    message(STATUS "Fetch eigen")
    FetchContent_Populate(eigen)
    message(${eigen_SOURCE_DIR})
    add_subdirectory(${eigen_SOURCE_DIR})
endif()

add_library(eigen_lib INTERFACE)
target_include_directories(eigen_lib INTERFACE ${eigen_SOURCE_DIR})

