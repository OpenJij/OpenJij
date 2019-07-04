include(FetchContent)

#### Eigen ####
FetchContent_Declare(
    eigen
    URL http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
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

