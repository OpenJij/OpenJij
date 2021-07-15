include(FetchContent)

#### Eigen ####
FetchContent_Declare(
    eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen
    GIT_TAG         3.3.9
    )
set(CMAKE_CXX_STANDARD 11) 
add_definitions(-DEIGEN_MPL2_ONLY)
if (APPLE)
    if(BLAS_FOUND AND LAPACK_FOUND) 
      add_definitions(-DEIGEN_USE_BLAS)
      add_definitions(-DEIGEN_USE_LAPACKE)
    endif()
endif()
FetchContent_MakeAvailable(eigen)

add_library(eigen_lib INTERFACE)
target_include_directories(eigen_lib INTERFACE ${eigen_SOURCE_DIR})
target_compile_definitions(eigen_lib INTERFACE EIGEN_MPL2_ONLY)
if (APPLE)
    if(BLAS_FOUND AND LAPACK_FOUND) 
      target_compile_definitions(eigen_lib INTERFACE EIGEN_USE_BLAS=ON)
      target_compile_definitions(eigen_lib INTERFACE EIGEN_USE_LAPACKE=ON)
    endif()
endif()




