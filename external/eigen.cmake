include(FetchContent)

#### Eigen ####
FetchContent_Declare(
    eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen
    GIT_TAG         3.3.9
    )
set(EIGEN_CPP_STANDARD -std=c++11)
FetchContent_MakeAvailable(eigen)

add_library(Eigen3::Eigen INTERFACE)
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MPL2_ONLY)
if (APPLE)
    if(BLAS_FOUND AND LAPACK_FOUND) 
      target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_USE_BLAS=ON)
      target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_USE_LAPACKE=ON)
    endif()
endif()




