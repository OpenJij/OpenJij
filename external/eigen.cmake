include(FetchContent)

#### Eigen ####
FetchContent_Declare(
    eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen
    GIT_TAG         3.3.9
    )

set(EIGEN_MPL2_ONLY ON)
if (APPLE)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    if(BLAS_FOUND AND LAPACK_FOUND) 
      set(EIGEN_USE_BLAS ON) 
      set(EIGEN_USE_LAPACKE ON)
    endif()
endif()
FetchContent_MakeAvailable(eigen)


