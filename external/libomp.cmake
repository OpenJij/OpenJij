include(FetchContent)

#### OpenMP ####
FetchContent_Declare(
    openmp
    GIT_REPOSITORY  https://github.com/llvm/llvm-project
    GIT_TAG         llvmorg-12.0.1
    )

FetchContent_MakeAvailable(openmp)

add_library(eigen_lib INTERFACE)
target_include_directories(eigen_lib INTERFACE ${eigen_SOURCE_DIR})
target_compile_definitions(eigen_lib INTERFACE LIBOMP_ENABLE_SHARED=)
