cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_FLAGS "-DDEBUGTEST")
set(CMAKE_CXX_FLAGS "-g")

add_executable(cxxjij_test
    EXCLUDE_FROM_ALL
    cxxtest.cpp
)

target_link_libraries(cxxjij_test PRIVATE
    cxxjij_header_only
    eigen_lib
    gmock_main
    gtest
    pthread
)

if(USE_CUDA)
    target_link_libraries(cxxjij_test PRIVATE cxxjij_gpu_kernel)
    target_link_libraries(cxxjij_test PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(cxxjij_test PRIVATE ${CUDA_curand_LIBRARY})
    target_link_libraries(cxxjij_test PRIVATE ${CUDA_CUBLAS_LIBRARIES})
endif()
