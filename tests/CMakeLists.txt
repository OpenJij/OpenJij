cmake_minimum_required(VERSION 3.11)

add_executable(cxxjij_test
    EXCLUDE_FROM_ALL
    cxxtest.cpp
)

if(USE_CUDA)
    set_target_properties(cxxjij_test PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    set_target_properties(cxxjij_test PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

target_link_libraries(cxxjij_test PRIVATE
    cxxjij_header_only
    eigen_lib
    nlohmann_json::nlohmann_json
    gmock_main
    gmock
    gtest_main
    gtest
    pthread
)

if(USE_CUDA)
    target_link_libraries(cxxjij_test PRIVATE cxxjij_chimera_gpu_kernel)
    target_link_libraries(cxxjij_test PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(cxxjij_test PRIVATE ${CUDA_curand_LIBRARY})
    target_link_libraries(cxxjij_test PRIVATE ${CUDA_CUBLAS_LIBRARIES})
endif()
