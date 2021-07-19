
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
    $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,Eigen3::Eigen,eigen_lib>
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
    $<$<BOOL:USE_CUDA>:cxxjij_chimera_gpu_kernel>
    $<$<BOOL:USE_CUDA>:${CUDA_LIBRARIES}>
    $<$<BOOL:USE_CUDA>:${CUDA_curand_LIBRARY}>
    $<$<BOOL:USE_CUDA>:${CUDA_CUBLAS_LIBRARIES}>
    nlohmann_json::nlohmann_json
    gmock_main
    gmock
    gtest_main
    gtest
    pthread
)
