add_library(cxxjij MODULE main.cpp)

if(USE_CUDA)
    set_target_properties(cxxjij PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    set_target_properties(cxxjij PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()


target_link_libraries(cxxjij
    pybind11::module
    eigen_lib
    cxxjij_header_only
    pybind11_json
    nlohmann_json::nlohmann_json
)

if(USE_CUDA)
    target_link_libraries(cxxjij 
        ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} 
        cxxjij_chimera_gpu_kernel_static)
endif()

add_custom_target(python
    DEPENDS cxxjij
)
