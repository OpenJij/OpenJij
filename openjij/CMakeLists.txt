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


target_link_libraries(cxxjij PRIVATE
    pybind11::module
    eigen_lib
    cxxjij_header_only
)

if(USE_CUDA)
    target_link_libraries(cxxjij PRIVATE cxxjij_chimera_gpu_kernel)
    target_link_libraries(cxxjij PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(cxxjij PRIVATE ${CUDA_curand_LIBRARY})
    target_link_libraries(cxxjij PRIVATE ${CUDA_CUBLAS_LIBRARIES})
endif()

add_custom_target(python
    DEPENDS cxxjij
)
