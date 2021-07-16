
if (${OPENJIJ_BUILD_SHARED_LIBS})
  add_library(cxxjij SHARED main.cpp)
else()
  add_library(cxxjij MODULE main.cpp)
endif()

if(USE_CUDA)
    set_target_properties(cxxjij PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(cxxjij 
        ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} 
        cxxjij_chimera_gpu_kernel_static
    )
else()
    set_target_properties(cxxjij PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        VISIBLITY_INLINES_HIDDEN ON
        INTERPROCEDURAL_OPTIMIZATION TRUE
        POSITION_INDEPENDENT_CODE ON
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()
target_link_libraries(cxxjij PRIVATE
      pybind11::headers
      $<IF:$<BOOL:${Python_Development.Module_FOUND}>,Python::Module, pybind11::python_link_helper>
      pybind11::lto 
      Eigen3::Eigen
      cxxjij_header_only
      pybind11_json
      nlohmann_json::nlohmann_json
      $<IF:$<BOOL:${OpenMP_FOUND}>,OpenMP::OpenMP_CXX, >
      )
add_custom_target(python
    DEPENDS cxxjij
)
