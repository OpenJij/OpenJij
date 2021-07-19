
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
      pybind11::module
      pybind11::lto   
      pybind11_json
      nlohmann_json::nlohmann_json
      $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,Eigen3::Eigen,eigen_lib>
      $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
      $<$<BOOL:USE_CUDA>:${CUDA_LIBRARIES}>
      $<$<BOOL:USE_CUDA>:${CUDA_curand_LIBRARY}>
      $<$<BOOL:USE_CUDA>:${CUDA_CUBLAS_LIBRARIES}>
      $<$<BOOL:USE_CUDA>:cxxjij_chimera_gpu_kernel_static>
      cxxjij_header_only
)

add_custom_target(python
    DEPENDS cxxjij
)
