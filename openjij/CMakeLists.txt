add_library(cxxjij MODULE main.cpp)

if(CUDAToolkit_FOUND)
    set_target_properties(cxxjij PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
        VERSION_INFO "${OPENJIJ_VERSION_INFO}"
    )
else()
    set_target_properties(cxxjij PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        VISIBLITY_INLINES_HIDDEN ON
        INTERPROCEDURAL_OPTIMIZATION TRUE
        POSITION_INDEPENDENT_CODE ON
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        VERSION_INFO "${OPENJIJ_VERSION_INFO}"
    )
endif()
target_link_libraries(cxxjij PRIVATE
      pybind11::module
      pybind11::thin_lto
      pybind11_json
      nlohmann_json::nlohmann_json
      $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,Eigen3::Eigen,openjij-eigen_lib>
      $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
      $<$<TARGET_EXISTS:CUDA::cudart_static>:CUDA::cudart_static>
      $<$<TARGET_EXISTS:CUDA::cublas_static>:CUDA::cublas_static>
      $<$<TARGET_EXISTS:CUDA::curand_static>:CUDA::curand_static>
      $<$<TARGET_EXISTS:cxxjij_chimera_gpu_kernel_static>:cxxjij_chimera_gpu_kernel_static>
      cxxjij_header_only
)

add_custom_target(python
    DEPENDS cxxjij
)
  
if(APPLE AND OpenMP_FOUND)
   add_custom_command(TARGET cxxjij
           POST_BUILD COMMAND
           ${CMAKE_INSTALL_NAME_TOOL} -change libomp.dylib @loader_path/libomp.dylib $<TARGET_FILE:cxxjij>)
endif()
