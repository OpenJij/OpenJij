
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

if(OpenMP_FOUND)
  if(NOT EIGEN3_INCLUDE_DIR)
    target_link_libraries(cxxjij PRIVATE
      pybind11::headers
      Python::Module
      pybind11::lto 
      eigen_lib
      cxxjij_header_only
      pybind11_json
      nlohmann_json::nlohmann_json
      OpenMP::OpenMP_CXX
    )
  else()  
    target_link_libraries(cxxjij PRIVATE
      pybind11::headers
      Python::Module
      pybind11::lto 
      Eigen3::Eigen
      cxxjij_header_only
      pybind11_json
      nlohmann_json::nlohmann_json
      OpenMP::OpenMP_CXX
    )
  endif()
else() 
  if(NOT EIGEN3_INCLUDE_DIR)
    target_link_libraries(cxxjij PRIVATE
      pybind11::headers
      Python::Module
      pybind11::lto 
      eigen_lib
      cxxjij_header_only
      pybind11_json
      nlohmann_json::nlohmann_json
    )
  else()  
    target_link_libraries(cxxjij PRIVATE
      pybind11::headers
      Python::Module
      pybind11::lto 
      Eigen3::Eigen
      cxxjij_header_only
      pybind11_json
      nlohmann_json::nlohmann_json
    )
  endif()
endif()
add_custom_target(python
    DEPENDS cxxjij
)
