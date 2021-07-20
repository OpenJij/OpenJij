add_library(cxxjij_header_only INTERFACE)

target_include_directories(cxxjij_header_only INTERFACE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<IF:$<TARGET_EXISTS:Eigen3::Eigen>, ${EIGEN3_INCLUDE_DIR}, ${eigen_SOURCE_DIR}> 
)


#for GPU
if(CUDAToolkit_FOUND)
    add_subdirectory(system)
endif()
if(OpenMP_FOUND)
  target_link_libraries(cxxjij_header_only INTERFACE OpenMP::OpenMP_CXX)
endif()
