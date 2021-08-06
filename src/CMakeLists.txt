add_library(cxxjij_header_only INTERFACE)

target_include_directories(cxxjij_header_only INTERFACE 
  ${CMAKE_CURRENT_SOURCE_DIR}
)


#for GPU
if(CUDAToolkit_FOUND)
    add_subdirectory(system)
endif()
target_link_libraries(cxxjij_header_only INTERFACE 
    $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,Eigen3::Eigen,openjij-eigen_lib>
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
)
