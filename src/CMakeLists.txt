add_library(cxxjij_header_only INTERFACE)

target_include_directories(cxxjij_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT EIGEN3_INCLUDE_DIR)
  target_include_directories(cxxjij_header_only INTERFACE ${eigen_SOURCE_DIR})
else() 
  target_include_directories(cxxjij_header_only INTERFACE ${EIGEN3_INCLUDE_DIR})
endif()

#for GPU
if(USE_CUDA)
    add_subdirectory(system)
endif()
if(OpenMP_FOUND)
  target_link_libraries(cxxjij_header_only INTERFACE OpenMP::OpenMP_CXX)
endif()
