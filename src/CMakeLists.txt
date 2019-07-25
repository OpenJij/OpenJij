add_library(cxxjij_header_only INTERFACE)

target_include_directories(cxxjij_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

#for GPU
if(USE_CUDA)
    add_subdirectory(system)
endif()
