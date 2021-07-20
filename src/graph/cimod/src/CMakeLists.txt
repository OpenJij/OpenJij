add_library(cxxcimod_header_only INTERFACE)

target_include_directories(cxxcimod_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_FOUND)
  target_link_libraries(cxxcimod_header_only INTERFACE OpenMP::OpenMP_CXX)
endif()
