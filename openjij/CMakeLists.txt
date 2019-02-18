pybind11_add_module(cxxjij SHARED main.cpp)
target_link_libraries(cxxjij PRIVATE graph dense sparse square chimera clising qising sa sqa)
