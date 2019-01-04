pybind11_add_module(cxxjij SHARED main.cpp)
target_link_libraries(cxxjij PRIVATE updater sampler)
