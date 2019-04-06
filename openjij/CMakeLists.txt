add_library(cxxjij MODULE main.cpp)
set_target_properties(cxxjij PROPERTIES
	CXX_VISIBILITY_PRESET "hidden"
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX "${PYTHON_MODULE_PREFIX}"
	SUFFIX "${PYTHON_MODULE_EXTENSION}"
	)

target_link_libraries(cxxjij PRIVATE graph_static algorithm_static method_static)
target_link_libraries(cxxjij PRIVATE pybind11::module)

add_custom_target(python
	DEPENDS cxxjij
)
