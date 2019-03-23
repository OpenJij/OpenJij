add_library(cxxjij MODULE main.cpp)
set_target_properties(cxxjij PROPERTIES
	CXX_VISIBILITY_PRESET "hidden"
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX "${PYTHON_MODULE_PREFIX}"
	SUFFIX "${PYTHON_MODULE_EXTENSION}"
	)

target_link_libraries(cxxjij PRIVATE graph)
target_link_libraries(cxxjij PRIVATE algorithm)
target_link_libraries(cxxjij PRIVATE method)
target_link_libraries(cxxjij PRIVATE pybind11::module)
