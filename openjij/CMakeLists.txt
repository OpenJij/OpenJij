add_library(cxxjij MODULE main.cpp)

if(USE_CUDA)
	target_include_directories(cxxjij PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()


if(USE_CUDA)
	set_target_properties(cxxjij PROPERTIES
		CXX_VISIBILITY_PRESET "hidden"
		INTERPROCEDURAL_OPTIMIZATION TRUE
		PREFIX "${PYTHON_MODULE_PREFIX}"
		SUFFIX "${PYTHON_MODULE_EXTENSION}"
		POSITION_INDEPENDENT_CODE ON
		CUDA_VISIBILITY_PRESET "hidden"
		# CUDA_SEPARABLE_COMPILATION ON
		)
else()
	set_target_properties(cxxjij PROPERTIES
		CXX_VISIBILITY_PRESET "hidden"
		INTERPROCEDURAL_OPTIMIZATION TRUE
		PREFIX "${PYTHON_MODULE_PREFIX}"
		SUFFIX "${PYTHON_MODULE_EXTENSION}"
		)
endif()



target_link_libraries(cxxjij PRIVATE graph_static algorithm_static system_static)
target_link_libraries(cxxjij PRIVATE pybind11::module)

add_custom_target(python
	DEPENDS cxxjij
)
