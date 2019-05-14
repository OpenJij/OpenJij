set(METHOD_LIBRARY classical_ising.cpp quantum_ising.cpp)

if(USE_CUDA)
	set(METHOD_LIBRARY ${METHOD_LIBRARY} chimera_gpu/chimera_gpu_quantum.cpp chimera_gpu/cuda/kernel.cu)
endif()

add_library(system SHARED ${METHOD_LIBRARY})
add_library(system_static STATIC ${METHOD_LIBRARY})

if(USE_CUDA)
	target_include_directories(system PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	target_include_directories(system_static PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

	set_target_properties(system PROPERTIES
		POSITION_INDEPENDENT_CODE ON
		CUDA_VISIBILITY_PRESET "hidden"
		# CUDA_SEPARABLE_COMPILATION ON
		)
	set_target_properties(system_static PROPERTIES
		POSITION_INDEPENDENT_CODE ON
		CUDA_VISIBILITY_PRESET "hidden"
		# CUDA_SEPARABLE_COMPILATION ON
		)
endif()

target_link_libraries(system PRIVATE algorithm graph)
target_link_libraries(system_static PRIVATE algorithm_static graph_static)

if(USE_CUDA)
	target_link_libraries(system PRIVATE ${CUDA_curand_LIBRARY})
	target_link_libraries(system_static PRIVATE ${CUDA_curand_LIBRARY})
endif()
