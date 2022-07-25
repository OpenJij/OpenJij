if(CUDAToolkit_FOUND)
    # include cuda toolkit directory and link with cuda library.
    target_include_directories(cxxjij_header_only INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    add_library(cxxjij_chimera_gpu_kernel_static STATIC "${PROJECT_SOURCE_DIR}\/include\/openjij\/system\/gpu\/chimera_cuda\/kernel.cu")
    add_library(cxxjij_chimera_gpu_kernel SHARED "${PROJECT_SOURCE_DIR}\/include\/openjij\/system\/gpu\/chimera_cuda\/kernel.cu")

 	set_target_properties(cxxjij_chimera_gpu_kernel_static PROPERTIES
        CUDA_STANDARD 11
        POSITION_INDEPENDENT_CODE ON
        #CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
           )

 	set_target_properties(cxxjij_chimera_gpu_kernel PROPERTIES
        CUDA_STANDARD 11
        POSITION_INDEPENDENT_CODE ON
        #CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
 		)

 	target_link_libraries(cxxjij_chimera_gpu_kernel_static cxxjij_header_only CUDA::curand_static CUDA::cublas_static CUDA::cudart_static)
 	target_link_libraries(cxxjij_chimera_gpu_kernel cxxjij_header_only CUDA::curand CUDA::cublas CUDA::cudart)
endif()
