# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

message(CHECK_START "cxxjij")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

message(STATUS "Build cxxjij")

set(python_module_name cxxjij)

add_library(${python_module_name} MODULE main.cpp)

target_include_directories(${python_module_name} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

pybind11_extension(${python_module_name})

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${python_module_name})
endif()

if(CUDAToolkit_FOUND)
    set_target_properties(${python_module_name} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    set_target_properties(${python_module_name} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

target_compile_definitions(${python_module_name} PRIVATE 
    PYTHON_VERSION_INFO=${PYTHON_VERSION_STRING}
)

target_link_libraries(${python_module_name}  PRIVATE
      cxxjij_header_only
      pybind11::module
      pybind11::thin_lto
      pybind11::windows_extras
      pybind11_json
      nlohmann_json::nlohmann_json
      $<$<TARGET_EXISTS:cxxjij_chimera_gpu_kernel_static>:cxxjij_chimera_gpu_kernel_static>
)

install(TARGETS ${python_module_name} LIBRARY DESTINATION .)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "Finished")
