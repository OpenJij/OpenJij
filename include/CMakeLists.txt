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

message(CHECK_START "cxxjij_header_only")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

add_library(cxxjij_header_only INTERFACE)

#target_precompile_headers(cxxcimod_header_only INTERFACE
#    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/all.hpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/graph/all.hpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/result/all.hpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/system/all.hpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/updater/all.hpp
#) 

#for GPU
if(CUDAToolkit_FOUND)
    add_subdirectory(openjij/system)
endif()

target_include_directories(cxxjij_header_only INTERFACE 
  $<BUILD_INTERFACE:${OPENJIJ_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(cxxjij_header_only INTERFACE 
    cxxcimod_header_only
    nlohmann_json::nlohmann_json
    $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,Eigen3::Eigen,openjij-eigen_lib>
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
    $<$<TARGET_EXISTS:BLAS::BLAS>:BLAS::BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:LAPACK::LAPACK>
)

target_compile_definitions(cxxjij_header_only INTERFACE 
    EIGEN_MPL2_ONLY
    BUILD_TESTING=OFF
    TEST_LIB=OFF
    EIGEN_BUILD_PKGCONFIG=OFF
    EIGEN_BUILD_DOC=OFF
    EIGEN_DOC_USE_MATHJAX=OFF 
    EIGEN_BUILD_TESTING=OFF 
    EIGEN_TEST_NOQT=OFF 
    EIGEN_LEAVE_TEST_IN_ALL_TARGET=OFF 
    $<$<TARGET_EXISTS:BLAS::BLAS>:EIGEN_USE_BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:EIGEN_USE_LAPACKE>
    $<$<CXX_COMPILER_ID:MSVC>:EIGEN_Fortran_COMPILER_WORKS=OFF>   
)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "Finished")
