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

#include(GoogleTest)

add_executable(cxxjij_test
    cxxtest/cxxtest.cpp
)

if(ENABLE_COVERAGE AND (NOT MSVC))
  add_coverage(cxxjij_test)
endif()

if(CUDAToolkit_FOUND)
    set_target_properties(cxxjij_test PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_include_directories(cxxjij_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cxxjij_test PRIVATE
    BUILD_TESTING=OFF
)

target_link_libraries(cxxjij_test PRIVATE
    nlohmann_json::nlohmann_json
    cxxjij_header_only
    $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,Eigen3::Eigen,openjij-eigen_lib>
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,gtest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,gtest_main>
    $<IF:$<TARGET_EXISTS:GTest::gmock>,GTest::gmock,gmock>
    $<IF:$<TARGET_EXISTS:GTest::gmock_main>,GTest::gmock_main,gmock_main>
    $<$<TARGET_EXISTS:cxxjij_chimera_gpu_kernel>:cxxjij_chimera_gpu_kernel>
)

gtest_discover_tests(cxxjij_test)
