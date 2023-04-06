//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.


// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <chrono>

// include OpenJij
#include <openjij/graph/all.hpp>
#include <openjij/system/all.hpp>
#include <openjij/updater/all.hpp>
#include <openjij/algorithm/all.hpp>
#include <openjij/result/all.hpp>
#include <openjij/utility/schedule_list.hpp>
#include <openjij/utility/union_find.hpp>
#include <openjij/utility/random.hpp>
#include <openjij/utility/gpu/memory.hpp>
#include <openjij/utility/gpu/cublas.hpp>
#include <openjij/sampler/sa_sampler.hpp>

// include Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// include tests
#define TEST_CASE_INDEX 1

#include "testcase.hpp"
#include "polynomial_test.hpp"

#include "sampler/all.hpp"
#include "system/all.hpp"
#include "graph/all.hpp"
#include "result/all.hpp"
#include "utility/all.hpp"

std::int32_t main(std::int32_t argc, char **argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
