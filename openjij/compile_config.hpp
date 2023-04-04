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

#pragma once

#include <openjij/utility/random.hpp>

#ifdef USE_CUDA
#endif

namespace openjij {

/**
 * Note:
 *
 * By default, cxxjij (python implementation of openjij) is installed following
 * the configuration listed below. If you want to use cxxjij with non-default
 * settings (e.g. using mersenne twister, calculating with long double
 * precision, etc), please change the following configuration and recompile with
 * the command:
 *
 * $ python setup.py clean && python setup.py install
 *
 */

/**********************************************************
 default floating point precision on CPU (default: double)
 **********************************************************/
using FloatType = double;
// using FloatType = float;
// using FloatType = long double;

/**********************************************************
 default random number engine on CPU (default: xorshift)
 you may use mersenne twister or your own random number generator.
 **********************************************************/
using RandomEngine = utility::Xorshift;
// using RandomEngine = std::mt19937;
//...

/**********************************************************
 default floating point precision on GPU (default: float)
 **********************************************************/
using GPUFloatType = float;
// using GPUFloatType = double;

#ifdef USE_CUDA
/**********************************************************
 default random number engine on GPU (default: xorwow)
 **********************************************************/
constexpr auto GPURandomEngine = CURAND_RNG_PSEUDO_XORWOW;
// constexpr auto GPURandomEngine = CURAND_RNG_PSEUDO_MT19937;
// constexpr auto GPURandomEngine = CURAND_RNG_PSEUDO_DEFAULT;
//...

#endif

/**********************************************************
 default row x column x trotter size in each block in GPU
 This setting will be used in Chimera GPU.
 **********************************************************/

// note that the size of sharedmem must be smaller than 64kB.
constexpr std::size_t BLOCK_ROW = 2;  // row
constexpr std::size_t BLOCK_COL = 2;  // column
constexpr std::size_t BLOCK_TROT = 1; // trotter size (should not be changed)

} // namespace openjij
