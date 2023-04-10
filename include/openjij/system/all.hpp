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

// disable eigen -Wdeprecated-copy warning
#include "openjij/utility/disable_eigen_warning.hpp"

#include "openjij/system/classical_ising.hpp"
#include "openjij/system/classical_ising_polynomial.hpp"
#include "openjij/system/continuous_time_ising.hpp"
#include "openjij/system/k_local_polynomial.hpp"
#include "openjij/system/transverse_ising.hpp"
#include "openjij/system/binary_polynomial_sa_system.hpp"
#include "openjij/system/ising_polynomial_sa_system.hpp"

#ifdef USE_CUDA
#include "openjij/system/gpu/chimera_gpu_classical.hpp"
#include "openjij/system/gpu/chimera_gpu_transverse.hpp"
#endif
