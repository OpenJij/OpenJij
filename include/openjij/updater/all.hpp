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

// disable Eigen warning
#include "openjij/utility/disable_eigen_warning.hpp"

#include "openjij/updater/continuous_time_swendsen_wang.hpp"
#include "openjij/updater/k_local.hpp"
#include "openjij/updater/single_spin_flip.hpp"
#include "openjij/updater/swendsen_wang.hpp"

#ifdef USE_CUDA
#include "openjij/updater/gpu.hpp"
#endif
