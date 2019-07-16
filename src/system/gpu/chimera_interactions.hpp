//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef OPENJIJ_SYSTEM_CHIMERA_INTERACTIONS_HPP__
#define OPENJIJ_SYSTEM_CHIMERA_INTERACTIONS_HPP__

#ifdef USE_CUDA

#include <utility/gpu/memory.hpp>

namespace openjij {
    namespace system {

        /**
         * @brief chimera interactions
         *
         * @tparam FloatType
         */
        template<typename FloatType>
        struct ChimeraInteractions{
            utility::cuda::unique_dev_ptr<FloatType[]> J_out_p;
            utility::cuda::unique_dev_ptr<FloatType[]> J_out_n;
            utility::cuda::unique_dev_ptr<FloatType[]> J_in_04;
            utility::cuda::unique_dev_ptr<FloatType[]> J_in_15;
            utility::cuda::unique_dev_ptr<FloatType[]> J_in_26;
            utility::cuda::unique_dev_ptr<FloatType[]> J_in_37;
            utility::cuda::unique_dev_ptr<FloatType[]> h;

            ChimeraInteractions(std::size_t n)
            : J_out_p(utility::cuda::make_dev_unique<FloatType[]>(n)),
            J_out_n(utility::cuda::make_dev_unique<FloatType[]>(n)),
            J_in_04(utility::cuda::make_dev_unique<FloatType[]>(n)),
            J_in_15(utility::cuda::make_dev_unique<FloatType[]>(n)),
            J_in_26(utility::cuda::make_dev_unique<FloatType[]>(n)),
            J_in_37(utility::cuda::make_dev_unique<FloatType[]>(n)),
            h(utility::cuda::make_dev_unique<FloatType[]>(n)){
            }
        };
    } // namespace system
} // namespace openjij

#endif
#endif
