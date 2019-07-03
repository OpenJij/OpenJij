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

#ifndef OPENJIJ_SYSTEM_CLASSICAL_ISING_HPP__
#define OPENJIJ_SYSTEM_CLASSICAL_ISING_HPP__

#include "../graph/dense.hpp"

namespace openjij {
    namespace system {

        /**
         * Classical Ising structure
         *
         * @tparam GraphType type of graph (Dense, Sparse, ...)
         */
        template<typename GraphType>
            struct ClassicalIsing {
                /**
                 * Constructor to initialize spin and interaction
                 *
                 * @param spin spin array
                 * @param interaction interactions
                 */
                ClassicalIsing(const graph::Spins& spin, const GraphType& interaction)
                    : spin{spin}, interaction{interaction} {
                    }

                /**
                 * Constructor to initialize spin and interaction
                 * spins are initialized randomly.
                 *
                 * @param interaction interactions
                 */
                ClassicalIsing(const GraphType& interaction)
                    : spin{interaction.gen_spin()}, interaction{interaction} {
                    }

                graph::Spins spin;
                GraphType interaction;
            };
    } // namespace system
} // namespace openjij

#endif
