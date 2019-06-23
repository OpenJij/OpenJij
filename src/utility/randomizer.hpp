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

#ifndef OPENJIJ_UTILITY_RANDOMIZER_HPP__
#define OPENJIJ_UTILITY_RANDOMIZER_HPP__

#include <random>

namespace openjij {
    namespace utility {
        template <typename RandomNumberGenerator = std::mt19937>
        struct Randomizer {
            explicit Randomizer(const size_t seed) noexcept
                : _mt{seed} {
            }

            template<typename IntType = int>
            IntType generate_uniform_int_distribution(IntType min, IntType max) {
                std::uniform_int_distribution<IntType> uid(min, max);

                return uid(_mt);
            }

            template<typename RealType = double>
            RealType generate_uniform_real_distribution(RealType min, RealType max) {
                std::uniform_real_distribution<RealType> urd(min, max);

                return urd(_mt);
            }

            private:
            RandomNumberGenerator _mt;
        };

        using RandomizerMT32 = Randomizer<std::mt19937>;
        using RandomizerMT64 = Randomizer<std::mt19937_64>;
    } // namespace utility
} // namespace openjij
#endif
