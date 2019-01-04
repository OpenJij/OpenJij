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

#ifndef SINGLESPINFLIP
#define SINGLESPINFLIP

#include "updater.h"

#include <random>

namespace openjij{
    namespace updater{
        struct SingleSpinFlip: Updater{
            SingleSpinFlip(const SquareMatrix<double>& int_mat): Updater(int_mat){
                std::random_device rd{};
                mt = std::mt19937(rd());
            }

            SingleSpinFlip(const SquareMatrix<double>& int_mat, int seed): Updater(int_mat), mt(std::mt19937(seed)){}

            virtual void spins_update(Spins& spins, double beta) override;
            virtual void quantum_spins_update(std::vector<Spins>& trotterized_spins, double beta, double gamma) override;
            #ifdef TEST
            public:
            #else
            protected:
            #endif // TEST
                void metropolis_update(Spins& spins, double beta, int flip, double e_bias);
                std::mt19937 mt;
                std::uniform_real_distribution<double> urd{0.0, 1.0};
        };
    }
}

#endif // SINGLESPINFLIP