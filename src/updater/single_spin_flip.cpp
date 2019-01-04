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

#include "single_spin_flip.h"

namespace openjij{
    namespace updater{
        void SingleSpinFlip::spins_update(Spins& spins, double beta){
            for(int flip=0; flip < N; flip++)
                metropolis_update(spins, beta, flip, 0.0);
        }
        void SingleSpinFlip::metropolis_update(Spins& spins, double beta, int flip, double e_bias){
            double de = e_bias;
            for(int adj: adj_mat[flip])
                de += int_mat(flip, adj) * spins[adj];
            
            de = -2.0 * (de + int_mat(flip, flip)) * spins[flip];

            if (de < 0){
                spins[flip] *= -1;
                return;
            }
            double logu = log(urd(mt));
            if (logu < -beta * de){
                spins[flip] *= -1;
                return;
            }
        }

        void SingleSpinFlip::quantum_spins_update(std::vector<Spins>& trotterized_spins, double beta, double gamma){
            int trotter = trotterized_spins.size();
            double beta_eff = beta/trotter; 
            for(int t=0; t < trotter; t++){
                int tp1 = (t+1)%trotter;
                int tm1 = t == 0 ? trotter-1 : t-1;
                for(int flip=0; flip < N; flip++){
                    double trotter_energy = 1.0/(beta_eff*2.0) * log(tanh(beta_eff * gamma)) * (trotterized_spins[tp1][flip] + trotterized_spins[tm1][flip]);
                    metropolis_update(trotterized_spins[t], beta_eff, flip, trotter_energy);
                }
            }
        }
    } // namespace updater
} // namespace openjij
