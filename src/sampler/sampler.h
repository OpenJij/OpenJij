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

#ifndef SAMPLER
#define SAMPLER

#include "../model.h"
#include "../updater/single_spin_flip.h"

#include <string>
#include <memory>

namespace openjij{
    namespace sampler{

        struct Results{
            Results(){}
			std::vector<double> energies;
            std::vector<Spins> states;
            std::vector<std::vector<Spins>> quantum_states;
            void add_spins(Spins& spins, double energy);
            void add_quantum_spins(std::vector<Spins>& q_spins){quantum_states.push_back(q_spins);}
        };

        struct Sampler{
            Sampler(const SquareMatrix<double>& int_mat): int_mat(int_mat){}

            // sampling
            void sampling(double beta, int burn_out, int observe_num, Results& resu);
            void quantum_sampling(double beta, double gamma, int burn_out, int observe_num, int trotter, Results& resu);

            // optimization
            void simulated_annealing(double beta_min, double beta_max, int step_length, int step_num, int iter, Results& resu);
            void simulated_quantum_annealing(double beta, double gamma_min, double gamma_max, int trotter, int step_length, int step_num, int iter, Results& resu);

			double calc_energy(Spins& spins) const;
            protected:
                const SquareMatrix<double> int_mat;
                Spins random_initialize(int system_size);
                std::unique_ptr<updater::Updater> select_updater(std::string updater_name);
        };
    }
}

#endif // SAMPLER
