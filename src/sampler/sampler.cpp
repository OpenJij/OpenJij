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

#include "sampler.h"
#include "../updater/single_spin_flip.h"

namespace openjij{
    namespace sampler{

		void Results::add_spins(Spins& spins, double energy){
			states.push_back(spins);
			energies.push_back(energy);
		}	

        Spins Sampler::random_initialize(int system_size){
            std::random_device rd{};
            std::mt19937 mt(rd());
            std::uniform_int_distribution<int> ram_q(0,1);

            Spins spins(system_size);
            for(int i=0; i < system_size; i++){
                spins[i] = 2*ram_q(mt) - 1;
            }
            return spins;
        }

        std::unique_ptr<updater::Updater> Sampler::select_updater(std::string updater_name){
            return std::unique_ptr<updater::SingleSpinFlip>(new updater::SingleSpinFlip(int_mat));
        }

        // sampling
        void Sampler::sampling(double beta, int burn_out, int observe_num, Results& resu){
            auto updater = select_updater("single_spin_flip");
            Spins spins = random_initialize(int_mat.size());
            for(int i=0; i < burn_out; i++){
                updater->spins_update(spins, beta);
            }
            for(int i=0; i < observe_num; i++){
                updater->spins_update(spins, beta);
                resu.add_spins(spins, calc_energy(spins));
            }
        }

        void Sampler::quantum_sampling(double beta, double gamma, int burn_out, int observe_num, int trotter, Results& resu){
            auto updater = select_updater("single_spin_flip");
            std::vector<Spins> trotterized_spins(trotter, random_initialize(int_mat.size()));
            for(int i=0; i < burn_out; i++){
                updater->quantum_spins_update(trotterized_spins, beta, gamma);
            }

            for(int i=0; i < observe_num; i++){
                updater->quantum_spins_update(trotterized_spins, beta, gamma);
                resu.add_quantum_spins(trotterized_spins);
            }
        }

        // optimization
        void Sampler::simulated_annealing(double beta_min, double beta_max, int step_length, int step_num, int iter, Results& resu){
            Spins spins = random_initialize(int_mat.size());
            double r_beta = pow(beta_max/beta_min, 1.0/static_cast<double>(step_num));
            auto updater = select_updater("single_spin_flip");

            for(int r=0; r < iter; r++){
                double beta = beta_min;
                while(beta <= beta_max){
                    for(int i=0; i < step_length; i++)
                        updater->spins_update(spins, beta);
                    beta *= r_beta;
                }
                resu.add_spins(spins, calc_energy(spins));
            }
        } 

        void Sampler::simulated_quantum_annealing(double beta, double gamma_min, double gamma_max, int trotter, int step_length, int step_num, int iter, Results& resu){
            std::vector<Spins> trotterized_spins(trotter, random_initialize(int_mat.size()));
            double r_g = pow(gamma_min/gamma_max, 1.0/static_cast<double>(step_num));
            auto updater = select_updater("single_spin_flip");

            for(int r=0; r < iter; r++){
                double gamma = gamma_max;
                while(gamma >= gamma_min){
                    for(int i=0; i < step_length; i++)
                        updater->quantum_spins_update(trotterized_spins, beta, gamma);
                    gamma *= r_g;
                }
				int min_index = 0;
				double e_min = calc_energy(trotterized_spins[0]);
				double energy;
				for(int t=1; t < trotter; t++){
					energy = calc_energy(trotterized_spins[t]);
					if (energy < e_min){
						min_index = t;
						e_min = energy;
					}	
				}
                resu.add_spins(trotterized_spins[min_index], e_min);
            }
        }

		double Sampler::calc_energy(Spins& spins) const {
			// calculate energy
			double energy = 0.0;
			for(int i=0; i < int_mat.size()-1; i++){
			   for(int j=i+1; j < int_mat.size(); j++){
				   energy += int_mat(i,j) * spins[i] * spins[j];
			   }
			}
			return energy;
		}

    } // namespace sampler
} // namespace openjij
