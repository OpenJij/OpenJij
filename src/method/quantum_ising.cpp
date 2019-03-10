#include "quantum_ising.h"
#include "../algorithm/sqa.h"
#include <cassert>
#include <cmath>

namespace openjij {
	namespace method {

		inline size_t QuantumIsing::mod_t(int64_t a) const{
			//a -> [-1:num_trotter_slices]
			//return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
			return (a+spins.size())%spins.size();
		}

		QuantumIsing::QuantumIsing(const graph::Dense<double>& interaction, size_t num_trotter_slices)
			:spins(num_trotter_slices), interaction(interaction), urd{0.0, 1.0}{
				//TODO: add exception
				for(auto& elem : spins){
					elem = interaction.gen_spin();

					//random number generators
					std::random_device rd;
					mt = std::mt19937(rd());
					uid = std::uniform_int_distribution<>{0, (int)elem.size()-1};
					uid_trotter = std::uniform_int_distribution<>{0, (int)num_trotter_slices-1};
				}
				assert(spins.size() != 0 and spins[0].size() != 0);
			}

		double QuantumIsing::update(double beta, double gamma, const std::string& algo){
			double totaldE = 0;
			size_t num_classical_spins = spins[0].size();
			size_t num_trotter_slices = spins.size();

			if(algo == "single_spin_flip" or algo == ""){
				//default updater (single_spin_flip)
				for(size_t i=0; i<num_classical_spins*num_trotter_slices; i++){
					size_t index_trot = uid_trotter(mt);
					size_t index = uid(mt);
					//do metropolis
					double dE = 0;
					//adjacent nodes
					for(auto&& adj_index : interaction.adj_nodes(index)){
						dE += -2 * (beta/num_trotter_slices) * spins[index_trot][index] * (index != adj_index ? (interaction.J(index, adj_index) * spins[index_trot][adj_index]) : interaction.h(index));
					}

					//trotter direction
					dE += -2 * (1/2.) * log(tanh(beta*gamma/num_trotter_slices)) * spins[index_trot][index]*(spins[mod_t((int64_t)index_trot+1)][index] + spins[mod_t((int64_t)index_trot-1)][index]);


					//metropolis 
					if(exp(-dE) > urd(mt)){
						spins[index_trot][index] *= -1;
						totaldE += dE;
					}

				}
			}

			return totaldE;
		}

		void QuantumIsing::simulated_quantum_annealing(double beta, double gamma_min, double gamma_max, double step_length, size_t step_num, const std::string& algo){
			algorithm::SQA sqa(beta, gamma_min, gamma_max, step_length, step_num);
			//do simulated quantum annealing
			sqa.exec(*this, algo);
		}

		TrotterSpins QuantumIsing::get_spins() const{
			return this->spins;
		}

	} // namespace method
} // namespace openjij
