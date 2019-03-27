#include "classical_ising.h"
#include "../algorithm/sa.h"
#include <cassert>
#include <cmath>

namespace openjij {
	namespace method {

		ClassicalIsing::ClassicalIsing(const graph::Dense<double>& interaction)
			: spins(interaction.gen_spin()), interaction(interaction), urd{0.0, 1.0}{
				//random number generator
				std::random_device rd;
				mt = std::mt19937(rd());
				uid = std::uniform_int_distribution<>{0,(int)spins.size()-1};
			}

		double ClassicalIsing::update(double beta, const std::string& algo){
			double totaldE = 0;
			size_t num_spins = spins.size();

			if(algo == "single_spin_flip" or algo == ""){
				//default updater (single_spin_flip)
				for(size_t i=0; i<num_spins; i++){
					size_t index = uid(mt);
					//do metropolis
					double dE = 0;
					for(auto&& adj_index : interaction.adj_nodes(index)){
						dE += -2 * spins[index] * (index != adj_index ? (interaction.J(index, adj_index) * spins[adj_index]) : interaction.h(index));
					}

					//metropolis
					if(exp(-beta*dE) > urd(mt)){
						spins[index] *= -1;
						totaldE += dE;
					}
				}
			}

			return totaldE;
		}

		void ClassicalIsing::simulated_annealing(double beta_min, double beta_max, double step_length, size_t step_num, const std::string& algo){
			algorithm::SA sa(beta_min, beta_max, step_length, step_num);
			//do simulated annealing
			sa.exec(*this, algo);
		}

		const graph::Spins ClassicalIsing::get_spins() const{
			return spins;
		}
	} // namespace method
} // namespace openjij
