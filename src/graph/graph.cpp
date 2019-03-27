#include "graph.h"
#include <cassert>
#include <random>

namespace openjij {
	namespace graph {
		const Spins Graph::gen_spin(bool random_initialize) const{
			//generate spin array
			Spins ret_spin(num_spins);
			if(random_initialize){
				std::random_device rd;
				auto mt = std::mt19937(rd());
				std::uniform_int_distribution<> unif(0, 1);
				for(auto& elem : ret_spin){
					elem = 2*unif(mt)-1;
				}
			}
			else{
				for(auto& elem : ret_spin){
					elem = 1;
				}
			}

			return ret_spin;
		}

		size_t Graph::get_num_spins() const{
			return num_spins;
		}
	} // namespace graph
} // namespace openjij
