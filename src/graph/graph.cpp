#include "graph.h"
#include <cassert>
#include <random>

namespace openjij {
	namespace graph {
		const Spins Graph::gen_spin(uint_fast32_t seed) const{
			//generate spin array
			Spins ret_spin(num_spins);
			auto mt = std::mt19937(seed);
			std::uniform_int_distribution<> unif(0, 1);
			for(auto& elem : ret_spin){
				elem = 2*unif(mt)-1;
			}

			return ret_spin;
		}

		//without seed
		const Spins Graph::gen_spin() const{
			return gen_spin(std::random_device()());
		}

		size_t Graph::get_num_spins() const{
			return num_spins;
		}
	} // namespace graph
} // namespace openjij
