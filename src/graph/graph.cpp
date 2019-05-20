#include "graph.h"
#include <cassert>
#include <random>
#include <stdexcept>
#include <sstream>

namespace openjij {
	namespace graph {

		void Graph::range_check(size_t num) const{
			if(num >= this->num_spins){
				//throw out of range exception 
				std::ostringstream stream;
				stream << "Graph::range_check: num (which is " << num <<  ") >= this->num_spins (which is " << this->num_spins << ")";
				throw std::out_of_range(stream.str());
			}
		}

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

		bool Graph::is_err_check() const{
			return err_check;
		}

		void Graph::set_err_check(bool err_check){
			this->err_check = err_check;
		}
	} // namespace graph
} // namespace openjij
