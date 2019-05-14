#pragma once

#include <vector>
#include <cstddef>
#include <random>

namespace openjij {
	namespace graph {

		//type aliases
		//TODO: Spin must take value +1 or -1
		using Spin = int;
		using Spins = std::vector<Spin>;
		using Index = size_t;

		using Nodes = std::vector<Index>;

		//Abstract graph class
		class Graph{
			private:
				//the number of total spins
				size_t num_spins;
			public:
				explicit Graph(size_t num_spins)
					:num_spins(num_spins){}
				//generate spin array
				const Spins gen_spin(uint_fast32_t seed) const;
				//without seed (use random_device)
				const Spins gen_spin() const;

				size_t get_num_spins() const;
		};
	} // namespace graph 
} // namespace openjij


