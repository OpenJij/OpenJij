#pragma once

#include <map>
#include <vector>
#include <cassert>

#include "graph.h"

namespace openjij {
	namespace graph {

		//forward declaration
		template<typename FloatType>
			class Sparse;

		//two-body all-to-all interactions
		template<typename FloatType>
			class Dense : public Graph{
				public:
					using Interactions = std::vector<FloatType>;
				private:
					//interactions (the number of intereactions is num_spins*(num_spins+1)/2)
					Interactions m_J;
					inline size_t convert_index(Index i, Index j) const;
				public:
					explicit Dense(size_t num_spins)
						: Graph(num_spins), m_J(num_spins*(num_spins+1)/2){}

					Dense(const Dense<FloatType>&) = default;
					Dense(Dense<FloatType>&&) = default;

					Dense(const Sparse<FloatType>&);
					Dense(Sparse<FloatType>&&);

					FloatType calc_energy(const Spins& spins) const;

					//access to J
					FloatType& J(Index i, Index j);
					const FloatType& J(Index i, Index j) const;

					//access to h (local field)
					FloatType& h(Index i);
					const FloatType& h(Index i) const;

			};
	} // namespace graph 
} // namespace openjij

