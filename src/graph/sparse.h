#pragma once

#include <utility>
#include <map>
#include "graph.h"

namespace openjij {
	namespace graph {

		//forward declaration
		template<typename FloatType>
			class Dense;

		//two-body intereactions with O(1) connectivity
		template<typename FloatType>
			class Sparse : public Graph{
				public:
					using Interactions = std::map<std::pair<Index, Index>, FloatType>;
				private:
					//interactions (the number of intereactions is num_spins*(num_spins+1)/2)
					Interactions m_J;
					//the uppder limit of the number of edges per site
					size_t num_edges;
					//the list of the indices of adjacent nodes
					std::vector<Nodes> list_adj_nodes;
					//add adjacent node
					void set_adj_node(Index from, Index to);

				public:
					Sparse(size_t num_spins, size_t num_edges);
					explicit Sparse(size_t num_spins) : Sparse(num_spins, num_spins){}

					Sparse(const Sparse<FloatType>&) = default;
					Sparse(Sparse<FloatType>&&) = default;

					Sparse(const Dense<FloatType>& obj);
					Sparse(Dense<FloatType>&& obj);

					//list of adjacent nodes
					const Nodes& adj_nodes(Index ind) const;

					size_t get_num_edges() const;

					FloatType calc_energy(const Spins& spins) const;

					FloatType& J(Index i, Index j);
					const FloatType& J(Index i, Index j) const;

					FloatType& h(Index i);
					const FloatType& h(Index i) const;

			};
	} // namespace graph
} // namespace openjij

