#include <cassert>
#include <algorithm>
#include "dense.h"
#include "sparse.h"
#include <stdexcept>

namespace openjij{
	namespace graph{

		template<typename FloatType>
			inline size_t Dense<FloatType>::convert_index(Index i, Index j) const{
				assert(i<=j);
				return get_num_spins()*i-i*(i-1)/2+j-i;
			}

		template<typename FloatType>
			void Dense<FloatType>::set_adj_node(Index from, Index to){
				assert(from < this->get_num_spins());
				assert(to < this->get_num_spins());

				//get adjacent nodes of node "from"
				Nodes& nodes = list_adj_nodes[from];
				//check if the node "to" exists in the nodes 
				if(std::find(nodes.begin(), nodes.end(), to)==nodes.end()){
					//add node
					nodes.push_back(to);
					//add node from "to" to "from"
					set_adj_node(to, from);
				}
			}

		template<typename FloatType>
			Dense<FloatType>::Dense(size_t num_spins)
			: Graph(num_spins), m_J(num_spins*(num_spins+1)/2), list_adj_nodes(num_spins)
			{
				//initialize list_adj_nodes
				for(auto& elem : list_adj_nodes){
					elem.reserve(num_spins); //not resize()
				}
			}


		template<typename FloatType>
			Dense<FloatType>::Dense(const Sparse<FloatType>& s)
			: Graph(s.get_num_spins()), m_J(s.get_num_spins()*(s.get_num_spins()+1)/2), list_adj_nodes(s.get_num_spins()){
				for(Index i=0; i<get_num_spins(); i++){
					for(Index j : s.adj_nodes(i)){
						J(i, j) = s.J(i, j);
					}
				}
			}

		template<typename FloatType>
			Dense<FloatType>::Dense(Sparse<FloatType>&& s)
			: Graph(s.get_num_spins()), m_J(s.get_num_spins()*(s.get_num_spins()+1)/2), list_adj_nodes(s.get_num_spins()){
				for(Index i=0; i<get_num_spins(); i++){
					for(Index j : s.adj_nodes(i)){
						J(i, j) = s.J(i, j);
					}
				}
			}

		template<typename FloatType>
			const Nodes& Dense<FloatType>::adj_nodes(Index ind) const{
				assert(ind < this->get_num_spins());

				return list_adj_nodes[ind];
			}

		template<typename FloatType>
			FloatType Dense<FloatType>::calc_energy(const Spins& spins) const{

				FloatType ret = 0;
				for(size_t ind=0; ind<this->get_num_spins(); ind++){
					for(auto& adj_ind : list_adj_nodes[ind]){
						if(ind != adj_ind)
							ret += (1./2) * this->J(ind, adj_ind) * spins[ind] * spins[adj_ind];
						else
							ret += this->h(ind) * spins[ind];
					}
				}

				return ret;
			}

		template<typename FloatType>
			FloatType& Dense<FloatType>::J(Index i, Index j){
				//check if i and j are smaller than num_spins
				assert(i < get_num_spins());
				assert(j < get_num_spins());
				//i <= j
				//add node if it does not exist
				set_adj_node(i, j);
				return m_J[convert_index(std::min(i, j), std::max(i, j))];
			}

		template<typename FloatType>
			const FloatType& Dense<FloatType>::J(Index i, Index j) const{
				//check if i and j are smaller than num_spins
				assert(i < get_num_spins());
				assert(j < get_num_spins());
				//i <= j
				return m_J[convert_index(std::min(i, j), std::max(i, j))];
			}

		template<typename FloatType>
			FloatType& Dense<FloatType>::h(Index i){
				//check if i is smaller than num_spins
				assert(i < get_num_spins());

				//add node if it does not exist
				set_adj_node(i, i);
				return m_J[convert_index(i, i)];
			}

		template<typename FloatType>
			const FloatType& Dense<FloatType>::h(Index i) const{
				//check if i is smaller than num_spins
				assert(i < get_num_spins());

				return m_J[convert_index(i, i)];
			}

		//instantiations
		template class Dense<double>;
		template class Dense<float>;
	}
} // namespace openjij
