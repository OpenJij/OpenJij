#include <cassert>
#include <random>
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
			Dense<FloatType>::Dense(const Sparse<FloatType>& s)
			: Graph(s.get_num_spins()), m_J(s.get_num_spins()*(s.get_num_spins()+1)/2){
				for(Index i=0; i<get_num_spins(); i++){
					for(Index j=i; j<get_num_spins(); j++){
						J(i, j) = s.J(i, j);
					}
				}
			}

		template<typename FloatType>
			Dense<FloatType>::Dense(Sparse<FloatType>&& s)
			: Graph(s.get_num_spins()), m_J(s.get_num_spins()*(s.get_num_spins()+1)/2){
				for(Index i=0; i<get_num_spins(); i++){
					for(Index j=i; j<get_num_spins(); j++){
						J(i, j) = s.J(i, j);
					}
				}
			}

		template<typename FloatType>
			FloatType Dense<FloatType>::calc_energy(const Spins& spins) const{

				FloatType ret = 0;
				for(Index i=0; i<get_num_spins(); i++){
					for(Index j=i; j<get_num_spins(); j++){
						if(i != j)
							ret += J(i, j) * spins[i] * spins[j];
						else
							ret += h(i) * spins[i];
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
