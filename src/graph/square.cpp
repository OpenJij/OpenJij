#include <algorithm>
#include <cassert>
#include "square.h"

namespace openjij {
	namespace graph { 
		template<typename FloatType>
			inline size_t Square<FloatType>::mod_r(int64_t a) const{
				//a -> [-1:num_row]
				return (a+num_row)%num_row;
			}
		
		template<typename FloatType>
			inline size_t Square<FloatType>::mod_c(int64_t a) const{
				//a -> [-1:num_column]
				return (a+num_column)%num_column;
			}

		template<typename FloatType>
			inline Index Square<FloatType>::to_ind(int64_t r, int64_t c) const{
				//row-major matrix
				assert(-1 <= r && r <= (int64_t)num_row);
				assert(-1 <= c && c <= (int64_t)num_column);

				return num_column * mod_r(r) + mod_c(c);
			}

		template<typename FloatType>
			RowColumn Square<FloatType>::to_rc(Index ind) const{
				int64_t r = ind/num_column;
				int64_t c = ind%num_column;
				return std::make_pair(r, c);
			}

		template<typename FloatType>
			Square<FloatType>::Square(size_t num_row, size_t num_column, FloatType init_val)
			: Sparse<FloatType>(num_row*num_column, 4+1), init_val(init_val), num_row(num_row), num_column(num_column){
				//generate sparse graph
				//TODO: add exception
				assert(num_row >= 1);
				assert(num_column >= 1);

				for(size_t r=0; r<num_row; r++){
					for(size_t c=0; c<num_column; c++){
						//open boundary
						if(r > 0){
							//MINUS_R
							this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r-1,c)) = init_val;
						}
						if(c > 0){
							//MINUS_C
							this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,c-1)) = init_val;
						}
						if(r < num_row-1){
							//PLUS_R
							this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r+1,c)) = init_val;
						}
						if(c < num_column-1){
							//PLUS_C
							this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,c+1)) = init_val;
						}
						//local field
						this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,c)) = init_val;
					}
				}
			}

		template<typename FloatType>
			FloatType& Square<FloatType>::J(size_t r, size_t c, Dir dir){
				//TODO: add exception
				assert(r<num_row);
				assert(c<num_column);

				switch (dir) {
					case Dir::MINUS_R:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind((int64_t)r-1,c));
					case Dir::MINUS_C:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,(int64_t)c-1));
					case Dir::PLUS_R:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind((int64_t)r+1,c));
					case Dir::PLUS_C:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,(int64_t)c+1));
					default:
						//TODO: add exception
						assert(false);
						return init_val;
				}
			}

		template<typename FloatType>
			const FloatType& Square<FloatType>::J(size_t r, size_t c, Dir dir) const{
				//TODO: add exception
				assert(r<num_row);
				assert(c<num_column);

				switch (dir) {
					case Dir::MINUS_R:
						//TODO: access m_J directly to improve performance?
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind((int64_t)r-1,c));
					case Dir::MINUS_C:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,(int64_t)c-1));
					case Dir::PLUS_R:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind((int64_t)r+1,c));
					case Dir::PLUS_C:
						return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,(int64_t)c+1));
					default:
						assert(false);
						return init_val;
				}
			}

		template<typename FloatType>
			FloatType& Square<FloatType>::h(size_t r, size_t c){
				//TODO: add exception
				assert(r<num_row);
				assert(c<num_column);

				//TODO: access m_J directly to improve performance?
				return this->Sparse<FloatType>::h(to_ind(r,c));
			}

		template<typename FloatType>
			const FloatType& Square<FloatType>::h(size_t r, size_t c) const{
				//TODO: add exception
				assert(r<num_row);
				assert(c<num_column);

				//TODO: access m_J directly to improve performance?
				return this->Sparse<FloatType>::h(to_ind(r,c));

			}
		template<typename FloatType>
			Spin& Square<FloatType>::spin(Spins& spins, size_t r, size_t c) const{
				return spins[to_ind(r, c)];
			}
		
		template<typename FloatType>
			const Spin& Square<FloatType>::spin(const Spins& spins, size_t r, size_t c) const{
				return spins[to_ind(r, c)];
			}

		//instantiation
		template class Square<double>;
		template class Square<float>;
	} // namespace model
} // namespace openjij
