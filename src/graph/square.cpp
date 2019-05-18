#include <algorithm>
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
			Index Square<FloatType>::to_ind(int64_t r, int64_t c) const{
				//row-major matrix
				//assert(-1 <= r && r <= (int64_t)num_row);
				//assert(-1 <= c && c <= (int64_t)num_column);
				if(this->is_err_check()){
					if(!(-1 <= r && r <= (int64_t)num_row))
						throw std::invalid_argument("Square::to_ind: invalid argument of r");
					if(!(-1 <= c && c <= (int64_t)num_column))
						throw std::invalid_argument("Square::to_ind: invalid argument of c");
				}

				return num_column * mod_r(r) + mod_c(c);
			}

		template<typename FloatType>
			RowColumn Square<FloatType>::to_rc(Index ind) const{
				if(this->is_err_check()){
					this->Graph::range_check(ind);
				}
				int64_t r = ind/num_column;
				int64_t c = ind%num_column;
				return std::make_pair(r, c);
			}

		template<typename FloatType>
			Square<FloatType>::Square(size_t num_row, size_t num_column, FloatType init_val, bool err_check)
			: Sparse<FloatType>(num_row*num_column, 4+1, err_check), init_val(init_val), num_row(num_row), num_column(num_column){
				//generate sparse graph
				//assert(num_row >= 1);
				//assert(num_column >= 1);
				if(this->is_err_check()){
					if(!(num_row >= 1))
						throw std::invalid_argument("Square::to_ind: invalid argument of num_row");
					if(!(num_column >= 1))
						throw std::invalid_argument("Square::to_ind: invalid argument of num_column");
				}

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
				//assert(r<num_row);
				//assert(c<num_column);
				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Square::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Square::to_ind: invalid argument of c");
				}

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
						//assert(false);
						return init_val;
				}
			}

		template<typename FloatType>
			const FloatType& Square<FloatType>::J(size_t r, size_t c, Dir dir) const{
				//assert(r<num_row);
				//assert(c<num_column);
				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Square::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Square::to_ind: invalid argument of c");
				}

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
						//assert(false);
						return init_val;
				}
			}

		template<typename FloatType>
			FloatType& Square<FloatType>::h(size_t r, size_t c){
				//assert(r<num_row);
				//assert(c<num_column);
				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Square::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Square::to_ind: invalid argument of c");
				}

				return this->Sparse<FloatType>::h(to_ind(r,c));
			}

		template<typename FloatType>
			const FloatType& Square<FloatType>::h(size_t r, size_t c) const{
				//assert(r<num_row);
				//assert(c<num_column);
				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Square::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Square::to_ind: invalid argument of c");
				}

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
