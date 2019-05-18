#include <algorithm>
#include "chimera.h"

namespace openjij {
	namespace graph { 
		template<typename FloatType>
			inline size_t Chimera<FloatType>::mod_r(int64_t a) const{
				//a -> [-1:num_row]
				return (a+num_row)%num_row;
			}
		
		template<typename FloatType>
			inline size_t Chimera<FloatType>::mod_c(int64_t a) const{
				//a -> [-1:num_column]
				return (a+num_column)%num_column;
			}

		template<typename FloatType>
			Index Chimera<FloatType>::to_ind(int64_t r, int64_t c, int64_t i) const{
				//row-major matrix

				//assert(-1 <= r && r <= (int64_t)num_row);
				//assert(-1 <= c && c <= (int64_t)num_column);
				//assert(0 <= i && i < (int64_t)num_in_chimera);
				
				if(this->is_err_check()){
					if(!(-1 <= r && r <= (int64_t)num_row))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of r");
					if(!(-1 <= c && c <= (int64_t)num_column))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of c");
					if(!(0 <= i && i < (int64_t)num_in_chimera))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
				}

				return num_column*num_in_chimera * mod_r(r) + num_in_chimera * mod_c(c) + i;
			}

		template<typename FloatType>
			ChimeraIndex Chimera<FloatType>::to_rci(Index ind) const{
				if(this->is_err_check()){
					this->Graph::range_check(ind);
				}

				int64_t r = ind/(num_column*num_in_chimera);
				ind -= (num_column*num_in_chimera)*r;
				int64_t c = ind/num_in_chimera;
				ind -= (num_in_chimera)*c;
				int64_t i = ind;
				return std::make_tuple(r, c, i);
			}

		template<typename FloatType>
			Chimera<FloatType>::Chimera(size_t num_row, size_t num_column, FloatType init_val, bool err_check)
			: Sparse<FloatType>(num_row*num_column*num_in_chimera, 6+1, err_check), init_val(init_val), num_row(num_row), num_column(num_column){
				//generate sparse graph
				//assert(num_row >= 1);
				//assert(num_column >= 1);
				if(this->is_err_check()){
					if(!(num_row >= 1))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of num_row");
					if(!(num_column >= 1))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of num_column");
				}

				for(size_t r=0; r<num_row; r++){
					for(size_t c=0; c<num_column; c++){
						for(size_t i=0; i<num_in_chimera; i++){
							//open boundary
							if(r > 0 && i < 4){
								//MINUS_R (0<=i<4)
								this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r-1,c,i)) = init_val;
							}
							if(c > 0 && 4 <= i){
								//MINUS_C (4<=i<8)
								this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c-1,i)) = init_val;
							}
							if(r < num_row-1 && i < 4){
								//PLUS_R (0<=i<4)
								this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r+1,c,i)) = init_val;
							}
							if(c < num_column-1 && 4 <= i){
								//PLUS_C (4<=i<8)
								this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c+1,i)) = init_val;
							}

							//inside chimera unit

							this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?4:0)) = init_val;
							this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?5:1)) = init_val;
							this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?6:2)) = init_val;
							this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?7:3)) = init_val;

							//local field
							this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,i)) = init_val;
						}
					}
				}
			}

		template<typename FloatType>
			FloatType& Chimera<FloatType>::J(size_t r, size_t c, size_t i, ChimeraDir dir){
				//assert(r<num_row);
				//assert(c<num_column);
				//assert(i<num_in_chimera);
				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of c");
					if(!(i<num_in_chimera))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
				}

				switch (dir) {
					case ChimeraDir::MINUS_R:
						//assert(i < 4);
						if(this->is_err_check()){
							if(!(i < 4))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind((int64_t)r-1,c,i));
					case ChimeraDir::MINUS_C:
						//assert(4 <= i);
						if(this->is_err_check()){
							if(!(4 <= i))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,(int64_t)c-1,i));
					case ChimeraDir::PLUS_R:
						//assert(i < 4);
						if(this->is_err_check()){
							if(!(i < 4))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind((int64_t)r+1,c,i));
					case ChimeraDir::PLUS_C:
						//assert(4 <= i);
						if(this->is_err_check()){
							if(!(4 <= i))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,(int64_t)c+1,i));

					case ChimeraDir::IN_0or4:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?4:0));
					case ChimeraDir::IN_1or5:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?5:1));
					case ChimeraDir::IN_2or6:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?6:2));
					case ChimeraDir::IN_3or7:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?7:3));

					default:
						//assert(false);
						return init_val;
				}
			}

		template<typename FloatType>
			const FloatType& Chimera<FloatType>::J(size_t r, size_t c, size_t i, ChimeraDir dir) const{
				//assert(r<num_row);
				//assert(c<num_column);
				//assert(i<num_in_chimera);

				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of c");
					if(!(i<num_in_chimera))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
				}

				switch (dir) {
					case ChimeraDir::MINUS_R:
						//assert(i < 4);
						if(this->is_err_check()){
							if(!(i < 4))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind((int64_t)r-1,c,i));
					case ChimeraDir::MINUS_C:
						//assert(4 <= i);
						if(this->is_err_check()){
							if(!(4 <= i))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,(int64_t)c-1,i));
					case ChimeraDir::PLUS_R:
						//assert(i < 4);
						if(this->is_err_check()){
							if(!(i < 4))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind((int64_t)r+1,c,i));
					case ChimeraDir::PLUS_C:
						//assert(4 <= i);
						if(this->is_err_check()){
							if(!(4 <= i))
								throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
						}
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,(int64_t)c+1,i));

					case ChimeraDir::IN_0or4:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?4:0));
					case ChimeraDir::IN_1or5:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?5:1));
					case ChimeraDir::IN_2or6:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?6:2));
					case ChimeraDir::IN_3or7:
						return this->Sparse<FloatType>::J(to_ind(r,c,i), to_ind(r,c,(i < 4)?7:3));

					default:
						//assert(false);
						return init_val;
				}
			}

		template<typename FloatType>
			FloatType& Chimera<FloatType>::h(size_t r, size_t c, size_t i){
				//assert(r<num_row);
				//assert(c<num_column);
				//assert(i<num_in_chimera);
				
				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of c");
					if(!(i<num_in_chimera))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
				}

				return this->Sparse<FloatType>::h(to_ind(r,c,i));
			}

		template<typename FloatType>
			const FloatType& Chimera<FloatType>::h(size_t r, size_t c, size_t i) const{
				//assert(r<num_row);
				//assert(c<num_column);
				//assert(i<num_in_chimera);

				if(this->is_err_check()){
					if(!(r<num_row))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of r");
					if(!(c<num_column))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of c");
					if(!(i<num_in_chimera))
						throw std::invalid_argument("Chimera::to_ind: invalid argument of i");
				}

				return this->Sparse<FloatType>::h(to_ind(r,c,i));

			}
		template<typename FloatType>
			Spin& Chimera<FloatType>::spin(Spins& spins, size_t r, size_t c, size_t i) const{
				return spins[to_ind(r, c, i)];
			}
		
		template<typename FloatType>
			const Spin& Chimera<FloatType>::spin(const Spins& spins, size_t r, size_t c, size_t i) const{
				return spins[to_ind(r, c, i)];
			}

		//instantiation
		template class Chimera<double>;
		template class Chimera<float>;
	} // namespace graph
} // namespace openjij
