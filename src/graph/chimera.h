#pragma once 

#include <tuple>
#include "sparse.h"

namespace openjij {
	namespace graph {

		//row, column, in-chimera index (0-7)

		/**
		 * in-chimera index
		 *
		 *      0 - 4
		 *        x
		 *      1 - 5
		 *        x
		 *      2 - 6
		 *        x
		 *      3 - 7
		 *
		 */

		using ChimeraIndex = std::tuple<size_t, size_t, size_t>;

		enum class ChimeraDir{
			PLUS_R,  //(r  , c  , ind) -> (r+1, c  , ind)
			MINUS_R, //(r  , c  , ind) -> (r-1, c  , ind)
			PLUS_C,  //(r  , c  , ind) -> (r  , c+1, ind)
			MINUS_C, //(r  , c  , ind) -> (r  , c-1, ind)
			IN_0or4, //(r  , c  , ind) -> (r  , c  , 0 or 4)
			IN_1or5, //(r  , c  , ind) -> (r  , c  , 1 or 5)
			IN_2or6, //(r  , c  , ind) -> (r  , c  , 2 or 6)
			IN_3or7, //(r  , c  , ind) -> (r  , c  , 3 or 7)
		};

		// chimera lattice graph
		template<typename FloatType=double>
			class Chimera : public Sparse<FloatType>{
				private:
					FloatType init_val;
					size_t mod_r(int64_t a) const;
					size_t mod_c(int64_t a) const;
				protected:
					size_t num_row; // index r
					size_t num_column; // index c
					constexpr static size_t num_in_chimera = 8;

				public:
					Index to_ind(int64_t r, int64_t c, int64_t i) const;
					ChimeraIndex to_rci(Index ind) const;

					Chimera(size_t num_row, size_t num_column, FloatType init_val=0);

					Chimera(const Chimera<FloatType>&) = default;
					Chimera(Chimera<FloatType>&&) = default;

					size_t get_num_row() const{return num_row;}
					size_t get_num_column() const{return num_column;}
					size_t get_num_in_chimera() const{return num_in_chimera;}

					FloatType& J(size_t r, size_t c, size_t i, ChimeraDir dir);

					const FloatType& J(size_t r, size_t c, size_t i, ChimeraDir dir) const;

					FloatType& h(size_t r, size_t c, size_t i);

					const FloatType& h(size_t r, size_t c, size_t i) const;

					Spin& spin(Spins& spins, size_t r, size_t c, size_t i) const;
					
					const Spin& spin(const Spins& spins, size_t r, size_t c, size_t i) const;
			};
	} // namespace graph
} // namespace openjij


