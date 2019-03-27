#pragma once 

#include <utility>
#include "sparse.h"

namespace openjij {
	namespace graph {

		using RowColumn = std::pair<size_t, size_t>;

		enum class Dir{
			PLUS_R,  //(r  , c  ) -> (r+1, c  )
			MINUS_R, //(r  , c  ) -> (r-1, c  )
			PLUS_C,  //(r  , c  ) -> (r  , c+1)
			MINUS_C, //(r  , c  ) -> (r  , c-1)
		};

		// square lattice graph
		template<typename FloatType=double>
			class Square : public Sparse<FloatType>{
				private:
					FloatType init_val;
					inline size_t mod_r(int64_t a) const;
					inline size_t mod_c(int64_t a) const;

					size_t num_row; // index r
					size_t num_column; // index c

				public:

					Index to_ind(int64_t r, int64_t c) const;
					RowColumn to_rc(Index ind) const;

					Square(size_t num_row, size_t num_column, FloatType init_val=0);

					Square(const Square<FloatType>&) = default;
					Square(Square<FloatType>&&) = default;

					size_t get_num_row(){return num_row;}
					size_t get_num_column(){return num_column;}

					FloatType& J(size_t r, size_t c, Dir dir);

					const FloatType& J(size_t r, size_t c, Dir dir) const;

					FloatType& h(size_t r, size_t c);

					const FloatType& h(size_t r, size_t c) const;

					Spin& spin(Spins& spins, size_t r, size_t c) const;
					
					const Spin& spin(const Spins& spins, size_t r, size_t c) const;
			};
	} // namespace graph
} // namespace openjij

