//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef OPENJIJ_GRAPH_SQUARE_HPP__
#define OPENJIJ_GRAPH_SQUARE_HPP__

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "sparse.hpp"

namespace openjij {
    namespace graph {


        /**
         * Row x Column type
         */
        using RowColumn = std::pair<std::size_t, std::size_t>;

        /**
         * direction enum class
         */
        enum class Dir{

            /**
             * plus-row direction: (r, c) -> (r+1, c) 
             */
            PLUS_R,

            /**
             * minus-row direction: (r, c) -> (r-1, c) 
             */
            MINUS_R,

            /**
             * plus-column direction: (r, c) -> (r, c+1) 
             */
            PLUS_C, 

            /**
             * minux-column direction: (r, c) -> (r, c-1) 
             */
            MINUS_C,
        };

        /**
         * square lattice graph
         *
         * @tparam FloatType floating-point type (default is double)
         */
        template<typename FloatType>
        class Square : public Sparse<FloatType>{
            static_assert( std::is_floating_point<FloatType>::value, "argument must be an arithmetic type" );
            private:

                /**
                 * initial value to be set to interactions
                 */
                FloatType _init_val;

                /**
                 * number of rows
                 */
                std::size_t _num_row;

                /**
                 * number of columns
                 */
                std::size_t _num_column;

                /**
                 * mod function (a mod num_row)
                 *
                 * @param a parameter ([-1:num_row])
                 *
                 * @return (a mod num_row)
                 */
                inline std::size_t mod_r(std::int64_t a) const{
                    return (a+_num_row)%_num_row;
                }

                /**
                 * mod function (a mod num_column)
                 *
                 * @param a parameter ([-1:num_column])
                 *
                 * @return (a mod num_column)
                 */
                inline std::size_t mod_c(std::int64_t a) const{
                    return (a+_num_column)% _num_column;
                }

            public:

                /**
                 * convert from (row x column) index to global index
                 *
                 * @param r row index
                 * @param c column index
                 *
                 * @return corresponding global index
                 */
                Index to_ind(std::int64_t r, std::int64_t c) const{
                    assert(-1 <= r && r <= (std::int64_t)_num_row);
                    assert(-1 <= c && c <= (std::int64_t)_num_column);

                    return _num_column * mod_r(r) + mod_c(c);
                }

                /**
                 * convert from global index to (row x column) index
                 *
                 * @param ind global index
                 *
                 * @return corresponding (row x column) index (RowColumn type)
                 */
                RowColumn to_rc(Index ind) const{
                    assert(ind < this->get_num_spins());
                    std::int64_t r = ind/_num_column;
                    std::int64_t c = ind%_num_column;
                    return std::make_pair(r, c);
                }

                /**
                 * square lattice graph constructor
                 *
                 * @param num_row number of rows
                 * @param num_column number of columns
                 * @param init_val initial value set to interaction (default: 0)
                 */
                Square(std::size_t num_row, std::size_t num_column, FloatType init_val=0)
                    : Sparse<FloatType>(num_row*num_column, 4+1), _init_val(init_val), _num_row(num_row), _num_column(num_column){
                        assert(num_row >= 1);
                        assert(num_column >= 1);

                        for(std::size_t r=0; r<_num_row; r++){
                            for(std::size_t c=0; c<_num_column; c++){
                                //open boundary
                                if(r > 0){
                                    //MINUS_R
                                    this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r-1,c)) = _init_val;
                                }
                                if(c > 0){
                                    //MINUS_C
                                    this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,c-1)) = _init_val;
                                }
                                if(r < num_row-1){
                                    //PLUS_R
                                    this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r+1,c)) = _init_val;
                                }
                                if(c < num_column-1){
                                    //PLUS_C
                                    this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,c+1)) = _init_val;
                                }
                                //local field
                                this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,c)) = _init_val;
                            }
                        }
                    }

                /**
                 * square lattice graph copy constructor
                 *
                 */
                Square(const Square<FloatType>&) = default;

                /**
                 * square lattice graph move constructor
                 *
                 */
                Square(Square<FloatType>&&) = default;

                /**
                 * get number of rows
                 *
                 * @return number of rows
                 */
                std::size_t get_num_row(){return _num_row;}

                /**
                 * get number of columns
                 *
                 * @return number of columns
                 */
                std::size_t get_num_column(){return _num_column;}

                /**
                 * access J(row, colum, direction)
                 *
                 * @param r row index
                 * @param c column index
                 * @param dir direction 
                 *
                 * @return corresponding interaction value
                 */
                FloatType& J(std::size_t r, std::size_t c, Dir dir){
                    assert(r<_num_row);
                    assert(c<_num_column);

                    switch (dir) {
                        case Dir::MINUS_R:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(static_cast<std::int64_t>(r)-1,c));
                        case Dir::MINUS_C:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,static_cast<std::int64_t>(c)-1));
                        case Dir::PLUS_R:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(static_cast<std::int64_t>(r)+1,c));
                        case Dir::PLUS_C:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,static_cast<std::int64_t>(c)+1));
                        default:
                            assert(false);
                            return _init_val;
                    }
                }

                /**
                 * access J(row, colum, direction)
                 *
                 * @param r row index
                 * @param c column index
                 * @param dir direction 
                 *
                 * @return corresponding interaction value
                 */
                const FloatType& J(std::size_t r, std::size_t c, Dir dir) const{
                    assert(r<_num_row);
                    assert(c<_num_column);

                    switch (dir) {
                        case Dir::MINUS_R:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(static_cast<std::int64_t>(r)-1,c));
                        case Dir::MINUS_C:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,static_cast<std::int64_t>(c)-1));
                        case Dir::PLUS_R:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(static_cast<std::int64_t>(r)+1,c));
                        case Dir::PLUS_C:
                            return this->Sparse<FloatType>::J(to_ind(r,c), to_ind(r,static_cast<std::int64_t>(c)+1));
                        default:
                            assert(false);
                            return _init_val;
                    }
                }

                /**
                 * access h(row, colum) (local field)
                 *
                 * @param r row index
                 * @param c column index
                 *
                 * @return corresponding interaction value
                 */
                FloatType& h(std::size_t r, std::size_t c){
                    assert(r<_num_row);
                    assert(c<_num_column);

                    return this->Sparse<FloatType>::h(to_ind(r,c));
                }

                /**
                 * access h(row, colum) (local field)
                 *
                 * @param r row index
                 * @param c column index
                 *
                 * @return corresponding interaction value
                 */
                const FloatType& h(std::size_t r, std::size_t c) const{
                    assert(r<_num_row);
                    assert(c<_num_column);

                    return this->Sparse<FloatType>::h(to_ind(r,c));
                }

                /**
                 * derive spin value at the index (row x column)
                 *
                 * @param spins spin array
                 * @param r row index
                 * @param c column index
                 *
                 * @return corresponding spin
                 */
                Spin& spin(Spins& spins, std::size_t r, std::size_t c) const{
                    return spins[to_ind(r, c)];
                }

                /**
                 * derive spin value at the index (row x column)
                 *
                 * @param spins spin array
                 * @param r row index
                 * @param c column index
                 *
                 * @return corresponding spin
                 */
                const Spin& spin(const Spins& spins, std::size_t r, std::size_t c) const{
                    return spins[to_ind(r, c)];
                }
        };

    } // namespace graph
} // namespace openjij

#endif
