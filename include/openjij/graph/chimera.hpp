//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "openjij/graph/sparse.hpp"

namespace openjij {
namespace graph {

/**
 * @brief Chimera index (row, column, in-chimera)
 * The structure of in-chimera is as follows
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
using ChimeraIndex = std::tuple<std::size_t, std::size_t, std::size_t>;

/**
 * @brief direction in chimera graph
 */
enum class ChimeraDir {

  /**
   * @brief plus-row direction: (r, c, ind) -> (r+1, c, ind)
   */
  PLUS_R,

  /**
   * @brief minus-row direction: (r, c, ind) -> (r-1, c, ind)
   */
  MINUS_R,

  /**
   * @brief plus-column direction: (r, c, ind) -> (r, c+1, ind)
   */
  PLUS_C,

  /**
   * @brief minus-column direction: (r, c, ind) -> (r, c-1, ind)
   */
  MINUS_C,

  /**
   * @brief inside-chimera 0or4 direction: (r, c, ind) -> (r, c, 0or4)
   */
  IN_0or4,

  /**
   * @brief inside-chimera 1or5 direction: (r, c, ind) -> (r, c, 1or5)
   */
  IN_1or5,

  /**
   * @brief inside-chimera 2or6 direction: (r, c, ind) -> (r, c, 2or6)
   */
  IN_2or6,

  /**
   * @brief inside-chimera 3or7 direction: (r, c, ind) -> (r, c, 3or7)
   */
  IN_3or7,
};

/**
 * @brief chimera lattice graph
 *
 * @tparam FloatType floating-point type (default: double)
 */
template <typename FloatType = double>
class Chimera : public Sparse<FloatType> {
  static_assert(std::is_arithmetic<FloatType>::value,
                "argument must be an arithmetic type");

private:
  /**
   * @brief initial value to be set to inreactions
   */
  FloatType _init_val;

  /**
   * @brief number of rows
   */
  std::size_t _num_row;

  /**
   * @brief number of columns
   */
  std::size_t _num_column;

  /**
   * @brief number of spins in each chimera units (8)
   */
  constexpr static std::size_t _num_in_chimera = 8;

  /**
   * @brief mod function (a mod num_row)
   *
   * @param a parameter ([-1:num_row])
   *
   * @return (a mod num_row)
   */
  inline std::size_t mod_r(std::int64_t a) const {
    // a -> [-1:num_row]
    return (a + _num_row) % _num_row;
  }

  /**
   * @brief mod function (a mod num_column)
   *
   * @param a parameter ([-1:num_column])
   *
   * @return (a mod num_column)
   */
  inline std::size_t mod_c(std::int64_t a) const {
    // a -> [-1:num_column]
    return (a + _num_column) % _num_column;
  }

  /**
   * @brief check if the pair has a valid connection
   *
   * @param idx1 index1
   * @param idx2 index2
   *
   */
  inline void _checkpair(Index idx1, Index idx2) const {
    std::int64_t r1, c1, i1, r2, c2, i2;
    std::tie(r1, c1, i1) = to_rci(idx1);
    std::tie(r2, c2, i2) = to_rci(idx2);
    if (!((r1 == r2 && std::abs(c1 - c2) == 1 && i1 == i2 &&
           i1 >= 4) || // horizontal connect
          (c1 == c2 && std::abs(r1 - r2) == 1 && i1 == i2 &&
           4 < i1) ||                                    // vertical connect
          (r1 == r2 && c1 == c2 && i1 < 4 && 4 <= i2) || // in-site connect
          (r1 == r2 && c1 == c2 && i2 < 4 && 4 <= i1) || // in-site connect
          (r1 == r2 && c1 == c2 && i1 == i2))            // local field
    ) {
      throw std::runtime_error("invalid index pair " + std::to_string(idx1) +
                               " " + std::to_string(idx2) +
                               " inserted in Chimera");
    }
  }

public:
  /**
   * @brief convert from (row x column x in-chimera) index to global index
   *
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   *
   * @return corresponding global index
   */
  Index to_ind(std::int64_t r, std::int64_t c, std::int64_t i) const {
    // row-major matrix

    if (!(-1 <= r && r <= static_cast<std::int64_t>(_num_row))) {
      throw std::runtime_error("invalid value r=" + std::to_string(r) +
                               " inserted in Chimera");
    }

    if (!(-1 <= c && c <= static_cast<std::int64_t>(_num_column))) {
      throw std::runtime_error("invalid value c=" + std::to_string(c) +
                               " inserted in Chimera");
    }

    if (!(0 <= i && i < static_cast<std::int64_t>(_num_in_chimera))) {
      throw std::runtime_error("invalid value i=" + std::to_string(i) +
                               " inserted in Chimera");
    }

    return _num_column * _num_in_chimera * mod_r(r) +
           _num_in_chimera * mod_c(c) + i;
  }

  /**
   * @brief convert from global index to (row x column x in-chimera) index
   *
   * @param ind
   *
   * @return
   */
  ChimeraIndex to_rci(Index ind) const {
    if (!(ind < this->get_num_spins())) {
      throw std::runtime_error("invalid value index=" + std::to_string(ind) +
                               " inserted in Chimera");
    }

    std::int64_t r = ind / (_num_column * _num_in_chimera);
    ind -= (_num_column * _num_in_chimera) * r;
    std::int64_t c = ind / _num_in_chimera;
    ind -= (_num_in_chimera)*c;
    std::int64_t i = ind;
    return std::make_tuple(r, c, i);
  }

  /**
   * @brief chimera lattice graph constructor
   *
   * @param num_row number of rows
   * @param num_column number of columns
   * @param init_val initial value set to interaction (default: 0)
   */
  Chimera(std::size_t num_row, std::size_t num_column, FloatType init_val = 0)
      : Sparse<FloatType>(num_row * num_column * _num_in_chimera, 6 + 1),
        _init_val(init_val), _num_row(num_row), _num_column(num_column) {
    // generate sparse graph
    assert(_num_row >= 1);
    assert(_num_column >= 1);

    for (std::size_t r = 0; r < _num_row; r++) {
      for (std::size_t c = 0; c < _num_column; c++) {
        for (std::size_t i = 0; i < _num_in_chimera; i++) {
          // open boundary
          if (r > 0 && i < 4) {
            // MINUS_R (0<=i<4)
            this->Sparse<FloatType>::J(to_ind(r, c, i), to_ind(r - 1, c, i)) =
                _init_val;
          }
          if (c > 0 && 4 <= i) {
            // MINUS_C (4<=i<8)
            this->Sparse<FloatType>::J(to_ind(r, c, i), to_ind(r, c - 1, i)) =
                _init_val;
          }
          if (r < num_row - 1 && i < 4) {
            // PLUS_R (0<=i<4)
            this->Sparse<FloatType>::J(to_ind(r, c, i), to_ind(r + 1, c, i)) =
                _init_val;
          }
          if (c < num_column - 1 && 4 <= i) {
            // PLUS_C (4<=i<8)
            this->Sparse<FloatType>::J(to_ind(r, c, i), to_ind(r, c + 1, i)) =
                _init_val;
          }

          // inside chimera unit

          this->Sparse<FloatType>::J(to_ind(r, c, i),
                                     to_ind(r, c, (i < 4) ? 4 : 0)) = _init_val;
          this->Sparse<FloatType>::J(to_ind(r, c, i),
                                     to_ind(r, c, (i < 4) ? 5 : 1)) = _init_val;
          this->Sparse<FloatType>::J(to_ind(r, c, i),
                                     to_ind(r, c, (i < 4) ? 6 : 2)) = _init_val;
          this->Sparse<FloatType>::J(to_ind(r, c, i),
                                     to_ind(r, c, (i < 4) ? 7 : 3)) = _init_val;

          // local field
          this->Sparse<FloatType>::J(to_ind(r, c, i), to_ind(r, c, i)) =
              _init_val;
        }
      }
    }
  }

  /**
   * @brief Square constructor (from nlohmann::json)
   *
   * @param j JSON object
   * @param num_row number of rows
   * @param num_column number of columns
   * @param init_val initial value set to interaction (default: 0)
   */
  Chimera(const json &j, std::size_t num_row, std::size_t num_column,
          FloatType init_val = 0)
      : Chimera<FloatType>(num_row, num_column, init_val) {
    if (!(j["num_variables"] <= num_row * num_column * _num_in_chimera)) {
      throw std::runtime_error("number of system size does not match");
    }
    // define bqm with ising variables
    auto bqm = json_parse<FloatType, cimod::Sparse>(j, false);
    // interactions
    for (auto &&elem : bqm.get_quadratic()) {
      const auto &key = elem.first;
      const auto &val = elem.second;
      _checkpair(key.first, key.second);
      this->Sparse<FloatType>::J(key.first, key.second) += val;
    }
    // local field
    for (auto &&elem : bqm.get_linear()) {
      const auto &key = elem.first;
      const auto &val = elem.second;
      this->Sparse<FloatType>::h(key) += val;
    }
  }

  /**
   * @brief chimera lattice graph copy constructor
   *
   */
  Chimera(const Chimera<FloatType> &) = default;

  /**
   * @brief chimera lattice graph move constructor
   *
   */
  Chimera(Chimera<FloatType> &&) = default;

  /**
   * @brief get number of rows
   *
   * @return number of rows
   */
  std::size_t get_num_row() const { return _num_row; }

  /**
   * @brief get number of columns
   *
   * @return number of columns
   */
  std::size_t get_num_column() const { return _num_column; }

  /**
   * @brief get number of spins in each chimera unit
   *
   * @return number of spins in each chimera unit
   */
  std::size_t get_num_in_chimera() const { return _num_in_chimera; }

  /**
   * @brief access J(row, colum, in-chimera, direction)
   *
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   * @param dir chimera direction
   *
   * @return corresponding interaction value
   */
  FloatType &J(std::size_t r, std::size_t c, std::size_t i, ChimeraDir dir) {
    assert(r < _num_row);
    assert(c < _num_column);
    assert(i < _num_in_chimera);

    switch (dir) {
    case ChimeraDir::MINUS_R:
      assert(i < 4);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(static_cast<std::int64_t>(r) - 1, c, i));
    case ChimeraDir::MINUS_C:
      assert(4 <= i);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(r, static_cast<std::int64_t>(c) - 1, i));
    case ChimeraDir::PLUS_R:
      assert(i < 4);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(static_cast<std::int64_t>(r) + 1, c, i));
    case ChimeraDir::PLUS_C:
      assert(4 <= i);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(r, static_cast<std::int64_t>(c) + 1, i));

    case ChimeraDir::IN_0or4:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 4 : 0));
    case ChimeraDir::IN_1or5:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 5 : 1));
    case ChimeraDir::IN_2or6:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 6 : 2));
    case ChimeraDir::IN_3or7:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 7 : 3));

    default:
      assert(false);
      return _init_val;
    }
  }

  /**
   * @brief access J(row, colum, in-chimera, direction)
   *
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   * @param dir chimera direction
   *
   * @return corresponding interaction value
   */
  const FloatType &J(std::size_t r, std::size_t c, std::size_t i,
                     ChimeraDir dir) const {
    assert(r < _num_row);
    assert(c < _num_column);
    assert(i < _num_in_chimera);

    switch (dir) {
    case ChimeraDir::MINUS_R:
      assert(i < 4);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(static_cast<std::int64_t>(r) - 1, c, i));
    case ChimeraDir::MINUS_C:
      assert(4 <= i);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(r, static_cast<std::int64_t>(c) - 1, i));
    case ChimeraDir::PLUS_R:
      assert(i < 4);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(static_cast<std::int64_t>(r) + 1, c, i));
    case ChimeraDir::PLUS_C:
      assert(4 <= i);
      return this->Sparse<FloatType>::J(
          to_ind(r, c, i), to_ind(r, static_cast<std::int64_t>(c) + 1, i));

    case ChimeraDir::IN_0or4:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 4 : 0));
    case ChimeraDir::IN_1or5:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 5 : 1));
    case ChimeraDir::IN_2or6:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 6 : 2));
    case ChimeraDir::IN_3or7:
      return this->Sparse<FloatType>::J(to_ind(r, c, i),
                                        to_ind(r, c, (i < 4) ? 7 : 3));

    default:
      assert(false);
      return _init_val;
    }
  }

  /**
   * @brief access h(row, colum, in-chimera) (local field)
   *
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   *
   * @return corresponding interaction value
   */
  FloatType &h(std::size_t r, std::size_t c, std::size_t i) {
    assert(r < _num_row);
    assert(c < _num_column);
    assert(i < _num_in_chimera);

    return this->Sparse<FloatType>::h(to_ind(r, c, i));
  }

  /**
   * @brief access h(row, colum, in-chimera) (local field)
   *
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   *
   * @return corresponding interaction value
   */
  const FloatType &h(std::size_t r, std::size_t c, std::size_t i) const {
    assert(r < _num_row);
    assert(c < _num_column);
    assert(i < _num_in_chimera);

    return this->Sparse<FloatType>::h(to_ind(r, c, i));
  }

  /**
   * @brief derive spin value at the index (row x column)
   *
   * @param spins spin array
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   *
   * @return corresponding spin
   */
  Spin &spin(Spins &spins, std::size_t r, std::size_t c, std::size_t i) const {
    return spins[to_ind(r, c, i)];
  }

  /**
   * @brief derive spin value at the index (row x column)
   *
   * @param spins spin array
   * @param r row index
   * @param c column index
   * @param i in-chimera index
   *
   * @return corresponding spin
   */
  const Spin &spin(const Spins &spins, std::size_t r, std::size_t c,
                   std::size_t i) const {
    return spins[to_ind(r, c, i)];
  }
};
} // namespace graph
} // namespace openjij
