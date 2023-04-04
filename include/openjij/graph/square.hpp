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
#include <type_traits>
#include <utility>
#include <exception>

#include "openjij/graph/sparse.hpp"

namespace openjij {
namespace graph {

/**
 * @brief Row x Column type
 */
using RowColumn = std::pair<std::size_t, std::size_t>;

/**
 * @brief direction enum class
 */
enum class Dir {

  /**
   * @brief plus-row direction: (r, c) -> (r+1, c)
   */
  PLUS_R,

  /**
   * @brief minus-row direction: (r, c) -> (r-1, c)
   */
  MINUS_R,

  /**
   * @brief plus-column direction: (r, c) -> (r, c+1)
   */
  PLUS_C,

  /**
   * @brief minux-column direction: (r, c) -> (r, c-1)
   */
  MINUS_C,
};

/**
 * @brief square lattice graph
 *
 * @tparam FloatType floating-point type (default is double)
 */
template <typename FloatType> class Square : public Sparse<FloatType> {
  static_assert(std::is_floating_point<FloatType>::value,
                "argument must be an arithmetic type");

private:
  /**
   * @brief initial value to be set to interactions
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
   * @brief mod function (a mod num_row)
   *
   * @param a parameter ([-1:num_row])
   *
   * @return (a mod num_row)
   */
  inline std::size_t mod_r(std::int64_t a) const {
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
    std::int64_t r1, c1, r2, c2;
    std::tie(r1, c1) = to_rc(idx1);
    std::tie(r2, c2) = to_rc(idx2);
    if (!((r1 == r2 && std::abs(c1 - c2) == 1) || // horizontal connect
          (c1 == c2 && std::abs(r1 - r2) == 1) || // vertical connect
          (r1 == r2 && c1 == c2))                 // local field
    ) {
      throw std::runtime_error("invalid index pair " + std::to_string(idx1) +
                               " " + std::to_string(idx2) +
                               " inserted in Square");
    }
  }

public:
  /**
   * @brief convert from (row x column) index to global index
   *
   * @param r row index
   * @param c column index
   *
   * @return corresponding global index
   */
  Index to_ind(std::int64_t r, std::int64_t c) const {
    if (!(-1 <= r && r <= (std::int64_t)_num_row)) {
      throw std::runtime_error("invalid value r=" + std::to_string(r) +
                               " inserted in Square");
    }

    if (!(-1 <= c && c <= (std::int64_t)_num_column)) {
      throw std::runtime_error("invalid value c=" + std::to_string(c) +
                               " inserted in Square");
    }

    return _num_column * mod_r(r) + mod_c(c);
  }

  /**
   * @brief convert from global index to (row x column) index
   *
   * @param ind global index
   *
   * @return corresponding (row x column) index (RowColumn type)
   */
  RowColumn to_rc(Index ind) const {
    if (!(ind < this->get_num_spins())) {
      throw std::runtime_error("invalid value index=" + std::to_string(ind) +
                               " inserted in Square");
    }

    std::int64_t r = ind / _num_column;
    std::int64_t c = ind % _num_column;
    return std::make_pair(r, c);
  }

  /**
   * @brief square lattice graph constructor
   *
   * @param num_row number of rows
   * @param num_column number of columns
   * @param init_val initial value set to interaction (default: 0)
   */
  Square(std::size_t num_row, std::size_t num_column, FloatType init_val = 0)
      : Sparse<FloatType>(num_row * num_column, 4 + 1), _init_val(init_val),
        _num_row(num_row), _num_column(num_column) {
    assert(num_row >= 1);
    assert(num_column >= 1);

    for (std::size_t r = 0; r < _num_row; r++) {
      for (std::size_t c = 0; c < _num_column; c++) {
        // open boundary
        if (r > 0) {
          // MINUS_R
          this->Sparse<FloatType>::J(to_ind(r, c), to_ind(r - 1, c)) =
              _init_val;
        }
        if (c > 0) {
          // MINUS_C
          this->Sparse<FloatType>::J(to_ind(r, c), to_ind(r, c - 1)) =
              _init_val;
        }
        if (r < num_row - 1) {
          // PLUS_R
          this->Sparse<FloatType>::J(to_ind(r, c), to_ind(r + 1, c)) =
              _init_val;
        }
        if (c < num_column - 1) {
          // PLUS_C
          this->Sparse<FloatType>::J(to_ind(r, c), to_ind(r, c + 1)) =
              _init_val;
        }
        // local field
        this->Sparse<FloatType>::J(to_ind(r, c), to_ind(r, c)) = _init_val;
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
  Square(const json &j, std::size_t num_row, std::size_t num_column,
         FloatType init_val = 0)
      : Square<FloatType>(num_row, num_column, init_val) {
    if (!(j["num_variables"] <= num_row * num_column)) {
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
   * @brief square lattice graph copy constructor
   *
   */
  Square(const Square<FloatType> &) = default;

  /**
   * @brief square lattice graph move constructor
   *
   */
  Square(Square<FloatType> &&) = default;

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
   * @brief access J(row, colum, direction)
   *
   * @param r row index
   * @param c column index
   * @param dir direction
   *
   * @return corresponding interaction value
   */
  FloatType &J(std::size_t r, std::size_t c, Dir dir) {
    assert(r < _num_row);
    assert(c < _num_column);

    switch (dir) {
    case Dir::MINUS_R:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(static_cast<std::int64_t>(r) - 1, c));
    case Dir::MINUS_C:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(r, static_cast<std::int64_t>(c) - 1));
    case Dir::PLUS_R:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(static_cast<std::int64_t>(r) + 1, c));
    case Dir::PLUS_C:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(r, static_cast<std::int64_t>(c) + 1));
    default:
      return _init_val;
    }
  }

  /**
   * @brief access J(row, colum, direction)
   *
   * @param r row index
   * @param c column index
   * @param dir direction
   *
   * @return corresponding interaction value
   */
  const FloatType &J(std::size_t r, std::size_t c, Dir dir) const {
    assert(r < _num_row);
    assert(c < _num_column);

    switch (dir) {
    case Dir::MINUS_R:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(static_cast<std::int64_t>(r) - 1, c));
    case Dir::MINUS_C:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(r, static_cast<std::int64_t>(c) - 1));
    case Dir::PLUS_R:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(static_cast<std::int64_t>(r) + 1, c));
    case Dir::PLUS_C:
      return this->Sparse<FloatType>::J(
          to_ind(r, c), to_ind(r, static_cast<std::int64_t>(c) + 1));
    default:
      return _init_val;
    }
  }

  /**
   * @brief access h(row, colum) (local field)
   *
   * @param r row index
   * @param c column index
   *
   * @return corresponding interaction value
   */
  FloatType &h(std::size_t r, std::size_t c) {
    assert(r < _num_row);
    assert(c < _num_column);

    return this->Sparse<FloatType>::h(to_ind(r, c));
  }

  /**
   * @brief access h(row, colum) (local field)
   *
   * @param r row index
   * @param c column index
   *
   * @return corresponding interaction value
   */
  const FloatType &h(std::size_t r, std::size_t c) const {
    assert(r < _num_row);
    assert(c < _num_column);

    return this->Sparse<FloatType>::h(to_ind(r, c));
  }

  /**
   * @brief derive spin value at the index (row x column)
   *
   * @param spins spin array
   * @param r row index
   * @param c column index
   *
   * @return corresponding spin
   */
  Spin &spin(Spins &spins, std::size_t r, std::size_t c) const {
    return spins[to_ind(r, c)];
  }

  /**
   * @brief derive spin value at the index (row x column)
   *
   * @param spins spin array
   * @param r row index
   * @param c column index
   *
   * @return corresponding spin
   */
  const Spin &spin(const Spins &spins, std::size_t r, std::size_t c) const {
    return spins[to_ind(r, c)];
  }
};

} // namespace graph
} // namespace openjij
