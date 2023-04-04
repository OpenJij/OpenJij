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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <map>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "openjij/utility/disable_eigen_warning.hpp"

#include "openjij/graph/graph.hpp"
#include "openjij/graph/json/parse.hpp"

namespace openjij {
namespace graph {

/**
 * @brief two-body all-to-all interactions
 * The Hamiltonian is like
 * \f[
 * H = \sum_{i<j}J_{ij} \sigma_i \sigma_j + \sum_{i}h_{i} \sigma_i
 * \f]
 *
 * @tparam FloatType float type of Sparse class (double or float)
 */
template <typename FloatType> class Dense : public Graph {
  static_assert(std::is_floating_point<FloatType>::value,
                "FloatType must be floating-point type.");

public:
  /**
   * @brief interaction type (Eigen)
   * The stored matrix has the following triangular form:
   *
   * \f[
   * \begin{pmatrix}
   * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
   * 0 & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
   * \vdots & \vdots & \vdots & \vdots & \vdots \\
   * 0 & 0 & \cdots & J_{N-1,N-1} & h_{N-1}\\
   * 0 & 0 & \cdots & 0 & 1 \\
   * \end{pmatrix}
   * \f]
   */
  using Interactions =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  /**
   * @brief float type
   */
  using value_type = FloatType;

private:
  /**
   * @brief interactions
   */
  Interactions _J;

public:
  /**
   * @brief Dense constructor
   *
   * @param num_spins the number of spins
   */
  explicit Dense(std::size_t num_spins)
      : Graph(num_spins), _J(Interactions::Zero(num_spins + 1, num_spins + 1)) {
    _J(num_spins, num_spins) = 1;
  }

  /**
   * @brief Dense constructor (from nlohmann::json)
   *
   * @param j JSON object this object must be a serialized object of dense BQM.
   */
  Dense(const json &j) : Dense(static_cast<size_t>(j["num_variables"])) {
    // define bqm with ising variables
    auto bqm = json_parse<FloatType, cimod::Dense>(j);
    // interactions
    // for(auto&& elem : bqm.get_quadratic()){
    //    const auto& key = elem.first;
    //    const auto& val = elem.second;
    //    J(key.first, key.second) += val;
    //}
    // local field
    // for(auto&& elem : bqm.get_linear()){
    //    const auto& key = elem.first;
    //    const auto& val = elem.second;
    //    h(key) += val;
    //}
    // the above insertion is simplified as
    this->_J = bqm.interaction_matrix();
  }

  /**
   * @brief set interaction matrix from Eigen Matrix.
   *
   * @param interaction Eigen matrix
   */
  void set_interaction_matrix(const Interactions &interaction) {
    if (interaction.rows() != interaction.cols()) {
      std::runtime_error("interaction.rows() != interaction.cols()");
    }

    if ((size_t)interaction.rows() != get_num_spins() + 1) {
      throw std::runtime_error("invalid matrix size.");
    }

    // check if diagonal elements are zero
    for (size_t i = 0; i < (size_t)(interaction.rows() - 1); i++) {
      if (interaction(i, i) != 0) {
        throw std::runtime_error(
            "The diagonal elements of interaction matrix must be zero.");
      }
    }

    if (interaction(interaction.rows() - 1, interaction.rows() - 1) != 1) {
      throw std::runtime_error(
          "The right bottom element of interaction matrix must be unity.");
    }

    _J = interaction.template selfadjointView<Eigen::Upper>();
  }

  /**
   * @brief Dense copy constructor
   */
  Dense(const Dense<FloatType> &) = default;

  /**
   * @brief Dense move constructor
   */
  Dense(Dense<FloatType> &&) = default;

  /**
   * @brief calculate total energy
   *
   * @param spins
   * @deprecated use energy(spins)
   *
   * @return corresponding energy
   */
  FloatType calc_energy(const Spins &spins) const {
    return this->energy(spins);
  }

  FloatType calc_energy(const Eigen::Matrix<FloatType, Eigen::Dynamic, 1,
                                            Eigen::ColMajor> &spins) const {
    return this->energy(spins);
  }

  /**
   * @brief calculate total energy
   *
   * @param spins
   *
   * @return corresponding energy
   */
  FloatType energy(const Spins &spins) const {
    if (spins.size() != this->get_num_spins()) {
      throw std::out_of_range("Out of range in energy in Dense graph.");
    }

    using Vec = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;
    Vec s(get_num_spins() + 1);
    for (size_t i = 0; i < spins.size(); i++) {
      s(i) = spins[i];
    }
    s(get_num_spins()) = 1;

    // the energy must be consistent with BinaryQuadraticModel.
    return (s.transpose() *
            (_J.template triangularView<Eigen::Upper>() * s))(0, 0) -
           1;
  }

  FloatType energy(const Eigen::Matrix<FloatType, Eigen::Dynamic, 1,
                                       Eigen::ColMajor> &spins) const {
    graph::Spins temp_spins(get_num_spins());
    for (size_t i = 0; i < temp_spins.size(); i++) {
      temp_spins[i] = spins(i);
    }
    return energy(temp_spins);
  }

  /**
   * @brief access J_{ij}
   *
   * @param i Index i
   * @param j Index j
   *
   * @return J_{ij}
   */
  FloatType &J(Index i, Index j) {
    assert(i < get_num_spins());
    assert(j < get_num_spins());

    if (i != j)
      return _J(std::min(i, j), std::max(i, j));
    else
      return _J(std::min(i, j), get_num_spins());
  }

  /**
   * @brief access J_{ij}
   *
   * @param i Index i
   * @param j Index j
   *
   * @return J_{ij}
   */
  const FloatType &J(Index i, Index j) const {
    assert(i < get_num_spins());
    assert(j < get_num_spins());

    if (i != j)
      return _J(std::min(i, j), std::max(i, j));
    else
      return _J(std::min(i, j), get_num_spins());
  }

  /**
   * @brief access h_{i} (local field)
   *
   * @param i Index i
   *
   * @return h_{i}
   */
  FloatType &h(Index i) {
    assert(i < get_num_spins());
    return J(i, i);
  }

  /**
   * @brief access h_{i} (local field)
   *
   * @param i Index i
   *
   * @return h_{i}
   */
  const FloatType &h(Index i) const {
    assert(i < get_num_spins());
    return J(i, i);
  }

  /**
   * @brief get interactions (Eigen Matrix)
   *
   * The returned matrix has the following symmetric form:
   *
   * \f[
   * \begin{pmatrix}
   * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
   * J_{0,1} & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
   * \vdots & \vdots & \vdots & \vdots & \vdots \\
   * J_{0,N-1} & J_{N-1,1} & \cdots & J_{N-1,N-1} & h_{N-1}\\
   * h_{0} & h_{1} & \cdots & h_{N-1} & 1 \\
   * \end{pmatrix}
   * \f]
   *
   * @return Eigen Matrix
   */
  const Interactions get_interactions() const {
    return this->_J.template selfadjointView<Eigen::Upper>();
  }
};
} // namespace graph
} // namespace openjij
