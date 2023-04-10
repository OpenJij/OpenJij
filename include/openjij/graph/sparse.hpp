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
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "openjij/graph/graph.hpp"
#include "openjij/graph/json/parse.hpp"
#include "openjij/utility/pairhash.hpp"

namespace openjij {
namespace graph {

/**
 * @brief Sparse graph: two-body intereactions with O(1) connectivity
 * The Hamiltonian is like
 * \f[
 * H = \sum_{i<j}J_{ij} \sigma_i \sigma_j + \sum_{i}h_{i} \sigma_i
 * \f]
 *
 * @tparam FloatType floating-point type
 */
template <typename FloatType> class Sparse : public Graph {
  static_assert(std::is_floating_point<FloatType>::value,
                "FloatType must be floating-point type.");

public:
  /**
   * @brief interaction type
   */
  using Interactions =
      std::unordered_map<std::pair<Index, Index>, FloatType, utility::PairHash>;

  /**
   * @brief float type
   */
  using value_type = FloatType;

private:
  /**
   * @brief interactions (the number of intereactions is
   * num_spins*(num_spins+1)/2)
   */
  Interactions _J;

  /**
   * @brief the uppder limit of the number of edges per site
   */
  std::size_t _num_edges;

  /**
   * @brief the list of the indices of adjacent nodes
   */
  std::vector<Nodes> _list_adj_nodes;

  /**
   * @brief add adjacent node from "from" Index to "to" Index
   *
   * @param from "from" Index
   * @param to "to" Index
   *
   * @return true if nodes is successfully added, false if this function failed.
   */
  inline bool set_adj_node(Index from, Index to) {
    assert(from < this->get_num_spins());
    assert(to < this->get_num_spins());

    // get adjacent nodes of node "from"
    Nodes &nodes = _list_adj_nodes[from];
    // check if the node "to" exists in the nodes
    if (std::find(nodes.begin(), nodes.end(), to) == nodes.end()) {
      // nodes size must be smaller than num_edges

#ifndef NDEBUG
      assert(nodes.size() < _num_edges);
#else
      // return false
      if (nodes.size() >= _num_edges)
        return false;
#endif
      // add node
      nodes.push_back(to);
      // add node from "to" to "from"
      set_adj_node(to, from);
    }
    return true;
  }

public:
  /**
   * @brief Sparse constructor
   *
   * @param num_spins number of spins
   * @param num_edges number of edges
   */
  Sparse(std::size_t num_spins, std::size_t num_edges)
      : Graph(num_spins), _num_edges(std::min(num_spins, num_edges)),
        _list_adj_nodes(num_spins) {}

  /**
   * @brief Sparse delegate constructor
   *
   * @param num_spins number of spins
   */
  explicit Sparse(std::size_t num_spins) : Sparse(num_spins, num_spins) {}

  /**
   * @brief Sparse constructor (from nlohmann::json)
   *
   * @param j JSON object
   * @param num_edges number of edges
   */
  Sparse(const json &j, std::size_t num_edges)
      : Sparse(static_cast<std::size_t>(j["num_variables"]), num_edges) {

    // define SparseMatrix and iterator
    using SparseMatrix = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
    using SpIter = typename SparseMatrix::InnerIterator;

    // define bqm with ising variables
    auto bqm = json_parse<FloatType, cimod::Sparse>(j);
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

    // insert elements
    SparseMatrix quadmat = bqm.interaction_matrix();
    size_t num_variables = quadmat.rows() - 1;
    for (int k = 0; k < quadmat.outerSize(); k++) {
      for (SpIter it(quadmat, k); it; ++it) {
        size_t r = it.row();
        size_t c = it.col();
        FloatType val = it.value();

        if (r == num_variables && c == num_variables)
          continue;

        if (c == num_variables) {
          // local field
          h(r) += val;
        } else {
          // quadratic
          J(r, c) += val;
        }
      }
    }
  }

  /**
   * @brief Sparse constructor (from nlohmann::json)
   *
   * @param j JSON object
   */
  Sparse(const json &j) : Sparse(j, j["num_variables"]) {}

  /**
   * @brief Sparse copy constructor
   *
   */
  Sparse(const Sparse<FloatType> &) = default;

  /**
   * @brief Sparse move constructor
   *
   */
  Sparse(Sparse<FloatType> &&) = default;

  /**
   * @brief list of adjacent nodes
   *
   * @param ind Node index
   *
   * @return corresponding list of adjacent nodes
   */
  const Nodes &adj_nodes(Index ind) const { return _list_adj_nodes[ind]; }

  /**
   * @brief get number of edges
   *
   * @return number of edges
   */
  std::size_t get_num_edges() const { return _num_edges; }

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
    if (!(spins.size() == this->get_num_spins())) {
      std::out_of_range("Out of range in energy in Sparse graph.");
    }

    FloatType ret = 0;
    for (std::size_t ind = 0; ind < this->get_num_spins(); ind++) {
      for (auto &adj_ind : _list_adj_nodes[ind]) {
        if (ind != adj_ind)
          ret += (1. / 2) * this->J(ind, adj_ind) * spins[ind] * spins[adj_ind];
        else
          ret += this->h(ind) * spins[ind];
      }
    }

    return ret;
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

    // i <= j
    // add node if it does not exist
    set_adj_node(i, j);
    return _J[std::make_pair(std::min(i, j), std::max(i, j))];
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
    return _J.at(std::make_pair(std::min(i, j), std::max(i, j)));
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
    set_adj_node(i, i);
    return _J[std::make_pair(i, i)];
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
    return _J.at(std::make_pair(i, i));
  }
};
} // namespace graph
} // namespace openjij
