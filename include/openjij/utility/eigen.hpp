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

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "openjij/graph/all.hpp"

namespace openjij {
namespace utility {

/**
 * @brief get Eigen Matrix type from Graph Type
 *
 * @tparam GraphType
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 */
template <typename GraphType, int Options = Eigen::ColMajor>
struct get_eigen_matrix_type {};

/**
 * @brief get Eigen Matrix type from Graph Type
 *
 * @tparam GraphType
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 */
template <typename FloatType, int Options>
struct get_eigen_matrix_type<graph::Dense<FloatType>, Options> {
  using type =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Options>;
};

/**
 * @brief get Eigen Matrix type from Graph Type
 *
 * @tparam GraphType
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 */
template <typename FloatType, int Options>
struct get_eigen_matrix_type<graph::Sparse<FloatType>, Options> {
  using type = Eigen::SparseMatrix<FloatType, Options>;
};

/**
 * @brief generate Eigen Vector from std::vector
 *
 * @tparam FloatType
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 * @param init_spin
 *
 * @return generated Eigen Vector (init_spin.size()+1 x 1)
 */
template <typename FloatType, int Options = Eigen::ColMajor>
inline static Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Options>
gen_vector_from_std_vector(const graph::Spins &init_spin) {
  Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Options> ret_vec(
      init_spin.size() + 1);

  // initialize spin
  for (size_t i = 0; i < init_spin.size(); i++) {
    ret_vec(i) = init_spin[i];
  }

  // for local field
  ret_vec[init_spin.size()] = 1;

  return ret_vec;
}

/**
 * @brief generate Eigen Matrix from TrotterSpins
 *
 * @tparam FloatType
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 * @param trotter_spins
 *
 * @return generated Eigen Matrix (trotter_spins[0].size()+1 x
 * trotter_spins.size())
 */
template <typename FloatType, int Options = Eigen::ColMajor>
inline static Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Options>
gen_matrix_from_trotter_spins(const std::vector<graph::Spins> &trotter_spins) {
  Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Options> ret_mat(
      trotter_spins[0].size() + 1, trotter_spins.size());

  // initialize spin
  for (size_t j = 0; j < trotter_spins.size(); j++) {
    for (size_t i = 0; i < trotter_spins[j].size(); i++) {
      ret_mat(i, j) = trotter_spins[j][i];
    }
  }

  // dummy spins
  for (size_t j = 0; j < trotter_spins.size(); j++) {
    ret_mat(trotter_spins[0].size(), j) = 1;
  }

  return ret_mat;
}

/**
 * @brief generate Eigen Dense Matrix from Dense graph
 *
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 * @tparam FloatType
 * @param graph
 *
 * @return generated Eigen Dense Matrix (graph.get_num_spins()+1 x
 * graph.get_num_spins()+1)
 */
template <int Options = Eigen::ColMajor, typename FloatType>
inline static Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Options>
gen_matrix_from_graph(const graph::Dense<FloatType> &graph) {
  // initialize interaction
  Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Options> ret_mat(
      graph.get_num_spins() + 1, graph.get_num_spins() + 1);

  ret_mat.setZero();

  for (size_t i = 0; i < graph.get_num_spins(); i++) {
    for (size_t j = i + 1; j < graph.get_num_spins(); j++) {
      ret_mat(i, j) = graph.J(i, j);
      ret_mat(j, i) = graph.J(i, j);
    }
  }

  // for local field
  for (size_t i = 0; i < graph.get_num_spins(); i++) {
    ret_mat(i, graph.get_num_spins()) = graph.h(i);
    ret_mat(graph.get_num_spins(), i) = graph.h(i);
  }

  // for local field
  ret_mat(graph.get_num_spins(), graph.get_num_spins()) = 1;

  return ret_mat;
}

/**
 * @brief generate Eigen Sparse Matrix from Sparse graph
 *
 * @tparam Options Eigen Options (RowMajor or ColMajor)
 * @tparam FloatType
 * @param graph
 *
 * @return generated Eigen Sparse Matrix (graph.get_num_spins()+1 x
 * graph.get_num_spins()+1)
 */
template <int Options = Eigen::ColMajor, typename FloatType>
inline static Eigen::SparseMatrix<FloatType, Options>
gen_matrix_from_graph(const graph::Sparse<FloatType> &graph) {
  // initialize interaction
  Eigen::SparseMatrix<FloatType, Options> ret_mat(graph.get_num_spins() + 1,
                                                  graph.get_num_spins() + 1);

  ret_mat.setZero();

  // make triplet list
  using T = std::vector<Eigen::Triplet<FloatType>>;
  T t_list;

  for (size_t ind = 0; ind < graph.get_num_spins(); ind++) {
    for (size_t adj_ind : graph.adj_nodes(ind)) {
      if (ind != adj_ind) {
        t_list.emplace_back(ind, adj_ind, graph.J(ind, adj_ind));
      } else {
        t_list.emplace_back(ind, graph.get_num_spins(), graph.h(ind));
        t_list.emplace_back(graph.get_num_spins(), ind, graph.h(ind));
      }
    }
  }

  t_list.emplace_back(graph.get_num_spins(), graph.get_num_spins(), 1);

  ret_mat.setFromTriplets(t_list.begin(), t_list.end());

  return ret_mat;
}

} // namespace utility
} // namespace openjij
