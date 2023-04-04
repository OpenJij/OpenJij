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

#include <cassert>
#include <utility>
#include <vector>

#include "openjij/graph/all.hpp"
#include "openjij/system/system.hpp"
#include "openjij/utility/eigen.hpp"

namespace openjij {
namespace system {

/**
 * @brief Continuous Time Quantum Ising system
 *
 * @tparam GraphType
 */

template <typename GraphType> struct ContinuousTimeIsing;

/**
 * @brief Continuous Time Quantum Ising system (for Sparse graph)
 *
 * @tparam FloatType
 */
template <typename FloatType>
struct ContinuousTimeIsing<graph::Sparse<FloatType>> {
  using GraphType = graph::Sparse<FloatType>;
  using system_type = transverse_field_system;
  using TimeType = FloatType;
  using CutPoint = std::pair<TimeType, graph::Spin>;

  /**
   * @brief Interaction type (Eigen sparse matrix)
   */
  using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;

  /**
   * @brief spin configuration in real and continuous time space
   * spin_config[i][j] -> at i th site, j th pair of imaginary time point and
   * spin value after the point
   */
  using SpinConfiguration = std::vector<std::vector<CutPoint>>;

  /**
   * @brief ContinuousTimeIsing constructor
   *
   * @param init_spin_config
   * @param init_interaction
   * @param gamma
   */
  ContinuousTimeIsing(const SpinConfiguration &init_spin_config,
                      const GraphType &init_interaction, const double gamma)
      : spin_config(init_spin_config), num_spins(init_spin_config.size() + 1),
        interaction(
            utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction)),
        gamma(gamma) {

    assert(init_spin_config.size() == init_interaction.get_num_spins());

    // insert numbers to diagnonal elements in the interaction
    SparseMatrixXx diag(init_interaction.get_num_spins() + 1,
                        init_interaction.get_num_spins() + 1);

    for (typename SparseMatrixXx::InnerIterator it(
             interaction, init_interaction.get_num_spins());
         it; ++it) {
      std::size_t j = it.index();
      FloatType v = it.value();
      if (j != init_interaction.get_num_spins()) {
        diag.insert(j, j) = v;
      } else {
        diag.insert(j, j) = -1;
      }
    }

    interaction += diag;

    spin_config.push_back(std::vector<CutPoint>{CutPoint(0.0, 1)});
    // initialize auxiliary spin with 1 along entire timeline
  }

  /**
   * @brief ContinuousTimeIsing constructor
   *
   * @details create timeline which has only one cut at time zero with given
   * spin state for each site
   *
   * @param init_spins
   * @param init_interaction
   * @param gamma
   */
  ContinuousTimeIsing(const graph::Spins &init_spins,
                      const GraphType &init_interaction, const double gamma)
      : ContinuousTimeIsing(convert_to_spin_config(init_spins),
                            init_interaction, gamma) {
  } // constructor delegation

  /* initialization helper functions */
private:
  /**
   * @brief convert classical spins to spin configuration; each site has only
   * one cut with given spin state at time zero
   *
   * @param spins
   */
  static SpinConfiguration convert_to_spin_config(const graph::Spins &spins) {
    SpinConfiguration spin_config;
    spin_config.reserve(spins.size());

    /* set spin configuration for each site, which has only one cut at time zero
     */
    for (auto spin : spins) {
      spin_config.push_back(std::vector<CutPoint>{
          CutPoint(TimeType(), spin)}); // TimeType() is zero value of the type
    }

    return spin_config;
  }

  /* member functions*/
public:
  /**
   * @brief reset spins with given spin configuration
   *
   * @param init_spin_config spin configuration to be set, which must NOT
   * contain auxiliary one
   */
  void reset_spins(const SpinConfiguration &init_spin_config) {
    assert(init_spin_config.size() == this->num_spins - 1);

    this->spin_config = init_spin_config;
    this->spin_config.push_back(std::vector<CutPoint>{CutPoint(TimeType(), 1)});
    // add auxiliary timeline
  }

  /**
   * @brief reset spins with given spin configuration
   *
   * @param classical_spins
   */
  void reset_spins(const graph::Spins &classical_spins) {
    assert(classical_spins.size() == this->num_spins - 1);

    for (size_t i = 0; i < this->num_spins - 1; i++) {
      this->spin_config[i] = std::vector<CutPoint>{
          CutPoint(TimeType(),
                   classical_spins[i]) // TimeType() is zero value of the type
      };
    }
    this->spin_config[this->num_spins - 1] =
        std::vector<CutPoint>{CutPoint(TimeType(), 1)};
  }

  /**
   * @brief return time-direction index which exists just before time_point at
   * "site_index"th site. The periodic boundary condition for time direction is
   * taken into account.
   *
   * @param site_index spacial index of site
   * @param time_point time-direction point
   */
  size_t get_temporal_spin_index(graph::Index site_index,
                                 TimeType time_point) const {
    static const auto first_lt = [](CutPoint x, CutPoint y) {
      return x.first < y.first;
    };
    // function to compare two time points (lt; less than)

    const auto &timeline = this->spin_config[site_index];
    const auto dummy_cut =
        CutPoint(time_point, 0); // dummy variable for binary search
    auto found_itr =
        std::upper_bound(timeline.begin(), timeline.end(), dummy_cut, first_lt);

    if (found_itr ==
        timeline.begin()) { // if the time_point lies before any time points
      found_itr = timeline.end() - 1; // periodic boundary condition
    } else {
      found_itr--;
    }

    return std::distance(timeline.begin(), found_itr);
  }

  /*
   * @brief return spin configuration at given temporal slice, not containing
   * auxiliary spin
   *
   * @param slice_time
   */
  graph::Spins get_slice_at(TimeType slice_time) const {
    graph::Spins slice;

    for (graph::Index i = 0; i < this->spin_config.size() - 1; i++) {
      auto temporal_index = get_temporal_spin_index(i, slice_time);
      slice.push_back(this->spin_config[i][temporal_index].second);
    }

    return slice;
  }

  /*
   * @brief return auxiliary spin state at given temporal slice
   *
   * @param slice_time
   */
  graph::Spin get_auxiliary_spin(TimeType slice_time) const {
    auto last_index = this->spin_config.size() - 1;
    auto temporal_index = get_temporal_spin_index(last_index, slice_time);
    return this->spin_config[last_index][temporal_index].second;
  }

  /* Member variables */

  /**
   * @brief spin configuration
   */
  SpinConfiguration spin_config;

  /**
   * @brief number of spins, including auxiliary spin for longitudinal magnetic
   * field
   */
  const std::size_t num_spins;

  /**
   * @brief interaction
   */
  SparseMatrixXx interaction;

  /**
   * @brief coefficient of transverse field term, actual field would be gamma *
   * s, where s = [0:1]
   */
  const FloatType gamma;
};

/**
 * @brief Continuous Time Quantum Ising system (for CSR Sparse graph)
 *
 * @tparam FloatType
 */
template <typename FloatType>
struct ContinuousTimeIsing<graph::CSRSparse<FloatType>> {
  using GraphType = graph::CSRSparse<FloatType>;
  using system_type = transverse_field_system;
  using TimeType = FloatType;
  using CutPoint = std::pair<TimeType, graph::Spin>;

  /**
   * @brief Interaction type (Eigen sparse matrix)
   */
  using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;

  /**
   * @brief spin configuration in real and continuous time space
   * spin_config[i][j] -> at i th site, j th pair of imaginary time point and
   * spin value after the point
   */
  using SpinConfiguration = std::vector<std::vector<CutPoint>>;

  /**
   * @brief ContinuousTimeIsing constructor
   *
   * @param init_spin_config
   * @param init_interaction
   * @param gamma
   */
  ContinuousTimeIsing(const SpinConfiguration &init_spin_config,
                      const GraphType &init_interaction, const double gamma)
      : spin_config(init_spin_config), num_spins(init_spin_config.size() + 1),
        interaction(init_interaction.get_interactions()),
        gamma(gamma) {

    assert(init_spin_config.size() == init_interaction.get_num_spins());

    // insert numbers to diagnonal elements in the interaction
    SparseMatrixXx diag(init_interaction.get_num_spins() + 1,
                        init_interaction.get_num_spins() + 1);

    for (typename SparseMatrixXx::InnerIterator it(
             interaction, init_interaction.get_num_spins());
         it; ++it) {
      std::size_t j = it.index();
      FloatType v = it.value();
      if (j != init_interaction.get_num_spins()) {
        diag.insert(j, j) = v;
      } else {
        diag.insert(j, j) = -1;
      }
    }

    interaction += diag;

    spin_config.push_back(std::vector<CutPoint>{CutPoint(0.0, 1)});
    // initialize auxiliary spin with 1 along entire timeline
  }

  /**
   * @brief ContinuousTimeIsing constructor
   *
   * @details create timeline which has only one cut at time zero with given
   * spin state for each site
   *
   * @param init_spins
   * @param init_interaction
   * @param gamma
   */
  ContinuousTimeIsing(const graph::Spins &init_spins,
                      const GraphType &init_interaction, const double gamma)
      : ContinuousTimeIsing(convert_to_spin_config(init_spins),
                            init_interaction, gamma) {
  } // constructor delegation

  /* initialization helper functions */
private:
  /**
   * @brief convert classical spins to spin configuration; each site has only
   * one cut with given spin state at time zero
   *
   * @param spins
   */
  static SpinConfiguration convert_to_spin_config(const graph::Spins &spins) {
    SpinConfiguration spin_config;
    spin_config.reserve(spins.size());

    /* set spin configuration for each site, which has only one cut at time zero
     */
    for (auto spin : spins) {
      spin_config.push_back(std::vector<CutPoint>{
          CutPoint(TimeType(), spin)}); // TimeType() is zero value of the type
    }

    return spin_config;
  }

  /* member functions*/
public:
  /**
   * @brief reset spins with given spin configuration
   *
   * @param init_spin_config spin configuration to be set, which must NOT
   * contain auxiliary one
   */
  void reset_spins(const SpinConfiguration &init_spin_config) {
    assert(init_spin_config.size() == this->num_spins - 1);

    this->spin_config = init_spin_config;
    this->spin_config.push_back(std::vector<CutPoint>{CutPoint(TimeType(), 1)});
    // add auxiliary timeline
  }

  /**
   * @brief reset spins with given spin configuration
   *
   * @param classical_spins
   */
  void reset_spins(const graph::Spins &classical_spins) {
    assert(classical_spins.size() == this->num_spins - 1);

    for (size_t i = 0; i < this->num_spins - 1; i++) {
      this->spin_config[i] = std::vector<CutPoint>{
          CutPoint(TimeType(),
                   classical_spins[i]) // TimeType() is zero value of the type
      };
    }
    this->spin_config[this->num_spins - 1] =
        std::vector<CutPoint>{CutPoint(TimeType(), 1)};
  }

  /**
   * @brief return time-direction index which exists just before time_point at
   * "site_index"th site. The periodic boundary condition for time direction is
   * taken into account.
   *
   * @param site_index spacial index of site
   * @param time_point time-direction point
   */
  size_t get_temporal_spin_index(graph::Index site_index,
                                 TimeType time_point) const {
    static const auto first_lt = [](CutPoint x, CutPoint y) {
      return x.first < y.first;
    };
    // function to compare two time points (lt; less than)

    const auto &timeline = this->spin_config[site_index];
    const auto dummy_cut =
        CutPoint(time_point, 0); // dummy variable for binary search
    auto found_itr =
        std::upper_bound(timeline.begin(), timeline.end(), dummy_cut, first_lt);

    if (found_itr ==
        timeline.begin()) { // if the time_point lies before any time points
      found_itr = timeline.end() - 1; // periodic boundary condition
    } else {
      found_itr--;
    }

    return std::distance(timeline.begin(), found_itr);
  }

  /*
   * @brief return spin configuration at given temporal slice, not containing
   * auxiliary spin
   *
   * @param slice_time
   */
  graph::Spins get_slice_at(TimeType slice_time) const {
    graph::Spins slice;

    for (graph::Index i = 0; i < this->spin_config.size() - 1; i++) {
      auto temporal_index = get_temporal_spin_index(i, slice_time);
      slice.push_back(this->spin_config[i][temporal_index].second);
    }

    return slice;
  }

  /*
   * @brief return auxiliary spin state at given temporal slice
   *
   * @param slice_time
   */
  graph::Spin get_auxiliary_spin(TimeType slice_time) const {
    auto last_index = this->spin_config.size() - 1;
    auto temporal_index = get_temporal_spin_index(last_index, slice_time);
    return this->spin_config[last_index][temporal_index].second;
  }

  /* Member variables */

  /**
   * @brief spin configuration
   */
  SpinConfiguration spin_config;

  /**
   * @brief number of spins, including auxiliary spin for longitudinal magnetic
   * field
   */
  const std::size_t num_spins;

  /**
   * @brief interaction
   */
  SparseMatrixXx interaction;

  /**
   * @brief coefficient of transverse field term, actual field would be gamma *
   * s, where s = [0:1]
   */
  const FloatType gamma;
};

/**
 * @brief helper function for ContinuousTimeIsing constructor
 *
 * @tparam GraphType
 * @param init_spins
 * @param interaction
 * @param gamma
 *
 * @return generated object
 */
template <typename GraphType>
ContinuousTimeIsing<GraphType>
make_continuous_time_ising(const graph::Spins &init_spins,
                           const GraphType &init_interaction, double gamma) {
  return ContinuousTimeIsing<GraphType>(init_spins, init_interaction, gamma);
}
} // namespace system
} // namespace openjij
