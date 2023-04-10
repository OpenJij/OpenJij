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
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "openjij/system/system.hpp"

namespace openjij {
namespace utility {

/**
 * @brief updater parameter for monte carlo simulation
 *
 * @tparam SystemType system type
 */
template <typename SystemType> struct UpdaterParameter;

/**
 * @brief updater parameter for classical ising system
 */
template <> struct UpdaterParameter<system::classical_system> {
  using Tuple = double;
  UpdaterParameter() = default;
  UpdaterParameter(Tuple beta) : beta{beta} {}
  inline Tuple get_tuple() const { return beta; }

  /**
   * @brief inverse temperature
   */
  double beta;
};

/**
 * @brief updater paramter for classical ising model with a single constraint
 */
template <> struct UpdaterParameter<system::classical_constraint_system> {
  using Tuple = std::pair<double, double>;
  UpdaterParameter() = default;
  UpdaterParameter(double beta, double lambda) : beta{beta}, lambda{lambda} {}
  UpdaterParameter(const Tuple &obj)
      : UpdaterParameter(obj.first, obj.second) {}
  inline Tuple get_tuple() const { return std::make_pair(beta, lambda); }

  /**
   * @brief inverse temperature
   */
  double beta;

  /**
   * @brief constraint coefficient
   */
  double lambda;
};

/**
 * @brief updater paramter for transverse ising model
 */
template <> struct UpdaterParameter<system::transverse_field_system> {
  using Tuple = std::pair<double, double>;
  UpdaterParameter() = default;
  UpdaterParameter(double beta, double s) : beta{beta}, s{s} {}
  UpdaterParameter(const Tuple &obj)
      : UpdaterParameter(obj.first, obj.second) {}
  inline Tuple get_tuple() const { return std::make_pair(beta, s); }

  /**
   * @brief inverse temperature
   */
  double beta;

  /**
   * @brief annealing schedule (from 0 (only transverse field) to 1 (only
   * classical Hamiltonian))
   */
  double s;
};

// TODO: add UpdaterParameter here if needed.
//

/**
 * @brief ClassicalUpdaterParameter alias
 */
using ClassicalUpdaterParameter = UpdaterParameter<system::classical_system>;

/**
 * @brief ClassicalUpdaterParameter alias
 */
using ClassicalConstraintUpdaterParameter =
    UpdaterParameter<system::classical_constraint_system>;

/**
 * @brief TransverseFieldUpdaterParameter alias
 */
using TransverseFieldUpdaterParameter =
    UpdaterParameter<system::transverse_field_system>;

// TODO: the above class is only for monte carlo system, add enable_if.

/**
 * @brief schedule struct
 *
 * @tparam SystemType system type
 */
template <typename SystemType> struct Schedule {
  Schedule() = default;
  Schedule(const std::pair<UpdaterParameter<SystemType>, std::size_t> &obj)
      : updater_parameter(obj.first), one_mc_step(obj.second) {}
  UpdaterParameter<SystemType> updater_parameter;
  std::size_t one_mc_step;
};

/**
 * @brief schedule list alias
 *
 * @tparam SystemType system type
 */
template <typename SystemType>
using ScheduleList = std::vector<Schedule<SystemType>>;

/**
 * @brief ClassicalScheduleList alias
 */
using ClassicalScheduleList = ScheduleList<system::classical_system>;

/**
 * @brief TransverseFieldScheduleList alias
 */
using TransverseFieldScheduleList =
    ScheduleList<system::transverse_field_system>;

/**
 * @brief ClassicalConstraintScheduleList alias
 */
using ClassicalConstraintScheduleList =
    ScheduleList<system::classical_constraint_system>;

/**
 * @brief helper function for making classical schedule list with geometric
 * series of inverse temperatures.
 *
 * @param beta_min initial (minimal) value of beta
 * @param beta_max final (maximal) value of beta
 * @param one_mc_step number of mc step for each temperature
 * @param num_call_updater number of updater calls
 *
 * @return generated list
 */
ClassicalScheduleList
make_classical_schedule_list(double beta_min, double beta_max,
                             std::size_t one_mc_step,
                             std::size_t num_call_updater) {
  double r_beta = std::pow(beta_max / beta_min,
                           1.0 / static_cast<double>(num_call_updater - 1));
  double beta = beta_min;

  auto schedule_list = ClassicalScheduleList(num_call_updater);
  for (auto &schedule : schedule_list) {
    schedule.one_mc_step = one_mc_step;
    schedule.updater_parameter = ClassicalUpdaterParameter(beta);
    beta *= r_beta;
  }

  return schedule_list;
}

/**
 * @brief helper function for making transverse field system schedule list with
 * arithmetic sequence of annealing schedule (s)
 *
 * @param beta inverse temperature
 * @param one_mc_step number of mc step for each temperature
 * @param num_call_updater number of updater calls
 *
 * @return
 */
TransverseFieldScheduleList
make_transverse_field_schedule_list(double beta, std::size_t one_mc_step,
                                    std::size_t num_call_updater) {
  double ds = 1.0 / static_cast<double>(num_call_updater - 1);
  double s = 0;

  auto schedule_list = TransverseFieldScheduleList(num_call_updater);
  for (auto &schedule : schedule_list) {
    schedule.one_mc_step = one_mc_step;
    schedule.updater_parameter = TransverseFieldUpdaterParameter(beta, s);
    s += ds;
  }

  return schedule_list;
}

/**
 * @brief helper function for making classical constraint schedule list with
 * geometric series of inverse temperatures (with constraint parameter fixed).
 *
 * @param lambda constraint paraemeters
 * @param beta_min
 * @param beta_max
 * @param one_mc_step
 * @param num_call_updater
 *
 * @return generated schedule list
 */
ClassicalConstraintScheduleList make_classical_constraint_schedule_list(
    double lambda, double beta_min, double beta_max, std::size_t one_mc_step,
    std::size_t num_call_updater) {
  double r_beta = std::pow(beta_max / beta_min,
                           1.0 / static_cast<double>(num_call_updater - 1));
  double beta = beta_min;

  auto schedule_list = ClassicalConstraintScheduleList(num_call_updater);
  for (auto &schedule : schedule_list) {
    schedule.one_mc_step = one_mc_step;
    schedule.updater_parameter =
        ClassicalConstraintUpdaterParameter(beta, lambda);
    beta *= r_beta;
  }

  return schedule_list;
}

/**
 * @brief helper function for making schedulelist from list of tuples
 *
 * @tparam SystemType
 * @param tuplelist
 *
 * @return
 */
template <typename SystemType>
ScheduleList<SystemType> make_schedule_list(
    const std::vector<std::pair<typename UpdaterParameter<SystemType>::Tuple,
                                std::size_t>> &tuplelist) {
  ScheduleList<SystemType> return_list;
  return_list.reserve(tuplelist.size());
  for (auto &elem : tuplelist) {
    return_list.emplace_back(std::make_pair(elem.first, elem.second));
  }
  return return_list;
}

enum class TemperatureSchedule {
  
   //! @brief Linear cooling
   LINEAR,
   
   //! @brief Geometric cooling
   GEOMETRIC,
   
};

//! @brief Generate linear temperature schedule
//! @tparam FloatType Floating point type
//! @param beta_min The minimum value of inverse temperature
//! @param beta_max The maximum value of inverse temperature
//! @param num_sweeps The number of sweeps
//! @return Linear temperature schedule
template<typename FloatType>
std::vector<FloatType> GenerateLinearBetaSchedule(const FloatType beta_min, const FloatType beta_max, const std::int32_t num_sweeps) {
   if (num_sweeps == 1) {
      return std::vector<FloatType>{beta_min};
   }
   std::vector<FloatType> beta_list(num_sweeps);
   for (std::int32_t i = 0; i < num_sweeps; ++i) {
      beta_list[i] = beta_min + i*(beta_max - beta_min)/(num_sweeps - 1);
   }
   return beta_list;
}

//! @brief Generate geometric temperature schedule
//! @tparam FloatType Floating point type
//! @param beta_min The minimum value of inverse temperature
//! @param beta_max The maximum value of inverse temperature
//! @param num_sweeps The number of sweeps
//! @return Geometric temperature schedule
template<typename FloatType>
std::vector<FloatType> GenerateGeometricBetaSchedule(const FloatType beta_min, const FloatType beta_max, const std::int32_t num_sweeps) {
   if (num_sweeps == 1) {
      return std::vector<FloatType>{beta_min};
   }
   std::vector<FloatType> beta_list(num_sweeps);
   const FloatType alpha = std::pow(beta_max/beta_min, 1/static_cast<FloatType>(num_sweeps - 1));
   FloatType beta = beta_min;
   for (std::int32_t i = 0; i < num_sweeps; ++i) {
      beta_list[i] = beta;
      beta = beta*alpha;
   }
   return beta_list;
}

//! @brief Generate temperature schedule from specific schedule
//! @tparam FloatType Floating point type
//! @param schedule_type The type of temperature schedule
//! @param beta_min The minimum value of inverse temperature
//! @param beta_max The maximum value of inverse temperature
//! @param num_sweeps The number of sweeps
//! @return Temperature schedule
template<typename FloatType>
std::vector<FloatType> GenerateBetaList(const TemperatureSchedule schedule_type,
                                        const FloatType beta_min,
                                        const FloatType beta_max,
                                        const std::int32_t num_sweeps) {
   std::vector<FloatType> beta_list;
   if (schedule_type == TemperatureSchedule::LINEAR) {
      beta_list = GenerateLinearBetaSchedule(beta_min, beta_max, num_sweeps);
   }
   else if (schedule_type == TemperatureSchedule::GEOMETRIC) {
      beta_list = GenerateGeometricBetaSchedule(beta_min, beta_max, num_sweeps);
   }
   else {
      throw std::runtime_error("Unknwon beta schedule list");
   }
   return beta_list;
}


} // namespace utility
} // namespace openjij
