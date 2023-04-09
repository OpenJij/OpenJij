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

#include <random>
#include <type_traits>

#include "openjij/system/classical_ising.hpp"
#include "openjij/system/transverse_ising.hpp"
#include "openjij/utility/schedule_list.hpp"
#include "openjij/algorithm/algorithm.hpp"

#ifdef USE_OMP
#include <omp.h>
#endif

namespace openjij {
namespace updater {

/**
 * @brief naive single spin flip updater
 *
 * @tparam System type of system
 */
template <typename System> struct SingleSpinFlip;

/**
 * @brief single spin flip for classical ising model (with Eigen implementation)
 *
 * @tparam GraphType graph type (assume Dense<FloatType> or Sparse<FloatType>)
 */
template <typename GraphType>
struct SingleSpinFlip<system::ClassicalIsing<GraphType>> {

  /**
   * @brief ClassicalIsing with dense interactions
   */
  using ClIsing = system::ClassicalIsing<GraphType>;

  /**
   * @brief float type
   */
  using FloatType = typename GraphType::value_type;

  /**
   * @brief operate single spin flip in a classical ising system
   *
   * @param system object of a classical ising system
   * @param random_number_engine random number gengine
   * @param parameter parameter object including inverse temperature
   * \f\beta:=(k_B T)^{-1}\f
   *
   * @return energy difference \f\Delta E\f
   */
  template <typename RandomNumberEngine>
  inline static void
  update(ClIsing &system, RandomNumberEngine &random_number_engine,
         const utility::ClassicalUpdaterParameter &parameter) {
    // set probability distribution object
    // to do Metroopolis
    auto urd = std::uniform_real_distribution<>(0, 1.0);

    Eigen::setNbThreads(1);
    Eigen::initParallel();

    // do a iteraction except for the auxiliary spin
    for (std::size_t index = 0; index < system.num_spins; ++index) {

      if (system.dE(index) <= 0 ||
          std::exp(-parameter.beta * system.dE(index)) >
              urd(random_number_engine)) {
        // update dE
        system.dE += 4 * system.spin(index) *
                     (system.interaction.row(index).transpose().cwiseProduct(
                         system.spin));

        system.dE(index) *= -1;
        system.spin(index) *= -1;
      }

      // assure that the dummy spin is not changed.
      system.spin(system.num_spins) = 1;
    }
  }
};

/**
 * @brief single spin flip for transverse field ising model (with Eigen
 * implementation)
 *
 * @tparam GraphType graph type (assume Dense<FloatType> or Sparse<FloatType>)
 */
template <typename GraphType>
struct SingleSpinFlip<system::TransverseIsing<GraphType>> {

  /**
   * @brief transverse field ising system
   */
  using QIsing = system::TransverseIsing<GraphType>;

  /**
   * @brief float type
   */
  using FloatType = typename GraphType::value_type;

  /**
   * @brief operate single spin flip in a transverse ising system
   *
   * @param system object of a transverse ising system
   * @param random_number_engine random number engine
   * @param parameter parameter object including inverse temperature
   * \f\beta:=(k_B T)^{-1}\f and transverse magnetic field \f\s\f
   *
   * @return energy difference \f\Delta E\f
   */
  template <typename RandomNumberEngine>
  inline static void
  update(QIsing &system, RandomNumberEngine &random_number_engine,
         const utility::TransverseFieldUpdaterParameter &parameter) {

    // get number of classical spins
    std::size_t num_classical_spins = system.num_classical_spins;
    // get number of trotter slices
    std::size_t num_trotter_slices = system.trotter_spins.cols();

    // do metropolis
    auto urd = std::uniform_real_distribution<>(0, 1.0);

    // aliases
    // const auto& spins = system.trotter_spins;
    const auto &gamma = system.gamma;
    const auto &beta = parameter.beta;
    const auto &s = parameter.s;

    // transverse factor
    FloatType B =
        (1 / 2.) * log(tanh(beta * gamma * (1.0 - s) / num_trotter_slices));

    Eigen::setNbThreads(1);
    Eigen::initParallel();

    // generate random number (col major)
    for (std::size_t t = 0; t < num_trotter_slices; t++) {
      for (std::size_t i = 0; i < num_classical_spins; i++) {
        system.rand_pool(i, t) = urd(random_number_engine);
      }
    }

    // using OpenMP

    // we have to consider the case num_trotter_slices is odd.
    std::size_t upper_limit = num_trotter_slices % 2 != 0
                                  ? num_trotter_slices - 1
                                  : num_trotter_slices;

    // NOTE: only signed integer is allowed for OpenMP
#pragma omp parallel for
    for (int64_t t = 0; t < (int64_t)upper_limit; t += 2) {
      for (int64_t i = 0; i < (int64_t)num_classical_spins; i++) {
        // calculate matrix dot product
        do_calc(system, parameter, i, t, B);
      }
    }

    // using OpenMP
#pragma omp parallel for
    for (int64_t t = 1; t < (int64_t)num_trotter_slices; t += 2) {
      for (int64_t i = 0; i < (int64_t)num_classical_spins; i++) {
        // calculate matrix dot product
        do_calc(system, parameter, i, t, B);
      }
    }

    // for the case num_trotter_slices is odd.
    if (num_trotter_slices % 2 != 0) {
      int64_t t = (int64_t)(num_trotter_slices - 1);
      for (int64_t i = 0; i < (int64_t)num_classical_spins; i++) {
        // calculate matrix dot product
        do_calc(system, parameter, i, t, B);
      }
    }
  }

private:
  inline static void
  do_calc(QIsing &system,
          const utility::TransverseFieldUpdaterParameter &parameter, size_t i,
          size_t t, FloatType B) {

    // get number of trotter slices
    std::size_t num_trotter_slices = system.trotter_spins.cols();

    // aliases
    auto &spins = system.trotter_spins;
    // const auto& gamma = system.gamma;
    const auto &beta = parameter.beta;
    const auto &s = parameter.s;

    // calculate dE for trotter direction
    FloatType dEtrot = -2 * spins(i, t) *
                       (spins(i, mod_t((int64_t)t + 1, num_trotter_slices)) +
                        spins(i, mod_t((int64_t)t - 1, num_trotter_slices)));

    // calculate total dE
    FloatType dE =
        s * (beta / num_trotter_slices) * system.dE(i, t) + B * dEtrot;

    // for debugging

    // FloatType testdE = 0;

    ////do metropolis
    // testdE += -2 * s * (beta/num_trotter_slices) * spins(i,
    // t)*(system.interaction.row(i).dot(spins.col(t)));

    ////trotter direction
    // testdE += -2 * (1/2.) * log(tanh(beta* gamma * (1.0-s)
    // /num_trotter_slices)) * spins(i, t) *
    //    (    spins(i, mod_t((int64_t)t+1, num_trotter_slices))
    //       + spins(i, mod_t((int64_t)t-1, num_trotter_slices)));

    // std::cout << "s=" << s << std::endl;
    // std::cout << "dE=" << dE << std::endl;
    // std::cout << "testdE=" << testdE << std::endl;
    // assert((std::isinf(dE) && std::isinf(testdE)) || abs(dE-testdE) < 1e-5);

    // metropolis

    if (dE < 0 || exp(-dE) > system.rand_pool(i, t)) {

      // update dE (spatial direction)
      system.dE.col(t) +=
          4 * spins(i, t) *
          (system.interaction.row(i).transpose().cwiseProduct(spins.col(t)));
      system.dE(i, t) *= -1;

      // update spins
      spins(i, t) *= -1;
    }
  }

  inline static std::size_t mod_t(std::int64_t a,
                                  std::size_t num_trotter_slices) {
    // a -> [-1:num_trotter_slices]
    // return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
    return (a + num_trotter_slices) % num_trotter_slices;
  }
};

//! @brief Single spin flip for Ising models with polynomial interactions and
//! polynomial unconstrained binary optimization models.
//! @tparam GraphType graph type for Polynomial graph class
template <typename GraphType>
struct SingleSpinFlip<system::ClassicalIsingPolynomial<GraphType>> {

  //! @brief ClassicalIsingPolynomial system
  using ClPIsing = system::ClassicalIsingPolynomial<GraphType>;

  //! @brief floating point type
  using FloatType = typename GraphType::value_type;

  //! @brief Operate single spin flip for Ising models with polynomial
  //! interactions and polynomial unconstrained binary optimization models.
  //! @param system ClPIsing&. Object of a ClassicalIsingPolynomial system.
  //! @param random_number_engine RandomNumberEngine&. Eandom number engine.
  //! @param parameter const utility::ClassicalUpdaterParameter&. Parameter
  //! object including inverse temperature \f\beta:=(k_B T)^{-1}\f.
  template <typename RandomNumberEngine>
  inline static void
  update(ClPIsing &system, RandomNumberEngine &random_number_engine,
         const utility::ClassicalUpdaterParameter &parameter) {

    if (system.vartype == cimod::Vartype::SPIN) {
      update_spin<RandomNumberEngine>(system, random_number_engine, parameter);
    } else if (system.vartype == cimod::Vartype::BINARY) {
      update_binary<RandomNumberEngine>(system, random_number_engine,
                                        parameter);
    } else {
      std::stringstream ss;
      ss << "Unknown vartype detected in " << __func__ << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  //! @brief Operate single spin flip when the system is composed of spin
  //! variables.
  //! @param system ClPIsing&. ClassicalIsingPolynomial system.
  //! @param random_number_engine RandomNumberEngine&. Eandom number engine.
  //! @param parameter const utility::ClassicalUpdaterParameter&. Parameter
  //! object including inverse temperature \f\beta:=(k_B T)^{-1}\f.
  template <typename RandomNumberEngine>
  inline static void
  update_spin(ClPIsing &system, RandomNumberEngine &random_number_engine,
              const utility::ClassicalUpdaterParameter &parameter) {

    auto urd = std::uniform_real_distribution<>(0, 1.0);
    for (const auto &index_spin : system.get_active_variables()) {
      if (system.dE(index_spin) <= 0.0 ||
          std::exp(-parameter.beta * system.dE(index_spin)) >
              urd(random_number_engine)) {
        system.update_spin_system(index_spin);
      }
    }
  }

  //! @brief Operate single spin flip when the system is composed of binary
  //! variables.
  //! @param system ClPIsing&. ClassicalIsingPolynomial system.
  //! @param random_number_engine RandomNumberEngine&. Random number engine.
  //! @param parameter const utility::ClassicalUpdaterParameter&. Parameter
  //! object including inverse temperature \f\beta:=(k_B T)^{-1}\f.
  template <typename RandomNumberEngine>
  inline static void
  update_binary(ClPIsing &system, RandomNumberEngine &random_number_engine,
                const utility::ClassicalUpdaterParameter &parameter) {

    auto urd = std::uniform_real_distribution<>(0, 1.0);
    for (const auto &index_binary : system.get_active_variables()) {
      if (system.dE(index_binary) <= 0.0 ||
          std::exp(-parameter.beta * system.dE(index_binary)) >
              urd(random_number_engine)) {
        system.update_binary_system(index_binary);
      }
    }
  }
};


template<class SystemType, typename RandType>
void SingleFlipUpdater(SystemType *system,
                       const std::int32_t num_sweeps,
                       const std::vector<typename SystemType::ValueType> &beta_list,
                       const typename RandType::result_type seed,
                       const algorithm::UpdateMethod update_metod) {
   
   const std::int32_t system_size = system->GetSystemSize();
   
   // Set random number engine
   RandType random_number_engine(seed);
   std::uniform_real_distribution<typename SystemType::ValueType> dist_real(0, 1);
   
   if (update_metod == algorithm::UpdateMethod::METROPOLIS) {
      // Do sequential update
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         const auto beta = beta_list[sweep_count];
         for (std::int32_t i = 0; i < system_size; i++) {
            const auto delta_energy = system->GetEnergyDifference(i);
            if (delta_energy <= 0 || std::exp(-beta*delta_energy) > dist_real(random_number_engine)) {
               system->Flip(i);
            }
         }
      }
   }
   else if (update_metod == algorithm::UpdateMethod::HEAT_BATH) {
      // Do sequential update
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         const auto beta = beta_list[sweep_count];
         for (std::int32_t i = 0; i < system_size; i++) {
            const auto delta_energy = system->GetEnergyDifference(i);
            if (1/(1 + std::exp(beta*delta_energy)) > dist_real(random_number_engine)) {
               system->Flip(i);
            }
         }
      }
   }
   else {
      throw std::runtime_error("Unknown UpdateMethod");
   }
}

} // namespace updater
} // namespace openjij
