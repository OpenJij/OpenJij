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

#include <functional>

#include "openjij/system/system.hpp"
#include "openjij/utility/schedule_list.hpp"
#include "openjij/utility/random.hpp"


namespace openjij {
namespace algorithm {
template <template <typename> class Updater> struct Algorithm {
  template <typename System, typename RandomNumberEngine>
  static void
  run(System &system, RandomNumberEngine &random_number_engine,
      const utility::ScheduleList<
          typename system::get_system_type<System>::type> &schedule_list,
      const std::function<void(
          const System &,
          const utility::UpdaterParameter<
              typename system::get_system_type<System>::type> &)> &callback =
          nullptr) {
    if (callback) {
      // with callback
      for (auto &&schedule : schedule_list) {
        for (std::size_t i = 0; i < schedule.one_mc_step; ++i) {
          Updater<System>::update(system, random_number_engine,
                                  schedule.updater_parameter);
          callback(system, schedule.updater_parameter);
        }
      }
    } else {
      // without callback
      for (auto &&schedule : schedule_list) {
        for (std::size_t i = 0; i < schedule.one_mc_step; ++i) {
          Updater<System>::update(system, random_number_engine,
                                  schedule.updater_parameter);
        }
      }
    }
  }
};

// type alias (Monte Carlo method)
// TODO: Algorithm class will be deprecated shortly.
template <template <typename> class Updater> using MCMC = Algorithm<Updater>;


enum class UpdateMethod {
   
   //! @brief Metropolis update
   METROPOLIS,
   
   //! @brief Heat bath update
   HEAT_BATH,
      
};

enum class RandomNumberEngine {
  
   //! @brief 32-bit Xorshift
   XORSHIFT,
   
   //! @brief 32-bit Mersenne Twister
   MT,
   
   //! @brief 64-bit Mersenne Twister
   MT_64
   
};


std::variant<utility::Xorshift, std::mt19937, std::mt19937_64>
GenerateRandomNumberEngineClass(const RandomNumberEngine random_number_engine) {
   if (random_number_engine == RandomNumberEngine::XORSHIFT) {
      return utility::Xorshift();
   }
   else if (random_number_engine == RandomNumberEngine::MT) {
      return std::mt19937();
   }
   else if (random_number_engine == RandomNumberEngine::MT) {
      return std::mt19937_64();
   }
   else {
      throw std::runtime_error("Unknown RandomNumberEngine");
   }
}

} // namespace algorithm
} // namespace openjij
