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

#include "openjij/graph/all.hpp"
#include "openjij/updater/all.hpp"
#include "openjij/system/all.hpp"

namespace openjij {
namespace sampler {

//! @brief Class for executing simulated annealing.
//! @tparam ModelType The type of models.
template<class ModelType>
class SASampler {
   
   //! @brief The value type.
   using ValueType = typename ModelType::ValueType;
      
   //! @brief The variable type
   using VariableType = typename ModelType::VariableType;
   
public:
   //! @brief Constructor for SASampler class.
   //! @param model The model.
   SASampler(const ModelType &model): model_(model) {}
   
   //! @brief Set the number of sweeps.
   //! @param num_sweeps The number of sweeps, which must be larger than zero.
   void SetNumSweeps(const std::int32_t num_sweeps) {
      if (num_sweeps <= 0) {
         throw std::runtime_error("num_sweeps must be larger than zero.");
      }
      num_sweeps_ = num_sweeps;
   }
   
   //! @brief Set the number of samples.
   //! @param num_reads The number of samples, which must be larger than zero.
   void SetNumReads(const std::int32_t num_reads) {
      if (num_reads <= 0) {
         throw std::runtime_error("num_reads must be larger than zero.");
      }
      num_reads_ = num_reads;
   }
   
   //! @brief Set the number of threads in the calculation.
   //! @param num_threads The number of threads in the calculation, which must be larger than zero.
   void SetNumThreads(const std::int32_t num_threads) {
      if (num_threads <= 0) {
         throw std::runtime_error("num_threads must be non-negative integer.");
      }
      num_threads_ = num_threads;
   }
   
   //! @brief Set the minimum inverse temperature.
   //! @param beta_min The minimum inverse temperature, which must be larger than zero.
   void SetBetaMin(const ValueType beta_min) {
      if (beta_min < 0) {
         throw std::runtime_error("beta_min must be positive number");
      }
      beta_min_ = beta_min;
   }
   
   //! @brief Set the minimum inverse temperature.
   //! @param beta_max The minimum inverse temperature, which must be larger than zero.
   void SetBetaMax(const ValueType beta_max) {
      if (beta_max < 0) {
         throw std::runtime_error("beta_max must be positive number");
      }
      beta_max_ = beta_max;
   }
   
   //! @brief Set the minimum inverse temperature automatically.
   void SetBetaMinAuto() {
      beta_min_ = std::log(2.0)/model_.GetEstimatedMaxEnergyDifference();
   }
   
   //! @brief Set the maximum inverse temperature automatically.
   void SetBetaMaxAuto() {
      beta_max_ = std::log(100.0)/model_.GetEstimatedMinEnergyDifference();
   }
      
   //! @brief Set update method used in the state update.
   //! @param update_method The update method.
   void SetUpdateMethod(const algorithm::UpdateMethod update_method) {
      update_method_ = update_method;
   }
   
   //! @brief Set random number engine for updating initializing state.
   //! @param random_number_engine The random number engine.
   void SetRandomNumberEngine(const algorithm::RandomNumberEngine random_number_engine) {
      random_number_engine_ = random_number_engine;
   }
   
   //! @brief Set the cooling schedule.
   //! @param schedule The cooling schedule.
   void SetTemperatureSchedule(const utility::TemperatureSchedule schedule) {
      schedule_ = schedule;
   }
         
   //! @brief Get the model.
   //! @return The model.
   const ModelType &GetModel() const {
      return model_;
   }
   
   //! @brief Get the number of sweeps.
   //! @return The number of sweeps.
   std::int32_t GetNumSweeps() const {
      return num_sweeps_;
   }
   
   //! @brief Get the number of reads.
   //! @return The number of reads.
   std::int32_t GetNumReads() const {
      return num_reads_;
   }
   
   //! @brief Get the number of threads.
   //! @return The number of threads.
   std::int32_t GetNumThreads() const {
      return num_threads_;
   }
   
   //! @brief Get the minimum inverse temperature.
   //! @return The minimum inverse temperature.
   ValueType GetBetaMin() const {
      return beta_min_;
   }
   
   //! @brief Get the maximum inverse temperature.
   //! @return The maximum inverse temperature.
   ValueType GetBetaMax() const {
      return beta_max_;
   }
   
   //! @brief Get the update method used in the state update.
   //! @return The update method used in the state update.
   algorithm::UpdateMethod GetUpdateMethod() const {
      return update_method_;
   }
   
   //! @brief Get the random number engine for updating and initializing state.
   //! @return The random number engine for updating and initializing state.
   algorithm::RandomNumberEngine GetRandomNumberEngine() const {
      return random_number_engine_;
   }
   
   //! @brief Get the temperature schedule.
   //! @return The temperature schedule.
   utility::TemperatureSchedule GetTemperatureSchedule() const {
      return schedule_;
   }
   
   //! @brief Get the seed to be used in the calculation.
   //! @return The seed.
   std::uint64_t GetSeed() const {
      return seed_;
   }

   const std::vector<typename ModelType::IndexType> &GetIndexList() const {
      return model_.GetIndexList();
   }
   
   //! @brief Get the samples.
   //! @return The samples.
   const std::vector<std::vector<VariableType>> &GetSamples() const {
      return samples_;
   }
   
   std::vector<ValueType> CalculateEnergies() const {
      if (samples_.size() == 0) {
         throw std::runtime_error("The sample size is zero. It seems that sampling has not been carried out.");
      }
      std::vector<ValueType> energies(num_reads_);
      
      try {
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
         for (std::int32_t i = 0; i < num_reads_; ++i) {
            energies[i] = model_.CalculateEnergy(samples_[i]);
         }
      }
      catch (const std::exception &e) {
         std::cerr << e.what() << std::endl;
      }
      
      return energies;
   }
   
   //! @brief Execute sampling.
   //! Seed to be used in the calculation will be set automatically.
   void Sample() {
      Sample(std::random_device()());
   }
   
   //! @brief Execute sampling.
   //! @param seed The seed to be used in the calculation.
   void Sample(const std::uint64_t seed) {
      seed_ = seed;
      
      samples_.clear();
      samples_.shrink_to_fit();
      samples_.resize(num_reads_);
            
      if (random_number_engine_ == algorithm::RandomNumberEngine::XORSHIFT) {
         TemplateSampler<system::SASystem<ModelType, utility::Xorshift>, utility::Xorshift>();
      }
      else if (random_number_engine_ == algorithm::RandomNumberEngine::MT) {
         TemplateSampler<system::SASystem<ModelType, std::mt19937>, std::mt19937>();
      }
      else if (random_number_engine_ == algorithm::RandomNumberEngine::MT_64) {
         TemplateSampler<system::SASystem<ModelType, std::mt19937_64>, std::mt19937_64>();
      }
      else {
         throw std::runtime_error("Unknown RandomNumberEngine");
      }

   }
   
private:
   //! @brief The model.
   const ModelType model_;
   
   //! @brief The number of sweeps.
   std::int32_t num_sweeps_ = 1000;
   
   //! @brief The number of reads (samples).
   std::int32_t num_reads_ = 1;
   
   //! @brief The number of threads in the calculation.
   std::int32_t num_threads_ = 1;
   
   //! @brief The start inverse temperature.
   ValueType beta_min_ = 1;
   
   //! @brief The end inverse temperature.
   ValueType beta_max_ = 1;
      
   //! @brief The update method used in the state update.
   algorithm::UpdateMethod update_method_ = algorithm::UpdateMethod::METROPOLIS;
   
   //! @brief Random number engine for updating and initializing state.
   algorithm::RandomNumberEngine random_number_engine_ = algorithm::RandomNumberEngine::XORSHIFT;
   
   //! @brief Cooling schedule.
   utility::TemperatureSchedule schedule_ = utility::TemperatureSchedule::GEOMETRIC;
   
   //! @brief The seed to be used in the calculation.
   std::uint64_t seed_ = std::random_device()();
   
   //! @brief The samples.
   std::vector<std::vector<VariableType>> samples_;
   
   template<typename RandType>
   std::vector<std::pair<typename RandType::result_type, typename RandType::result_type>>
   GenerateSeedPairList(const typename RandType::result_type seed, const std::int32_t num_reads) const {
      RandType random_number_engine(seed);
      std::vector<std::pair<typename RandType::result_type, typename RandType::result_type>> seed_pair_list(num_reads);
      
      for (std::int32_t i = 0; i < num_reads; ++i) {
         seed_pair_list[i].first = random_number_engine();
         seed_pair_list[i].second = random_number_engine();
      }
      
      return seed_pair_list;
   }
   
   template<class SystemType, class RandType>
   void TemplateSampler() {
      const auto seed_pair_list = GenerateSeedPairList<RandType>(static_cast<typename RandType::result_type>(seed_), num_reads_);
      std::vector<ValueType> beta_list = utility::GenerateBetaList(schedule_, beta_min_, beta_max_, num_sweeps_);
      
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int32_t i = 0; i < num_reads_; ++i) {
         auto system = SystemType{model_, seed_pair_list[i].first};
         updater::SingleFlipUpdater<SystemType, RandType>(&system, num_sweeps_, beta_list, seed_pair_list[i].second, update_method_);
         samples_[i] = system.ExtractSample();
      }
   }
   
};

template<class ModelType>
auto make_sa_sampler(const ModelType &model) {
   return SASampler<ModelType>{model};
};



} //sampler
} //openjij
