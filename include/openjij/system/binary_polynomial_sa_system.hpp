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

#include "openjij/graph/graph.hpp"
#include "./sa_system.hpp"

namespace openjij {
namespace system {

template<typename FloatType, typename RandType>
class SASystem<graph::BinaryPolynomialModel<FloatType>, RandType> {
   
   //! @brief The model type, which must be BinaryPolynomialModel.
   using ModelType = graph::BinaryPolynomialModel<FloatType>;
   
   //! @brief The variable type, which here represents binary variables \f$ x_i\in \{0, 1\} \f$
   using VariableType = typename ModelType::VariableType;
   
   //! @brief The type of seed in random number engine.
   using SeedType = typename RandType::result_type;
   
public:
   //! @brief The value type.
   using ValueType = typename ModelType::ValueType;
   
   //! @brief Constructor of SASystem for BinaryPolynomialModel.
   //! @param model The BinaryPolynomialModel.
   //! @param seed The seed for initializing binary variables.
   SASystem(const ModelType &model, const SeedType seed):
   system_size_(model.GetSystemSize()),
   key_value_list_(model.GetKeyValueList()),
   adjacency_list_(model.GetAdjacencyList()) {
      SetRandomConfiguration(seed);
      SetZeroCount();
      SetEnergyDifference();
   }
   
   //! @brief Constructor of SASystem for BinaryPolynomialModel.
   //! @param model The BinaryPolynomialModel.
   //! @param sample The initializing binary variables.
   SASystem(const ModelType &model, const std::vector<VariableType> &sample):
   system_size_(model.GetSystemSize()),
   key_value_list_(model.GetKeyValueList()),
   adjacency_list_(model.GetAdjacencyList()) {
      if (static_cast<std::int32_t>(sample.size()) != system_size_) {
         throw std::runtime_error("The size of initial variables is not equal to the system size.");
      }
      for (std::int32_t i = 0; i < system_size_; ++i) {
         if (!(sample[i] == 0 || sample[i] == 1)) {
            throw std::runtime_error("The initial variables must be 0 or 1.");
         }
      }
      sample_.resize(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         sample_[i] = sample[i];
      }
      SetZeroCount();
      SetEnergyDifference();
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   void Flip(const std::int32_t index, const VariableType candidate_state) {
      const VariableType update_binary = sample_[index];
      const std::int32_t coeef = -2*update_binary + 1;
      const std::int32_t count = +2*update_binary - 1;
      for (const auto &index_key : adjacency_list_[index]) {
         const ValueType val = key_value_list_[index_key].second;
         for (const auto &index_binary: key_value_list_[index_key].first) {
            if (zero_count_[index_key] + update_binary + sample_[index_binary] == 2 && index_binary != index) {
               energy_difference_[index_binary] += coeef*(-2*sample_[index_binary] + 1)*val;
            }
         }
         zero_count_[index_key] += count;
      }
      energy_difference_[index] *= -1;
      sample_[index] = candidate_state;
   }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
   //! @brief Extract the sample.
   //! @return The sample.
   const std::vector<VariableType> &ExtractSample() const {
      return sample_;
   }
   
   //! @brief Get the energy difference when flipped and sample as list.
   //! @return The energy difference and sample.
   const std::vector<ValueType> &GetEnergyDifference() const {
      return energy_difference_;
   }
   
   //! @brief Get the energy difference when flipped.
   //! @param index The index of variables.
   //! @return The energy difference.
   ValueType GetEnergyDifference(const std::int32_t index, const VariableType candidate_state) const {
      return energy_difference_[index];
   }
   
   VariableType GenerateCandidateState(const std::int32_t index) const {
      return 1 - sample_[index];
   }
   
private:
   const std::int32_t system_size_;
   const std::vector<std::pair<std::vector<std::int32_t>, ValueType>> &key_value_list_;
   const std::vector<std::vector<std::size_t>> &adjacency_list_;
   
   std::vector<VariableType> sample_;
   std::vector<ValueType> energy_difference_;
   std::vector<std::int32_t> zero_count_;
   
   //! @brief Set initial binary variables.
   //! @param seed The seed for initializing binary variables.
   void SetRandomConfiguration(const SeedType seed) {
      sample_.resize(system_size_);
      std::uniform_int_distribution<std::int8_t> dist(0, 1);
      RandType random_number_engine(seed);
      for (std::int32_t i = 0; i < system_size_; i++) {
         sample_[i] = dist(random_number_engine);
      }
   }
   
   void SetZeroCount() {
      zero_count_.resize(key_value_list_.size());
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         std::int32_t count = 0;
         for (const auto &index : key_value_list_[i].first) {
            if (sample_[index] == 0) {
               count++;
            }
         }
         zero_count_[i] = count;
      }
   }
   
   void SetEnergyDifference() {
      energy_difference_.resize(system_size_);
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         const ValueType value = key_value_list_[i].second;
         for (const auto &index: key_value_list_[i].first) {
            if (sample_[index] + zero_count_[i] == 1) {
               energy_difference_[index] += value*(-2*sample_[index] + 1);
            }
         }
      }
   }
   
   
};


} // namespace system
} // namespace openjij
