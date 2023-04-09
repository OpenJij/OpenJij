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

#include <cstdint>
#include <vector>
#include <random>
#include "./sa_system.hpp"

namespace openjij {
namespace system {

template<typename FloatType, typename RandType>
class SASystem<graph::IsingPolynomialModel<FloatType>, RandType> {
   
   //! @brief The model type, which must be IsingPolynomialModel.
   using ModelType = graph::IsingPolynomialModel<FloatType>;
   
   //! @brief The variable type, which here represents binary variables \f$ x_i\in \{-1, 1\} \f$
   using VariableType = typename ModelType::VariableType;
   
   //! @brief The type of seed in random number engine.
   using SeedType = typename RandType::result_type;
   
public:
   //! @brief The value type.
   using ValueType = typename ModelType::ValueType;
   
   //! @brief Constructor of SASystem for IsingPolynomialModel.
   //! @param model The IsingPolynomialModel.
   //! @param seed The seed for initializing binary variables.
   SASystem(const ModelType &model, const SeedType seed):
   system_size_(model.GetSystemSize()),
   key_value_list_(model.GetKeyValueList()),
   adjacency_list_(model.GetAdjacencyList()) {
      SetRandomConfiguration(seed);
      SetTermProd();
      SetBaseEnergyDifference();
   }
   
   void SetSample(const std::vector<VariableType> &sample) {
      if (static_cast<std::int32_t>(sample.size()) != system_size_) {
         throw std::runtime_error("The size of initial variables is not equal to the system size.");
      }
      for (std::int32_t i = 0; i < system_size_; ++i) {
         if (!(sample[i] == -1 || sample[i] == 1)) {
            throw std::runtime_error("The initial variables must be -1 or 1.");
         }
      }
      sample_ = sample;
      SetTermProd();
      SetBaseEnergyDifference();
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   void Flip(const std::int32_t index) {
      sample_[index] *= -1;
      for (const auto &index_key: adjacency_list_[index]) {
         const ValueType val = -2*key_value_list_[index_key].second*term_prod_[index_key];
         term_prod_[index_key] *= -1;
         for (const auto &v_index: key_value_list_[index_key].first) {
            if (v_index != index) {
               base_energy_difference_[v_index] += val*sample_[v_index];
            }
         }
      }
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
   const std::vector<ValueType> &GetBaseEnergyDifference() const {
      return base_energy_difference_;
   }
   
   //! @brief Get the energy difference when flipped.
   //! @param index The index of variables.
   //! @return The energy difference.
   ValueType GetEnergyDifference(const std::int32_t index) const {
      return -2*sample_[index]*base_energy_difference_[index];
   }
   
private:
   const std::int32_t system_size_;
   const std::vector<std::pair<std::vector<std::int32_t>, ValueType>> &key_value_list_;
   const std::vector<std::vector<std::size_t>> &adjacency_list_;
   
   std::vector<VariableType> sample_;
   std::vector<ValueType> base_energy_difference_;
   std::vector<short> term_prod_;
   
   //! @brief Set initial binary variables.
   //! @param seed The seed for initializing binary variables.
   void SetRandomConfiguration(const SeedType seed) {
      sample_.resize(system_size_);
      std::uniform_int_distribution<short> dist(0, 1);
      RandType random_number_engine(seed);
      for (std::int32_t i = 0; i < system_size_; i++) {
         sample_[i] = 2*dist(random_number_engine) - 1;
      }
   }
   
   void SetTermProd() {
      term_prod_.resize(key_value_list_.size());
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         short prod = 1;
         for (const auto &index: key_value_list_[i].first) {
            prod *= sample_[index];
         }
         term_prod_[i] = prod;
      }
   }
      
   void SetBaseEnergyDifference() {
      base_energy_difference_.clear();
      base_energy_difference_.resize(system_size_);
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         const ValueType value = key_value_list_[i].second;
         for (const auto &index: key_value_list_[i].first) {
            base_energy_difference_[index] += value*term_prod_[i]*sample_[index];
         }
      }
   }
   
   
};


} // namespace system
} // namespace openjij
