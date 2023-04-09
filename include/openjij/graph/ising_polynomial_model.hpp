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


namespace openjij {
namespace graph {


template<typename FloatType>
class IsingPolynomialModel {
   static_assert(std::is_floating_point<FloatType>::value,
                 "Template parameter FloatType must be floating point type");
   
public:
   //! @brief The value type.
   using ValueType = FloatType;
   
   //! @brief The index type.
   using IndexType = utility::IndexType;
   
   //! @brief The hash for IndexType.
   using IndexHash = utility::IndexHash;
   
   //! @brief The variable type, which here represents binary variables \f$ x_i\in \{0, 1\} \f$
   using VariableType = std::int8_t;
   
   
   IsingPolynomialModel(std::vector<std::vector<IndexType>> &key_list,
                        std::vector<ValueType> &value_list) {
      
      ValueType abs_max_interaction = -1;
      
      // Generate index list and store max interactions
      std::unordered_set<IndexType, IndexHash> index_set;
      
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         if (std::abs(value_list[i]) > std::numeric_limits<ValueType>::epsilon()) {
            std::sort(key_list[i].begin(), key_list[i].end());
            const std::int32_t key_size = static_cast<std::int32_t>(key_list[i].size()) - 1;
            for (std::int32_t j = key_size; j >= 1; --j) {
               if (key_list[i][j] == key_list[i][j - 1]) {
                  std::swap(key_list[i][j], key_list[i].back());
                  key_list[i].pop_back();
                  --j;
                  std::swap(key_list[i][j], key_list[i].back());
                  key_list[i].pop_back();
               }
            }
            index_set.insert(key_list[i].begin(), key_list[i].end());
            if (std::abs(value_list[i]) > abs_max_interaction) {
               abs_max_interaction = std::abs(value_list[i]);
            }
         }
      }
      
      index_list_ = std::vector<IndexType>(index_set.begin(), index_set.end());
      std::sort(index_list_.begin(), index_list_.end());
      
      system_size_ = static_cast<std::int32_t>(index_list_.size());
      
      // Generate index map (from index to integer)
      index_map_.reserve(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         index_map_[index_list_[i]] = i;
      }
      
      // Generate interactions with integer index
      std::unordered_map<std::vector<std::int32_t>, ValueType, utility::VectorHash> poly;
      poly.reserve(key_list.size());
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         if (std::abs(value_list[i]) > std::numeric_limits<ValueType>::epsilon()) {
            std::vector<std::int32_t> int_key(key_list[i].size());
            for (std::size_t j = 0; j < key_list[i].size(); ++j) {
               int_key[j] = index_map_.at(key_list[i][j]);
            }
            poly[int_key] += value_list[i];
            if (degree_ < int_key.size()) {
               degree_ = static_cast<std::int32_t>(int_key.size());
            }
         }
      }
      
      key_value_list_.reserve(poly.size());
      for (const auto &it: poly) {
         key_value_list_.push_back({it.first, it.second});
      }
      
      poly.clear();
      
      //Sort by keys.
      std::sort(key_value_list_.begin(), key_value_list_.end(), [](const auto &a, const auto &b) {
         return a.first < b.first;
      });
      
      adjacency_list_.resize(system_size_);
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         for (const auto &index: key_value_list_[i].first) {
            adjacency_list_[index].push_back(i);
         }
      }
      
      for (std::int32_t i = 0; i < system_size_; ++i) {
         adjacency_list_[i].shrink_to_fit();
         std::sort(adjacency_list_[i].begin(), adjacency_list_[i].end());
      }
      
      // Set relevant absolute minimum interaction
      // Apply threshold to avoid extremely minimum interaction derived from numerical errors.
      ValueType relevant_abs_min_interaction = abs_max_interaction*min_max_energy_difference_ratio_;
      estimated_min_energy_difference_ = std::numeric_limits<ValueType>::max();
      estimated_max_energy_difference_ = -1;
      
      for (std::int32_t i = 0; i < system_size_; ++i) {
         ValueType abs_row_sum_interaction = 0;
         for (const auto &interaction_index: adjacency_list_[i]) {
            if (std::abs(key_value_list_[interaction_index].second) >= relevant_abs_min_interaction) {
               abs_row_sum_interaction += std::abs(key_value_list_[interaction_index].second);
               if (2*std::abs(key_value_list_[interaction_index].second) < estimated_min_energy_difference_) {
                  estimated_min_energy_difference_ = 2*std::abs(key_value_list_[interaction_index].second);
               }
            }
         }
         if (2*abs_row_sum_interaction > estimated_max_energy_difference_) {
            estimated_max_energy_difference_ = 2*abs_row_sum_interaction;
         }
      }
      
      if (degree_ == 0) {
         estimated_min_energy_difference_ = 0;
         estimated_max_energy_difference_ = 0;
      }
      
   }
   
   //! @brief Get the degree of the polynomial interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      return degree_;
   }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return static_cast<std::int32_t>(index_list_.size());
   }
   
   //! @brief Get the index list of the polynomial interactions.
   //! @return The index list.
   const std::vector<IndexType> &GetIndexList() const {
      return index_list_;
   }
   
   //! @brief Get the mapping from the index to the integer.
   //! @return The index map.
   const std::unordered_map<IndexType, std::int32_t, IndexHash> &GetIndexMap() const {
      return index_map_;
   }
   
   //! @brief Get the integer key and value list as pair.
   //! @return The integer key and value list as pair.
   const std::vector<std::pair<std::vector<std::int32_t>, ValueType>> &GetKeyValueList() const {
      return key_value_list_;
   }
   
   //! @brief Get the adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   //! @return The adjacency list.
   const std::vector<std::vector<std::size_t>> &GetAdjacencyList() const {
      return adjacency_list_;
   }
   
   //! @brief Get estimated minimum energy difference.
   //! @return The estimated minimum energy difference.
   ValueType GetEstimatedMinEnergyDifference() const {
      return estimated_min_energy_difference_;
   }
   
   //! @brief Get estimated maximum energy difference.
   //! @return The estimated maximum energy difference.
   ValueType GetEstimatedMaxEnergyDifference() const {
      return estimated_max_energy_difference_;
   }
   
   //! @brief Calculate energy corresponding to the variable configuration.
   //! @param variables The variable configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const std::vector<VariableType> &variables) const {
      if (variables.size() != system_size_) {
         throw std::runtime_error("The size of variables is not equal to the system size");
      }
      ValueType val = 0.0;
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         VariableType prod = 1;
         for (const auto &index: key_value_list_[i].first) {
            prod *= variables[index];
         }
         val += key_value_list_[i].second*prod;
      }
      return val;
   }
   
   
private:
   //! @brief The degree of the interactions.
   std::int32_t degree_ = 0;
   
   //! @brief The system size.
   std::int32_t system_size_ = 0;
   
   //! @brief The index list of the interactions.
   std::vector<IndexType> index_list_;
      
   //! @brief The mapping from the index to the integer.
   std::unordered_map<IndexType, std::int32_t, IndexHash> index_map_;
   
   //! @brief The integer key and value list as pair.
   std::vector<std::pair<std::vector<std::int32_t>, ValueType>> key_value_list_;
   
   //! @brief The adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   std::vector<std::vector<std::size_t>> adjacency_list_;
   
   //! @brief The estimated minimum energy difference.
   ValueType estimated_min_energy_difference_ = 0;
   
   //! @brief The estimated maximum energy difference.
   ValueType estimated_max_energy_difference_ = 0;
   
   //! @brief The ratio of minimum and maximum energy difference set by 1e-08.
   //! \f[ {\rm ratio} = \frac{\Delat E_{{\rm min}}}{\Delat E_{{\rm max}}}\f]
   const ValueType min_max_energy_difference_ratio_ = 1e-08;
   
};

}
}
