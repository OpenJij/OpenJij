//    Copyright 2021 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef UTILITIES_HPP__
#define UTILITIES_HPP__

#include "vartypes.hpp"

#include <unordered_map>

namespace cimod
{
/**
 * @brief Insert or assign a element of unordered_map (for C++14 or C++11)
 *
 * @tparam C_key
 * @tparam C_value
 * @param um
 * @param key
 * @param val
 */

template <class C_key, class C_value, class Hash>
void insert_or_assign
(
    std::unordered_map<C_key, C_value, Hash> &um,
    const C_key &key,
    const C_value &val
)
{
    // insert
    if(um.count(key)==0)
    {
        um.insert({{key, val}});
    }
    // assign
    else
    {
        um[key] = val;
    }
}

//! @brief Format the input key: for example, {2,1,1}-->{1,2} for BINARY variable and {2,1,1}-->{2} for SPIN variable.
//! @tparam IndexType Used to represent the indices of variables
//! @param key This may be formatted.
//! @param vartype The model's type. cimod::Vartype::SPIN or cimod::Vartype::BINARY.
template <typename IndexType>
void FormatPolynomialKey(std::vector<IndexType> *key, const Vartype &vartype) {
   if ((*key).size() <= 1) {
      return;
   }
   else if ((*key).size() == 2) {
      if ((*key)[0] == (*key)[1]) {
         if (vartype == Vartype::SPIN) {
            (*key).clear();
            return;
         }
         else if (vartype == Vartype::BINARY) {
            (*key).pop_back();
            return;
         }
         else {
            throw std::runtime_error("Unknown vartype detected");
         }
      }
      else if ((*key)[0] < (*key)[1]) {
         return;
      }
      else {
         std::swap((*key)[0], (*key)[1]);
         return;
      }
   }
   else {
      std::sort((*key).begin(), (*key).end());
      if (vartype == Vartype::SPIN) {
         for (int64_t i = static_cast<int64_t>((*key).size()) - 1; i > 0; --i) {
            if ((*key)[i] == (*key)[i - 1]) {
               std::swap((*key)[i], (*key).back());
               (*key).pop_back();
               --i;
               std::swap((*key)[i], (*key).back());
               (*key).pop_back();
            }
         }
         return;
      }
      else if (vartype == Vartype::BINARY) {
         (*key).erase(std::unique((*key).begin(), (*key).end()), (*key).end());
         return;
      }
      else {
         throw std::runtime_error("Unknown vartype detected");
      }
   }
}

//! @brief Convert vartype from string to cimod::Vartype
//! @param vartype_str "SPIN" or "BINARY"
//! @return cimod::Vartype::SPIN or cimod::Vartype::BINARY
cimod::Vartype ToCimodVartype(const std::string &vartype_str) {
   if      (vartype_str == "SPIN"  ) { return Vartype::SPIN;   }
   else if (vartype_str == "BINARY") { return Vartype::BINARY; }
   else                              { throw std::runtime_error("Unknown vartype detected"); }
}

//! @brief Convert vartype from string to cimod::Vartype.
//! @tparam IntegerType The type of spin/binary configuration
//! @param configurations The spin/binary configuration
//! @param vartype The model's type. cimod::Vartype::SPIN or cimod::Vartype::BINARY.
template <typename IntegerType>
void CheckVariables(const std::vector<IntegerType> &configurations, const Vartype &vartype) {
   if (vartype == Vartype::SPIN) {
      for (const auto &v: configurations) {
         if (!(v == -1 || v == +1)) {
            throw std::runtime_error("The initial variables must be -1 or +1");
         }
      }
   }
   else if (vartype == Vartype::BINARY) {
      for (const auto &v: configurations) {
         if (!(v == 0 || v == 1)) {
            throw std::runtime_error("The initial variables must be 0 or 1");
         }
      }
   }
   else {
      throw std::runtime_error("Unknown vartype detected");
   }
}

//! @brief Check if the input vartype is not Vartype::NONE
//! @param vartype The model's type. cimod::Vartype::SPIN or cimod::Vartype::BINARY.
void CheckVartypeNotNONE(const Vartype &vartype) {
   if (vartype == cimod::Vartype::NONE) {
      throw std::runtime_error("Unknow vartype detected");
   }
}

}

#endif
