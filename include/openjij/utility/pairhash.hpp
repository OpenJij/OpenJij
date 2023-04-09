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

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "openjij/graph/graph.hpp"
#include "./index_type.hpp"

namespace openjij {
namespace utility {

/**
 * @brief hash class for std::pair
 */
struct PairHash {

  /**
   * @brief generate hash for std::pair
   *
   * @param p std::pair variable
   *
   * @return generated hash
   */
  template <class T1, class T2>
  inline size_t operator()(const std::pair<T1, T2> &p) const {
    size_t lhs = std::hash<T1>()(p.first);
    size_t rhs = std::hash<T2>()(p.second);
    return lhs ^ (rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2));
  }
};


//! @brief Hash struct of IndexType
struct IndexHash {
   template<class... Types>
   std::size_t operator()(const std::variant<Types...> &v) const {
      if (std::holds_alternative<std::int32_t>(v)) {
         return std::hash<std::int32_t>()(std::get<std::int32_t>(v));
      }
      else if (std::holds_alternative<std::string>(v)) {
         return std::hash<std::string>()(std::get<std::string>(v));
      }
      else if (std::holds_alternative<AnyTupleType>(v)) {
         const auto &variant_vec = std::get<AnyTupleType>(v);
         std::size_t hash = variant_vec.size();
         for (const auto &i : variant_vec) {
            if (std::holds_alternative<std::int32_t>(i)) {
               hash ^= std::hash<std::int32_t>()(std::get<std::int32_t>(i)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            else if (std::holds_alternative<std::string>(i)) {
               hash ^= std::hash<std::string>()(std::get<std::string>(i)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            else {
               throw std::runtime_error("Invalid template parameters");
            }
         }
         return hash;
      }
      else {
         throw std::runtime_error("Invalid template parameters");
      }
   }
};

//! @brief Hash struct of std::vector<AnyIndexType>.
struct IndexVectorHash {
   std::size_t operator()(const std::vector<IndexType> &v) const {
      std::size_t hash = v.size();
      for (const auto &i : v) {
         hash ^= IndexHash()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
   }
};

//! @brief Hash struct of std::vector<T>.
struct VectorHash {
   template<class T>
   std::size_t operator()(const std::vector<T> &v) const {
      std::size_t hash = v.size();
      for (const auto &i : v) {
         hash ^= std::hash<T>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
   }
};

//! @brief Hash struct of std::pair<IndexType>.
struct IndexPairHash {
   std::size_t operator()(const std::pair<IndexType, IndexType> &p) const {
      std::size_t lhs = IndexHash()(p.first);
      std::size_t rhs = IndexHash()(p.second);
      return lhs^(rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2));
   }
};


} // namespace utility
} // namespace openjij
