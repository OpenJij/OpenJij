//
//  vectorhash.h
//  OpenJijXcode
//
//  Created by 鈴木浩平 on 2021/03/01.
//

#ifndef vectorhash_hpp
#define vectorhash_hpp

#include <cstddef>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <graph/graph.hpp>


namespace openjij {
namespace utility {

//! @brief Hash class for std::vector
struct VectorHash {
   
   //! @brief Generate hash for std::vector
   //! @param v std::vector variable
   //! @return generated hash
   template <class T>
   std::size_t operator() (const std::vector<T> &v) const {
      std::size_t hash = v.size();
      for (auto &i: v) {
         hash ^= std::hash<T>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
   }
};

}  // namespace utility
}  // namespace openjij

#endif /* vectorhash_hpp */
