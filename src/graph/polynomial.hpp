//
//  polynomial.hpp
//  OpenJijXcode
//
//  Created by 鈴木浩平 on 2021/03/01.
//

#ifndef polynomial_hpp
#define polynomial_hpp

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <unordered_map>

#include <graph/json/parse.hpp>
#include <graph/graph.hpp>
#include <utility/vectorhash.hpp>

namespace openjij {
namespace graph {

template<typename FloatType>
class Polynomial: public Graph {
   static_assert(std::is_floating_point<FloatType>::value, "FloatType must be floating-point type.");
   
public:
   using Interactions = std::unordered_map<std::vector<Index>, FloatType, utility::VectorHash>;
   using value_type   = FloatType;
   
   Polynomial(const json &j): Graph(j["num_variables"]) {
      auto bpm = json_parse_polynomial<FloatType>(j);
      for (const auto &it: bpm.get_polynomial()) {
         J_[it.first] = it.second;
      }
   };
   Polynomial(const Interactions J, std::size_t num_spins): Graph(num_spins), J_(J) {};
   
   const FloatType &J(std::vector<Index> &index) const {
      return J_.at(index);
   }
   
private:
   Interactions J_;
   
};
}
}


#endif /* polynomial_hpp */
