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
   
   Polynomial(const Interactions J, const std::size_t num_spins): Graph(num_spins), J_(J) {};
   Polynomial(const cimod::Polynomial<Index, FloatType> &polynomial, std::size_t num_spins): Graph(num_spins) {
      for (const auto &it: polynomial) {
         J_[it.first] = it.second;
      }
   }
   
   Polynomial(const json &j): Graph(static_cast<std::size_t>(j["num_variables"])) {
      auto bpm = json_parse_polynomial<FloatType>(j);
      for (const auto &it: bpm.get_polynomial()) {
         J_[it.first] = it.second;
      }
   }
   
   
   explicit Polynomial(std::size_t num_spins): Graph(num_spins) {}
   
   Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm): Graph(bpm.length()) {
      for (const auto &it: bpm.get_polynomial()) {
         J_[it.first] = it.second;
      }
   }
      
   FloatType &J(const std::vector<Index> &index) {
      return J_[index];
   }
   
   template<typename... Args>
   FloatType &J(Args... args) {
      std::vector<Index> index{(std::size_t)args...};
      return J_[index];
   }
   
   const FloatType &J(const std::vector<Index> &index) const {
      return J_.at(index);
   }
   
   const Interactions &J() const {
      return J_;
   }
   
private:
   Interactions J_;
   std::vector<std::vector<Nodes>> list_adj_nodes_;
   
};
}
}


#endif /* polynomial_hpp */
