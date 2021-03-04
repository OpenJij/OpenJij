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
   
   explicit Polynomial(std::size_t num_spins): Graph(num_spins) {}
   
   Polynomial(const Interactions &polynomial, const std::size_t num_spins): Graph(num_spins) {
      for (const auto it: polynomial) {
         std::unordered_set<Index> index_set(it.first.begin(), it.first.end());
         std::vector<Index> index(index_set.begin(), index_set.end());
         std::sort(index.begin(), index.end());
         UpdateMaxVariable(index);
         J_[index] = it.second;
      }
   }
   
   Polynomial(const json &j) {
      Polynomial(json_parse_polynomial<FloatType>(j));
   }
   
   
   Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm): Graph(bpm.length()) {
      for (auto &it: bpm.get_polynomial()) {
         std::sort(it.first.begin(), it.first.end());
         UpdateMaxVariable(it.first);
         J_[it.first] += it.second;
      }
   }
      
   FloatType &J(const std::unordered_set<Index> &index_set) {
      if (index_set.size() > Graph::size()) {
         std::cout << "Too small system size" << std::endl;
         std::cout << "The degree of the input polynomial interaction is " << index_set.size() << std::endl;
         std::cout << "But the number of system size is" << Graph::size() << std::endl;
         assert(index_set.size() <= Graph::size());
      }
      std::vector<Index> index(index_set.begin(), index_set.end());
      std::sort(index.begin(), index.end());
      UpdateMaxVariable(index);
      return J_[index];
   }
   
   template<typename... Args>
   FloatType &J(Args... args) {
      std::unordered_set<Index> index_set{(Index)args...};
      return J(index_set);
   }
   
   const FloatType &J(const std::unordered_set<Index> &index_set) const {
      std::vector<Index> index(index_set.begin(), index_set.end());
      std::sort(index.begin(), index.end());
      return J_.at(index);
   }
   
   template<typename... Args>
   const FloatType &J(Args... args) const {
      std::unordered_set<Index> index_set{(Index)args...};
      return J(index_set);
   }
   
   const Interactions &GetInteractions() const {
      return J_;
   }
   
   const Index &GetMaxVariable() const {
      return max_variable_;
   }
   
private:
   Interactions J_;
   Index max_variable_ = 0;
   
   void UpdateMaxVariable(const std::vector<Index> &index) {
      if (max_variable_ < index[index.size() - 1]) {
         max_variable_ = index[index.size() - 1];
      }
   }

   
};
}
}


#endif /* polynomial_hpp */
