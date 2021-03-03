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
   
   Polynomial(const Interactions &polynomial, const std::size_t num_spins): Graph(num_spins) {
      list_adj_nodes_.resize(num_spins);
      for (const auto &it: polynomial) {
         J_[it.first] = it.second;
         list_interactions_.push_back(it.first);
         for (const auto &it_index: it.first) {
            list_adj_nodes_[it_index].push_back(list_interactions_.size() - 1);
         }
      }
      
   }
   
   Polynomial(const cimod::Polynomial<Index, FloatType> &polynomial, std::size_t num_spins): Graph(num_spins) {
      list_adj_nodes_.resize(num_spins);
      for (const auto &it: polynomial) {
         J_[it.first] = it.second;
         list_interactions_.push_back(it.first);
         for (const auto &it_index: it.first) {
            list_adj_nodes_[it_index].push_back(list_interactions_.size() - 1);
         }
      }
   }
   
   Polynomial(const json &j): Graph(static_cast<std::size_t>(j["num_variables"])) {
      auto bpm = json_parse_polynomial<FloatType>(j);
      list_adj_nodes_.resize(bpm.length());
      for (const auto &it: bpm.get_polynomial()) {
         J_[it.first] = it.second;
         list_interactions_.push_back(it.first);
         for (const auto &it_index: it.first) {
            list_adj_nodes_[it_index].push_back(list_interactions_.size() - 1);
         }
      }
   }
   
   
   explicit Polynomial(std::size_t num_spins): Graph(num_spins) {}
   
   Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm): Graph(bpm.length()) {
      list_adj_nodes_.resize(bpm.length());
      for (const auto &it: bpm.get_polynomial()) {
         J_[it.first] = it.second;
         list_interactions_.push_back(it.first);
         for (const auto &it_index: it.first) {
            list_adj_nodes_[it_index].push_back(list_interactions_.size() - 1);
         }
      }
   }
      
   FloatType &J(const std::vector<Index> &index) {
      if (std::find(list_interactions_.begin(), list_interactions_.end(), index) == list_interactions_.end()) {
         list_interactions_.push_back(index);
         for (const auto &it: index) {
            if (list_adj_nodes_.size() <= it) {
               list_adj_nodes_.resize(it + 1);
            }
            list_adj_nodes_[it].push_back(list_interactions_.size() - 1);
         }
      }
      return J_[index];
   }
   
   template<typename... Args>
   FloatType &J(Args... args) {
      std::vector<Index> index{(Index)args...};
      
      if (std::find(list_interactions_.begin(), list_interactions_.end(), index) == list_interactions_.end()) {
        
         list_interactions_.push_back(index);
        
         for (const auto &it: index) {
            if (list_adj_nodes_.size() <= it) {
               list_adj_nodes_.resize(it + 1);
            }
            list_adj_nodes_[it].push_back(list_interactions_.size() - 1);
         }
      }
      return J_[index];
   }
   
   const FloatType &J(const std::vector<Index> &index) const {
      return J_.at(index);
   }
   
   const Interactions &J() const {
      return J_;
   }
   
   const std::vector<std::vector<Index>> &list_adj_nodes() const {
      return list_adj_nodes_;
   }
   
   const std::vector<std::vector<Index>> &list_interactions() const {
      return list_interactions_;
   }
   
private:
   Interactions J_;
   std::vector<std::vector<Index>> list_adj_nodes_;
   std::vector<std::vector<Index>> list_interactions_;
   
};
}
}


#endif /* polynomial_hpp */
