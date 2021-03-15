//
//  classical_ising_polynomial.hpp
//  OpenJijXcode
//
//  Created by 鈴木浩平 on 2021/03/02.
//

#ifndef classical_ising_polynomial_hpp
#define classical_ising_polynomial_hpp

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <graph/all.hpp>

namespace openjij {
namespace system {

template<typename GraphType>
struct ClassicalIsingPolynomial;

template<typename FloatType>
struct ClassicalIsingPolynomial<graph::Polynomial<FloatType>> {
   using Interactions = std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>;
   using system_type  = classical_system;
   
   ClassicalIsingPolynomial(const graph::Spins init_spins, const graph::Polynomial<FloatType> &init_interaction):
   num_spins(init_interaction.get_num_spins()), spin(init_spins), isIsing(init_interaction.isIsing()) {
      assert(init_spins.size() == num_spins);
      
      Interactions J;
      
      if (init_interaction.GetMaxVariable() >= num_spins) {
         J = relabel_interactions(init_interaction.GetInteractions());
      }
      else {
         J = init_interaction.GetInteractions();
      }
      
      std::vector<std::vector<graph::Index>> temp_interacted_spins;
      temp_interacted_spins.reserve(J.size());
      for (const auto &it: J) {
         temp_interacted_spins.push_back(it.first);
      }
      
      auto comp = [](const auto &lhs, const auto &rhs) {
         if (lhs.size() != rhs.size()) {
            return lhs.size() < rhs.size();
         }
         else {
            if (lhs[0] != rhs[0]) {
               return lhs[0] < rhs[0];
            }
            else {
               return lhs[1] < rhs[1];
            }
         }
      };
      
      std::sort(temp_interacted_spins.begin(), temp_interacted_spins.end(), comp);

      connected_J_term_index.resize(num_spins);
      for (auto i = 0; i < temp_interacted_spins.size(); ++i) {
         graph::Spin temp_spin_multiple = 1;
         for (const auto &index_spin: temp_interacted_spins[i]) {
            temp_spin_multiple *= spin[index_spin];
            connected_J_term_index[index_spin].push_back(J_term.size());
         }
         J_term.push_back(J[temp_interacted_spins[i]]*temp_spin_multiple);
      }
      
      row.push_back(0);
      for (auto index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
         std::unordered_set<graph::Index>     temp_set;
         std::vector<std::vector<FloatType*>> temp_val(num_spins);
         for (const auto &index_J_term: connected_J_term_index[index_spin_row]) {
            for (const auto &index_spin_col: temp_interacted_spins[index_J_term]) {
               if (index_spin_row != index_spin_col) {
                  temp_set.emplace(index_spin_col);
                  temp_val[index_spin_col].push_back(&J_term[index_J_term]);
               }
            }
         }
         std::vector<std::size_t> temp_vec(temp_set.begin(), temp_set.end());
         std::sort(temp_vec.begin(), temp_vec.end());
         for (const auto &it: temp_vec) {
            for (const auto &it2: temp_val[it]) {
               col.push_back(it);
               val.push_back(it2);
            }
         }
         row.push_back(col.size());
      }
      reset_dE();
   }
   
   void reset_dE() {
      dE.resize(num_spins);
      for (graph::Index i = 0; i < num_spins; ++i) {
         FloatType temp_energy = 0.0;
         for (const auto &it: connected_J_term_index[i]) {
            temp_energy += J_term[it];
         }
         if (isIsing) {
            dE[i] = -2*temp_energy;
         }
         else {
            if (spin[i] == 0) {
               dE[i] = temp_energy;
            }
            else {
               dE[i] = -temp_energy;
            }
         }
      }
   }
   
   Interactions relabel_interactions(const Interactions &input_interactions) {
      std::unordered_set<graph::Index> variable_set;
      for (const auto &it: input_interactions) {
         for (const auto &index: it.first) {
            variable_set.emplace(index);
         }
      }
      std::vector<graph::Index> variables(variable_set.begin(), variable_set.end());
      std::sort(variables.begin(), variables.end());
      std::unordered_map<graph::Index, graph::Index> relabeld_variables;
      for (auto i = 0; i < variables.size(); ++i) {
         relabeld_variables[variables[i]] = i;
      }
      
      Interactions relabeled_interactions;
      
      for (const auto &it: input_interactions) {
         std::vector<graph::Index> temp_vec(it.first.size());
         for (auto i = 0; i < it.first.size(); ++i) {
            temp_vec[i] = relabeld_variables[it.first[i]];
         }
         relabeled_interactions[temp_vec] = it.second;
      }
      return relabeled_interactions;
   }
   
   
   graph::Spins spin;
   graph::Index num_spins;
   std::vector<std::vector<graph::Index>> connected_J_term_index;
   std::vector<FloatType>     dE;
   std::vector<FloatType>      J;
   std::vector<FloatType> J_term;
   std::vector<FloatType*>   val;
   std::vector<graph::Index> row;
   std::vector<graph::Index> col;
   const bool isIsing;
   
};

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins& init_spin, const GraphType& init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
