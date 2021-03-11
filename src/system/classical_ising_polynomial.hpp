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
   num_spins(init_interaction.get_num_spins()), spin(init_spins) {
      assert(init_spins.size() == num_spins);
      
      if (init_interaction.GetMaxVariable() >= num_spins) {
         printf("ssss\n");
         std::exit(0);
      }
      
      std::vector<std::vector<graph::Index>> temp_spins_in_interaction;
      std::vector<FloatType> temp_J;
      temp_spins_in_interaction.reserve(init_interaction.GetInteractions().size());
      for (const auto &it: init_interaction.GetInteractions()) {
         temp_spins_in_interaction.push_back(it.first);
         temp_J.push_back(it.second);
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
      
      std::sort(temp_spins_in_interaction.begin(), temp_spins_in_interaction.end(), comp);
            
      connected_interaction_index.resize(num_spins);
      for (auto i = 0; i < temp_spins_in_interaction.size(); ++i) {
         graph::Spin temp_spin_multiple = 1;
         for (const auto &index_spin: temp_spins_in_interaction[i]) {
            temp_spin_multiple *= spin[index_spin];
            connected_interaction_index[index_spin].push_back(J_term.size());
         }
         J_term.push_back(temp_J[i]*temp_spin_multiple);
      }
      
      row.push_back(0);
      for (graph::Index index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
         std::unordered_set<std::size_t> temp_set;
         std::vector<std::vector<FloatType*>> temp_val(num_spins);
         for (const auto &index_interaction: connected_interaction_index[index_spin_row]) {
            for (const auto &index_spin_col: temp_spins_in_interaction[index_interaction]) {
               if (index_spin_row != index_spin_col) {
                  temp_set.emplace(index_spin_col);
                  temp_val[index_spin_col].push_back(&J_term[index_interaction]);
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
         for (const auto &it: connected_interaction_index[i]) {
            temp_energy += J_term[it];
         }
         dE[i] = -2*temp_energy;
      }
   }
   
   graph::Spins spin;
   graph::Index num_spins;
   std::vector<std::vector<graph::Index>> connected_interaction_index;
   std::vector<FloatType>    dE;
   std::vector<FloatType>    J_term;
   std::vector<FloatType*>   val;
   std::vector<graph::Index> row;
   std::vector<graph::Index> col;
   
};

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins& init_spin, const GraphType& init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
