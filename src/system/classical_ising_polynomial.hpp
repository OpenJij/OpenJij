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
      CheckVariables();
      Interactions interaction = SetInteractions(init_interaction);
      std::vector<std::vector<graph::Index>> interacted_spins = GenerateInteractedSpins(interaction);
      SetJTerm(interaction, interacted_spins);
      SetUpdateMatrix(interacted_spins);
      SetdE();
   }
   
   void CheckVariables() {
      if (isIsing) {
         for (auto i = 0; i < spin.size(); ++i) {
            if (spin[i] != -1 && spin[i] != 1) {
               std::stringstream ss;
               ss << "The variable at " << i << " is " << spin[i] << ".\n";
               ss << "But the spin variable must be -1 or +1.\n";
               std::runtime_error(ss.str());
            }
         }
      }
      else {
         for (auto i = 0; i < spin.size(); ++i) {
            if (spin[i] != 0 && spin[i] != 1) {
               std::stringstream ss;
               ss << "The variable at " << i << " is " << spin[i] << ".\n";
               ss << "But the binary variable must be 0 or 1.\n";
               std::runtime_error(ss.str());
            }
         }
      }
   }
   
   void SetdE() {
      dE.resize(num_spins);
      if (isIsing) {
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            for (const auto &it: connected_J_term_index[i]) {
               temp_energy += J_term[it];
            }
            dE[i] = -2*temp_energy;
         }
      }
      else {
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            auto temp_spin = spin[i];
            for (const auto &it: connected_J_term_index[i]) {
               temp_energy += J_term[it]*sign(temp_spin)*ZeroOrOne(temp_spin, zero_count_binary[it]);
            }
            dE[i] = temp_energy;
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
   
   int sign(graph::Spin spin) {
      return (spin%2 == 0) ? 1 : -1;
   }
   int ZeroOrOne(graph::Spin spin, std::size_t zero_count) {
      return (zero_count == 1 - spin) ? 1 : 0;
   }
   int ZeroOrOne(graph::Spin spin1, graph::Spin spin2, std::size_t zero_count) {
      if (zero_count == 2 - spin1 - spin2) {
         return 1;
      }
      else {
         return 0;
      }
   }
   
   Interactions SetInteractions(const graph::Polynomial<FloatType> &init_interaction) {
      if (init_interaction.GetMaxVariable() >= num_spins) {
         return relabel_interactions(init_interaction.GetInteractions());
      }
      else {
         return init_interaction.GetInteractions();
      }
   }
   
   std::vector<std::vector<graph::Index>> GenerateInteractedSpins(const Interactions &interaction) {
      std::vector<std::vector<graph::Index>> interacted_spins;
      interacted_spins.reserve(interaction.size());
      for (const auto &it: interaction) {
         interacted_spins.push_back(it.first);
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
      std::sort(interacted_spins.begin(), interacted_spins.end(), comp);
      return interacted_spins;
   }
   
   void SetJTerm(Interactions &interaction, std::vector<std::vector<graph::Index>> &interacted_spins) {
      connected_J_term_index.resize(num_spins);
      J_term.clear();
      zero_count_binary.clear();
      if (isIsing) {
         for (auto i = 0; i < interacted_spins.size(); ++i) {
            graph::Spin temp_spin_multiple = 1;
            for (const auto &index_spin: interacted_spins[i]) {
               temp_spin_multiple *= spin[index_spin];
               connected_J_term_index[index_spin].push_back(J_term.size());
            }
            J_term.push_back(interaction[interacted_spins[i]]*temp_spin_multiple);
         }
      }
      else {
         for (auto i = 0; i < interacted_spins.size(); ++i) {
            std::size_t temp_zero_count = 0;
            uint32_t    temp_spin_multiple = 1;
            for (const auto &index_spin: interacted_spins[i]) {
               temp_spin_multiple *= spin[index_spin];
               connected_J_term_index[index_spin].push_back(J_term.size());
               if (spin[index_spin] == 0) {
                  temp_zero_count++;
               }
            }
            J_term.push_back(interaction[interacted_spins[i]]);
            zero_count_binary.push_back(temp_zero_count);
         }
      }
   }
   
   void SetUpdateMatrix(std::vector<std::vector<graph::Index>> &interacted_spins) {
      row.push_back(0);
      if (isIsing) {
         for (auto index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
            std::unordered_set<graph::Index>     temp_set;
            std::vector<std::vector<FloatType*>> temp_val(num_spins);
            for (const auto &index_spin_term: connected_J_term_index[index_spin_row]) {
               for (const auto &index_spin_col: interacted_spins[index_spin_term]) {
                  if (index_spin_row != index_spin_col) {
                     temp_set.emplace(index_spin_col);
                     temp_val[index_spin_col].push_back(&J_term[index_spin_term]);
                  }
               }
            }
            std::vector<std::size_t> temp_vec(temp_set.begin(), temp_set.end());
            std::sort(temp_vec.begin(), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (const auto &it2: temp_val[it]) {
                  col.push_back(it);
                  val_p_spin.push_back(it2);
               }
            }
            row.push_back(col.size());
         }
      }
      else {
         for (auto index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
            std::unordered_set<graph::Index>    temp_set;
            std::vector<std::vector<FloatType>> temp_val(num_spins);
            std::vector<std::vector<std::size_t*>> temp_zero(num_spins);
            for (const auto &index_spin_term: connected_J_term_index[index_spin_row]) {
               for (const auto &index_spin_col: interacted_spins[index_spin_term]) {
                  if (index_spin_row != index_spin_col) {
                     temp_set.emplace(index_spin_col);
                     temp_val[index_spin_col].push_back(J_term[index_spin_term]);
                     temp_zero[index_spin_col].push_back(&zero_count_binary[index_spin_term]);
                  }
               }
            }
            std::vector<std::size_t> temp_vec(temp_set.begin(), temp_set.end());
            std::sort(temp_vec.begin(), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (auto i = 0; i < temp_val[it].size(); ++i) {
                  col.push_back(it);
                  val_binary.push_back(temp_val[it][i]);
                  zero_count_p_binary.push_back(temp_zero[it][i]);
               }
            }
            row.push_back(col.size());
         }
      }
   }
   
   graph::Spins spin;
   graph::Index num_spins;
   std::vector<std::vector<std::size_t>> connected_J_term_index;
   std::vector<FloatType>    dE;
   std::vector<FloatType>    J_term;
   std::vector<std::size_t>  row;
   std::vector<graph::Index> col;
   std::vector<FloatType*>   val_p_spin;          //used only for spin   variables
   std::vector<FloatType>    val_binary;          //used only for binary variables
   std::vector<std::size_t*> zero_count_p_binary; //used only for binary variables
   std::vector<std::size_t>  zero_count_binary;   //used only for binary variables
   const bool isIsing;
   
};

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins& init_spin, const GraphType& init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */