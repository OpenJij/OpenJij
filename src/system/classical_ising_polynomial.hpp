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
      
      connected_interaction_index.resize(num_spins);
      connected_spins.resize(num_spins);
      spins_in_iteraction.resize(init_interaction.GetInteractions().size());
      if (init_interaction.GetMaxVariable() >= num_spins) {
         printf("ssss\n");
         std::exit(0);
      }
      else {
         for (const auto &it: init_interaction.GetInteractions()) {
            graph::Spin temp_spin_multiple = 1;
            for (const auto &index: it.first) {
               temp_spin_multiple *= spin[index];
               connected_interaction_index[index].push_back(J_term.size());
               spins_in_iteraction[J_term.size()].push_back(index);
            }
            J_term.push_back(it.second*temp_spin_multiple);
            J.push_back(it.second);
         }
         
         for (std::size_t index = 0; index < num_spins; ++index) {
            std::unordered_set<std::size_t> temp_set;
            for (const auto &index_interaction: connected_interaction_index[index]) {
               for (const auto &index_spin: spins_in_iteraction[index_interaction]) {
                  if (index != index_spin) {
                     temp_set.emplace(index_spin);
                  }
               }
            }
            std::vector<std::size_t> temp_vec(temp_set.begin(), temp_set.end());
            std::sort(temp_vec.begin(), temp_vec.end());
            connected_spins[index] = temp_vec;
         }
         
         
         row.push_back(0);
         for (std::size_t i = 0; i < num_spins; ++i) {
            for (std::size_t j = 0; j < num_spins; ++j) {
               if (i != j) {
                  std::vector<std::size_t> temp;
                  for (const auto &it: connected_interaction_index[i]) {
                     for (const auto &index_spin: spins_in_iteraction[it]) {
                        if (j == index_spin) {
                           temp.push_back(it);
                           break;
                        }
                     }
                  }
                  if (temp.size() > 0) {
                     col.push_back(j);
                     val.emplace_back();
                     for (const auto &it: temp) {
                        val[val.size() - 1].push_back(&J_term[it]);
                     }
                  }
               }
            }
            row.push_back(col.size());
         }
      }
      
      
      reset_dE();
      
      /*
      *val[3][0] = 1234567;
      for (std::size_t i = 0; i < num_spins; ++i) {
         for (std::size_t j = row[i]; j < row[i+1]; ++j) {
            printf("M[%ld][%ld]=", i , col[j]);
            for (const auto &it: val[j]) {
               printf("%lf, ", *it);
            }
            printf("\n");
         }
      }
      exit(1);
      */
      
      /*
      for (std::size_t index = 0; index < num_spins; ++index) {
         printf("Spin[%ld]=%d\n", index, spin[index]);
      }
      for (std::size_t index = 0; index < num_spins; ++index) {
         for (const auto &it: connected_interaction_index[index]) {
            printf("connected_interaction_index[%ld]=%ld\n", index, it);
         }
      }
      for (std::size_t index = 0; index < J_term.size(); ++index) {
         for (const auto &it: spins_in_iteraction[index]) {
            printf("spins_in_iteraction[%ld]=%ld\n", index, it);
         }
      }
      
      for (std::size_t index = 0; index < J_term.size(); ++index) {
         printf("J_term[%ld]=%lf\n", index, J_term[index]);
      }
      exit(1);
      */
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
   
   void reset_dE_by_J_term(graph::Spin index) {
      FloatType temp_energy = 0.0;
      for (const auto &it: connected_interaction_index[index]) {
         temp_energy += J_term[it];
      }
      dE[index] = -2*temp_energy;
   }
   
   void reset_dE_by_J(graph::Spin index) {
      FloatType temp_energy = 0.0;
      for (const auto &it: connected_interaction_index[index]) {
         graph::Spin temp_spin_multiple = 1;
         for (const auto &it_index: spins_in_iteraction[it]) {
            temp_spin_multiple *= spin[it_index];
         }
         temp_energy += J[it]*temp_spin_multiple;
      }
      dE[index] = -2*temp_energy;
   }
   
   FloatType CalculateEnergyTerm(std::vector<graph::Index> &index, FloatType &J, graph::Spins &spin) {
      graph::Spin temp_spin_multiple = 1;
      for (const auto &it: index) {
         temp_spin_multiple *= spin[it];
      }
      return J*temp_spin_multiple;
   }
   
   graph::Spins spin;
   graph::Index num_spins;
   

   std::vector<std::vector<std::size_t>> connected_interaction_index;
   std::vector<std::vector<std::size_t>> spins_in_iteraction;
   
   std::vector<FloatType> J;
   std::vector<std::vector<std::size_t>> connected_spins;
   
   std::vector<FloatType> dE;
   std::vector<FloatType> J_term;
   std::vector<std::vector<FloatType*>> val;
   std::vector<std::size_t> row;
   std::vector<std::size_t> col;
   
   
};

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins& init_spin, const GraphType& init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
