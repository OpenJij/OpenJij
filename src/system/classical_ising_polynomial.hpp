//
//  classical_ising_polynomial.hpp
//  OpenJijXcode
//
//  Created by 鈴木浩平 on 2021/03/02.
//

#ifndef classical_ising_polynomial_hpp
#define classical_ising_polynomial_hpp

#include <vector>
#include <graph/all.hpp>

namespace openjij {
namespace system {

template<typename GraphType>
    struct ClassicalIsingPolynomial;

template<typename FloatType>
struct ClassicalIsingPolynomial<graph::Polynomial<FloatType>> {
   using Interactions = std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>;
   using system_type  = classical_system;
   
   ClassicalIsingPolynomial(const graph::Spins init_spin, const graph::Polynomial<FloatType> &init_interaction):
   num_spins(init_interaction.get_num_spins()), spin(init_spin) {
      assert(init_spin.size() == num_spins);
      
      list_interactions.resize(init_interaction.GetInteractions().size());
      list_adjacency.resize(num_spins);
      connected_spins.resize(num_spins);
      
      if (init_interaction.GetMaxVariable() >= num_spins) {
         printf("ssss\n");
         std::exit(0);
      }
      else {
         std::size_t index_interaction = 0;
         for (const auto &it: init_interaction.GetInteractions()) {
            J.push_back(it.second);
            for (const auto &it_index: it.first) {
               list_interactions[index_interaction].push_back(it_index);
               list_adjacency[it_index].push_back(index_interaction);
            }
            index_interaction++;
         }
      }

      std::vector<std::unordered_set<graph::Index>> temp;
      temp.resize(num_spins);
      for (std::size_t index = 0; index < num_spins; ++index) {
         for (const auto &it_adj: list_adjacency[index]) {
            for (const auto &it_inter: list_interactions[it_adj]) {
               if (it_inter != index) {
                  temp[index].emplace(it_inter);
               }
            }
         }
      }
      for (auto i = 0; i < num_spins; ++i) {
         connected_spins[i] = std::vector<graph::Index>(temp[i].begin(), temp[i].end());
         std::sort(connected_spins[i].begin(), connected_spins[i].end());
      }
      reset_dE();
      ResetEnergyTerm();
   }

   void reset_dE() {
      dE.clear();
      for (std::size_t index = 0; index < num_spins; ++index) {
         FloatType temp_dE = 0.0;
         for (const auto &it_adj: list_adjacency[index]) {
            graph::Spin temp_spin_multipl = 1;
            for (const auto &it_inter: list_interactions[it_adj]) {
               temp_spin_multipl *= spin[it_inter];
            }
            temp_dE += -2*J[it_adj]*temp_spin_multipl;
            //printf("%d(%d)-->%lf\n", index, it_adj, -2*J[it_adj]*temp_spin_multipl);
         }
         dE.push_back(temp_dE);
      }
   }
   
   void ResetEnergyTerm() {
      energy_term.clear();
      for (std::size_t index_interaction = 0; index_interaction < J.size(); ++index_interaction) {
         graph::Spin temp_spin_multipl = 1;
         for (const auto &it_inter: list_interactions[index_interaction]) {
            temp_spin_multipl *= spin[it_inter];
         }
         energy_term.push_back(temp_spin_multipl*J[index_interaction]);
      }
   }
   
   graph::Spins spin;
   std::vector<FloatType> dE;
   std::vector<FloatType> energy_term;
   std::vector<FloatType> J;
   std::size_t num_spins;
   std::vector<std::vector<graph::Index>> list_adjacency;
   std::vector<std::vector<graph::Index>> list_interactions;
   std::vector<std::vector<graph::Index>> connected_spins;
   
};

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins& init_spin, const GraphType& init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
