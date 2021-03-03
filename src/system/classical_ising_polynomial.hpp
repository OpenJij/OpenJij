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
   num_spins(init_interaction.get_num_spins()), spin(init_spin), list_adj_nodes(init_interaction.list_adj_nodes()), list_interactions(init_interaction.list_interactions()) {
      
      for (const auto &it: list_interactions) {
         J.push_back(init_interaction.J(it));
      }
      adj.resize(num_spins);
      for (std::size_t index = 0; index < num_spins; ++index) {
         FloatType temp_dE = 0.0;
         for (const auto &it_adj: list_adj_nodes[index]) {
            graph::Spin temp_spin_multipl = 1;
            for (const auto &it_inter: list_interactions[it_adj]) {
               temp_spin_multipl *= init_spin[it_inter];
               if (index != it_inter && std::find(adj[index].begin(), adj[index].end(), it_inter) == adj[index].end()) {
                  adj[index].push_back(it_inter);
               }
            }
            temp_dE += -2*J[it_adj]*temp_spin_multipl;
         }
         dE.push_back(temp_dE);
      }
      
      
      
      for (std::size_t index = 0; index < num_spins; ++index) {
         printf("dE[%d]=%lf\n", index, dE[index]);
      }
      
      for (std::size_t index = 0; index < num_spins; ++index) {
         printf("adj[%d]=", index);
         for (std::size_t i = 0; i < adj[index].size(); i++) {
            printf("%d, ",adj[index][i]);
         }
         printf("\n");
      }
   }
   
   graph::Spins spin;
   std::vector<FloatType> dE;
   std::vector<FloatType> J;
   std::size_t num_spins;
   std::vector<std::vector<graph::Index>> list_adj_nodes;
   std::vector<std::vector<graph::Index>> list_interactions;
   std::vector<std::vector<graph::Index>> adj;
   
};

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins& init_spin, const GraphType& init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
