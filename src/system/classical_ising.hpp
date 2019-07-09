//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef OPENJIJ_SYSTEM_CLASSICAL_ISING_HPP__
#define OPENJIJ_SYSTEM_CLASSICAL_ISING_HPP__

#include <cassert>
#include <system/system.hpp>
#include <graph/all.hpp>
#include <type_traits>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace openjij {
    namespace system {

        /**
         * @brief naive ClassicalIsing structure (system for classical Ising model)
         *
         * @tparam GraphType type of graph
         * @tparam eigen_impl specify that Eigen implementation is enabled.
         */
        template<typename GraphType, bool eigen_impl=false>
            struct ClassicalIsing {
                static_assert(!eigen_impl, "Eigen implementation is not supported.");

                using system_type = classical_system;
                
                /**
                 * @brief Constructor to initialize spin and interaction
                 *
                 * @param spin
                 * @param interaction
                 */
                ClassicalIsing(const graph::Spins& init_spin, const GraphType& init_interaction)
                    : spin{init_spin}, interaction{init_interaction} {
                        assert(init_spin.size() == init_interaction.get_num_spins());
                    }

                graph::Spins spin;
                const GraphType interaction;
            };


        /**
         * @brief ClassicalIsing structure for Dense graph (Eigen-based)
         *
         * @tparam FloatType type of floating-point
         */
        template<typename FloatType>
            struct ClassicalIsing<graph::Dense<FloatType>, true>{
                using system_type = classical_system;

                using MatrixXx = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
                using VectorXx = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;

                /**
                 * @brief Constructor to initialize spin and interaction
                 *
                 * @param spin
                 * @param interaction
                 */
                ClassicalIsing(const graph::Spins& init_spin, const graph::Dense<FloatType>& init_interaction)
                : spin(init_interaction.get_num_spins()+1), interaction(init_interaction.get_num_spins()+1, init_interaction.get_num_spins()+1){
                    assert(init_spin.size() == init_interaction.get_num_spins());

                    //initialize spin
                    for(size_t i=0; i<init_spin.size(); i++){
                        spin(i) = init_spin[i];
                    }

                    //for local field
                    spin[init_spin.size()] = 1;

                    //initialize interaction
                    interaction.setZero();

                    for(size_t i=0; i<init_interaction.get_num_spins(); i++){
                        for(size_t j=i+1; j<init_interaction.get_num_spins(); j++){
                            interaction(i,j) = init_interaction.J(i,j);
                            interaction(j,i) = init_interaction.J(i,j);
                        }
                    }

                    //for local field
                    for(size_t i=0; i<init_interaction.get_num_spins(); i++){
                            interaction(i,init_interaction.get_num_spins()) = init_interaction.h(i);
                            interaction(init_interaction.get_num_spins(),i) = init_interaction.h(i);
                    }

                    //for local field
                    interaction(init_interaction.get_num_spins(),init_interaction.get_num_spins()) = 1;
                }

                VectorXx spin;
                MatrixXx interaction;
            };

        /**
         * @brief ClassicalIsing structure for Sparse graph (Eigen-based)
         *
         * @tparam FloatType type of floating-point
         */
        template<typename FloatType>
            struct ClassicalIsing<graph::Sparse<FloatType>, true>{
                using system_type = classical_system;

                using SparseMatrixXx = Eigen::SparseMatrix<FloatType>;
                using VectorXx = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;

                /**
                 * @brief Constructor to initialize spin and interaction
                 *
                 * @param spin
                 * @param interaction
                 */
                ClassicalIsing(const graph::Spins& init_spin, const graph::Sparse<FloatType>& init_interaction)
                : spin(init_interaction.get_num_spins()+1), interaction(init_interaction.get_num_spins()+1, init_interaction.get_num_spins()+1){
                    assert(init_spin.size() == init_interaction.get_num_spins());

                    //initialize spin
                    for(size_t i=0; i<init_spin.size(); i++){
                        spin(i) = init_spin[i];
                    }

                    //for local field
                    spin[init_spin.size()] = 1;

                    //initialize interaction
                    interaction.setZero();

                    //make triplet list
                    using T = std::vector<Eigen::Triplet<FloatType>>;
                    T t_list;

                    for(size_t ind=0; ind<init_interaction.get_num_spins(); ind++){
                        for(size_t adj_ind : init_interaction.adj_nodes(ind)){
                            if(ind != adj_ind){
                                t_list.emplace_back(ind, adj_ind, init_interaction.J(ind, adj_ind));
                            }
                            else{
                                t_list.emplace_back(ind, init_interaction.get_num_spins(), init_interaction.h(ind));
                                t_list.emplace_back(init_interaction.get_num_spins(), ind, init_interaction.h(ind));
                            }
                        }
                    }

                    t_list.emplace_back(init_interaction.get_num_spins(), init_interaction.get_num_spins(), 1);

                    interaction.setFromTriplets(t_list.begin(), t_list.end());
                }

                VectorXx spin;
                SparseMatrixXx interaction;
            };


        /**
         * @brief helper function for ClassicalIsing constructor
         *
         * @tparam eigen_impl
         * @tparam GraphType
         * @param init_spin initial spin
         * @param init_interaction initial interaction
         *
         * @return generated object
         */
        template<bool eigen_impl=false,typename GraphType>
            ClassicalIsing<GraphType, eigen_impl> make_classical_ising(const graph::Spins& init_spin, const GraphType& init_interaction){
                return ClassicalIsing<GraphType, eigen_impl>(init_spin, init_interaction);
            }



    } // namespace system
} // namespace openjij

#endif
