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

#ifndef OPENJIJ_SYSTEM_TRANSVERSE_ISING_HPP__
#define OPENJIJ_SYSTEM_TRANSVERSE_ISING_HPP__

#include <cassert>
#include <system/system.hpp>
#include <graph/all.hpp>
#include <utility/eigen.hpp>
#include <vector>

namespace openjij {
    namespace system {

        /**
         * @brief trotterized spin (std::vector<Spins>)
         * trotter_spins[i][j] -> jth spin in ith trotter slice.
         */
        using TrotterSpins = std::vector<graph::Spins>;

        /**
         * @brief naive TransverseIsing structure with discrete-time trotter spins (no Eigen implementation)
         *
         * @tparam GraphType
         * @tparam eigen_impl specify that Eigen implementation is enabled.
         */
        template<typename GraphType, bool eigen_impl=false>
            struct TransverseIsing {
                using system_type = transverse_field_system;
                using FloatType = typename GraphType::value_type;

                /**
                 * @brief TransverseIsing Constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 */
                TransverseIsing(const TrotterSpins& init_trotter_spins, const GraphType& init_interaction, FloatType gamma)
                : trotter_spins(init_trotter_spins), interaction(init_interaction), gamma(gamma){
                    assert(trotter_spins.size() >= 2);
                }

                /**
                 * @brief TransverseIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                TransverseIsing(const graph::Spins& classical_spins, const GraphType& init_interaction, FloatType gamma, size_t num_trotter_slices)
                : trotter_spins(num_trotter_slices), interaction(init_interaction), gamma(gamma){
                    //initialize trotter_spins with classical_spins
                    assert(trotter_spins.size() >= 2);
                    for(auto& spins : trotter_spins){
                        spins = classical_spins;
                    }
                }

                /**
                 * @brief trotterlized spins
                 */
                TrotterSpins trotter_spins;

                /**
                 * @brief interaction 
                 */
                const GraphType interaction;

                /**
                 * @brief coefficient of transverse field term
                 */
                FloatType gamma;
            };

        //TODO: unify Dense and Sparse Eigen-implemented TransverselIsing struct

        /**
         * @brief naive Dense TransverseIsing structure with discrete-time trotter spins (with Eigen implementation)
         *
         * @tparam FloatTypeType
         */
        template<typename FloatType>
            struct TransverseIsing<graph::Dense<FloatType>, true> {
                using system_type = transverse_field_system;

                //matrix (row major)
                using MatrixXx = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                //trotter matrix (col major)
                using TrotterMatrix = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

                /**
                 * @brief TransverseIsing Constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 */
                TransverseIsing(const TrotterSpins& init_trotter_spins, const graph::Dense<FloatType>& init_interaction, FloatType gamma)
                : num_classical_spins(init_trotter_spins[0].size()), gamma(gamma){
                    assert(init_trotter_spins.size() >= 2);

                    //init trotter_spins
                    trotter_spins = utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(init_trotter_spins);

                    //init interactions
                    interaction = utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction);
                }

                /**
                 * @brief TransverseIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                TransverseIsing(const graph::Spins& init_classical_spins, const graph::Dense<FloatType>& init_interaction, FloatType gamma, size_t num_trotter_slices)
                : num_classical_spins(init_classical_spins.size()), gamma(gamma){
                    //initialize trotter_spins with classical_spins

                    assert(init_classical_spins.size() >= 2);

                    TrotterSpins init_trotter_spins;

                    for(auto& spins : init_trotter_spins){
                        spins = init_classical_spins;
                    }

                    //init trotter_spins
                    trotter_spins = utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(init_trotter_spins);

                    //init interactions
                    interaction = utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction);
                }

                /**
                 * @brief trotterlized spins
                 */
                TrotterMatrix trotter_spins;

                /**
                 * @brief interaction 
                 */
                MatrixXx interaction;

                /**
                 * @brief number of real classical spins (dummy spin excluded)
                 */
                std::size_t num_classical_spins; //trotter_spins[0].size()-1

                /**
                 * @brief coefficient of transverse field term
                 */
                FloatType gamma;
            };

        /**
         * @brief naive Sparse TransverseIsing structure with discrete-time trotter spins (with Eigen implementation)
         *
         * @tparam FloatTypeType
         */
        template<typename FloatType>
            struct TransverseIsing<graph::Sparse<FloatType>, true> {
                using system_type = transverse_field_system;

                //matrix (row major)
                using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
                //trotter matrix (col major)
                using TrotterMatrix = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

                /**
                 * @brief TransverseIsing Constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 */
                TransverseIsing(const TrotterSpins& init_trotter_spins, const graph::Sparse<FloatType>& init_interaction, FloatType gamma)
                : num_classical_spins(init_trotter_spins[0].size()), gamma(gamma){
                    assert(init_trotter_spins.size() >= 2);

                    //init trotter_spins
                    trotter_spins = utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(init_trotter_spins);

                    //init interactions
                    interaction = utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction);
                }

                /**
                 * @brief TransverseIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                TransverseIsing(const graph::Spins& init_classical_spins, const graph::Sparse<FloatType>& init_interaction, FloatType gamma, size_t num_trotter_slices)
                : num_classical_spins(init_classical_spins.size()), gamma(gamma){
                    //initialize trotter_spins with classical_spins

                    assert(init_classical_spins.size() >= 2);

                    TrotterSpins init_trotter_spins;

                    for(auto& spins : init_trotter_spins){
                        spins = init_classical_spins;
                    }

                    //init trotter_spins
                    trotter_spins = utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(init_trotter_spins);

                    //init interactions
                    interaction = utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction);
                }

                /**
                 * @brief trotterlized spins
                 */
                TrotterMatrix trotter_spins;

                /**
                 * @brief interaction 
                 */
                SparseMatrixXx interaction;

                /**
                 * @brief number of real classical spins (dummy spin excluded)
                 */
                std::size_t num_classical_spins; //trotter_spins[0].size()-1

                /**
                 * @brief coefficient of transverse field term
                 */
                FloatType gamma;
            };

        /**
         * @brief helper function for TransverseIsing constructor
         *
         * @tparam eigen_impl
         * @tparam GraphType
         * @param init_trotter_spins
         * @param init_interaction
         * @param gamma
         *
         * @return generated object
         */
        template<bool eigen_impl=false,typename GraphType>
            TransverseIsing<GraphType, eigen_impl> make_transverse_ising(const TrotterSpins& init_trotter_spins, const GraphType& init_interaction, double gamma){
                return TransverseIsing<GraphType, eigen_impl>(init_trotter_spins, init_interaction, static_cast<typename GraphType::value_type>(gamma));
            }

        /**
         * @brief helper function for TransverseIsing constructor
         *
         * @tparam eigen_impl
         * @tparam GraphType
         * @param classical_spins
         * @param init_interaction
         * @param gamma
         * @param num_trotter_slices
         *
         * @return generated object
         */
        template<bool eigen_impl=false,typename GraphType>
            TransverseIsing<GraphType, eigen_impl> make_transverse_ising(const graph::Spins& classical_spins, const GraphType& init_interaction, double gamma, std::size_t num_trotter_slices){
                return TransverseIsing<GraphType, eigen_impl>(classical_spins, init_interaction, static_cast<typename GraphType::value_type>(gamma), num_trotter_slices);
            }
    } // namespace system
} // namespace openjij

#endif
