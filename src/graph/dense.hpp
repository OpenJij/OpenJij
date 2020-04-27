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

#ifndef OPENJIJ_GRAPH_DENSE_HPP__
#define OPENJIJ_GRAPH_DENSE_HPP__

#include <map>
#include <vector>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <exception>

#include <utility/disable_eigen_warning.hpp>
#include <Eigen/Dense>

#include <graph/json/parse.hpp>
#include <graph/graph.hpp>

namespace openjij {
    namespace graph {

        /**
         * @brief two-body all-to-all interactions 
         *
         * @tparam FloatType float type of Sparse class (double or float)
         */
        template<typename FloatType>
            class Dense : public Graph{
                static_assert(std::is_floating_point<FloatType>::value, "FloatType must be floating-point type.");
                public:

                    /**
                     * @brief interaction type (Eigen)
                     */
                    using Interactions = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

                    /**
                     * @brief float type
                     */
                    using value_type = FloatType;
                    
                private:

                    /**
                     * @brief interactions 
                     */
                    Interactions _J;

                public:

                    /**
                     * @brief Dense constructor
                     *
                     * @param num_spins the number of spins
                     */
                    explicit Dense(std::size_t num_spins)
                        : Graph(num_spins), _J(Interactions::Zero(num_spins+1, num_spins+1)){
                            _J(num_spins, num_spins) = 1;
                        }

                    /**
                     * @brief Dense constructor (from nlohmann::json)
                     *
                     * @param j JSON object
                     */
                    Dense(const json& j) : Dense(static_cast<size_t>(j["num_variables"])){
                        //define bqm with ising variables
                        auto bqm = json_parse<FloatType>(j);
                        //interactions
                        for(auto&& elem : bqm.get_quadratic()){
                            const auto& key = elem.first;
                            const auto& val = elem.second;
                            J(key.first, key.second) += val;
                        }
                        //local field
                        for(auto&& elem : bqm.get_linear()){
                            const auto& key = elem.first;
                            const auto& val = elem.second;
                            h(key) += val;
                        }
                    }

                    /**
                     * @brief Dense constructor (from Eigen Matrix)
                     *
                     * @param interaction Eigen matrix
                     */
                    void set_interaction_matrix(const Interactions& interaction){
                        if(interaction.rows() != interaction.cols()){
                            std::runtime_error("interaction.rows() != interaction.cols()");
                        }

                        if((size_t)interaction.rows() != get_num_spins() + 1){
                            std::runtime_error("invalid matrix size.");
                        }

                        //check if diagonal elements are zero
                        for(size_t i=0; i<(size_t)(interaction.rows()-1); i++){
                            if(interaction(i,i) != 0){
                                std::runtime_error("The diagonal elements of interaction matrix must be zero.");
                            }
                        }

                        if(interaction(interaction.rows()-1,interaction.rows()-1) != 1){
                            std::runtime_error("The right bottom element of interaction matrix must be unity.");
                        }

                        _J = interaction.template selfadjointView<Eigen::Upper>();
                    }


                    /**
                     * @brief Dense copy constructor
                     */
                    Dense(const Dense<FloatType>&) = default;

                    /**
                     * @brief Dense move constructor
                     */
                    Dense(Dense<FloatType>&&) = default;

                    /**
                     * @brief calculate total energy 
                     *
                     * @param spins
                     *
                     * @return corresponding energy
                     */
                    FloatType calc_energy(const Spins& spins) const{
                        if(spins.size() != this->get_num_spins()){
                            std::out_of_range("Out of range in calc_energy in Dense graph.");
                        }

                        using Vec = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;
                        Vec s(get_num_spins()+1);
                        for(size_t i=0; i<spins.size(); i++){
                            s(i) = spins[i];
                        }
                        s(get_num_spins()) = 1;

                        return (s.transpose()*(_J*s))(0,0)-1;
                    }

                    FloatType calc_energy(const Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>& spins) const{
                        graph::Spins temp_spins(get_num_spins());
                        for(size_t i=0; i<temp_spins.size(); i++){
                            temp_spins[i] = spins(i);
                        }
                        return calc_energy(temp_spins);

                    }

                    /**
                     * @brief access J_{ij}
                     *
                     * @param i Index i
                     * @param j Index j
                     *
                     * @return J_{ij}
                     */
                    FloatType& J(Index i, Index j){
                        assert(i < get_num_spins());
                        assert(j < get_num_spins());

                        if(i != j)
                            return _J(std::min(i, j), std::max(i, j));
                        else
                            return _J(std::min(i, j), get_num_spins());
                    }

                    /**
                     * @brief access J_{ij}
                     *
                     * @param i Index i
                     * @param j Index j
                     *
                     * @return J_{ij}
                     */
                    const FloatType& J(Index i, Index j) const{
                        assert(i < get_num_spins());
                        assert(j < get_num_spins());

                        if(i != j)
                            return _J(std::min(i, j), std::max(i, j));
                        else
                            return _J(std::min(i, j), get_num_spins());
                    }

                    /**
                     * @brief access h_{i} (local field)
                     *
                     * @param i Index i
                     *
                     * @return h_{i}
                     */
                    FloatType& h(Index i){
                        assert(i < get_num_spins());
                        return J(i, i);
                    }

                    /**
                     * @brief access h_{i} (local field)
                     *
                     * @param i Index i
                     *
                     * @return h_{i}
                     */
                    const FloatType& h(Index i) const{
                        assert(i < get_num_spins());
                        return J(i, i);
                    }

                    /**
                     * @brief get interactions (Eigen Matrix)
                     *
                     * @return Eigen Matrix
                     */
                    const Interactions get_interactions() const{
                        return this->_J.template selfadjointView<Eigen::Upper>();
                    }

            };
    } // namespace graph 
} // namespace openjij

#endif
