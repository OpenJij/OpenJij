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
                     * @brief interaction type
                     */
                    using Interactions = std::vector<FloatType>;

                    /**
                     * @brief float type
                     */
                    using value_type = FloatType;
                    
                private:

                    /**
                     * @brief interactions (the number of intereactions is num_spins*(num_spins+1)/2) 
                     */
                    Interactions _J;


                    /**
                     * @brief convert index from pair (i,j) to unique value
                     *
                     * @param i Index i
                     * @param j Index j
                     *
                     * @return converted index
                     */
                    inline std::size_t convert_index(Index i, Index j) const{
                        assert(i<=j);
                        return get_num_spins()*i-i*(i-1)/2+j-i;
                    }

                    /**
                     * @brief the list of the indices of adjacent nodes
                     */
                    std::vector<Nodes> _list_adj_nodes;

                    /**
                     * @brief add adjacent node from "from" Index to "to" Index
                     *
                     * @param from "from" Index
                     * @param to "to" Index
                     */
                    inline void set_adj_node(Index from, Index to){
                        assert(from < this->get_num_spins());
                        assert(to < this->get_num_spins());

                        //get adjacent nodes of node "from"
                        Nodes& nodes = _list_adj_nodes[from];
                        //check if the node "to" exists in the nodes 
                        if(std::find(nodes.begin(), nodes.end(), to)==nodes.end()){
                            //add node
                            nodes.push_back(to);
                            //add node from "to" to "from"
                            set_adj_node(to, from);
                        }
                    }
                public:

                    /**
                     * @brief Dense constructor
                     *
                     * @param num_spins the number of spins
                     */
                    explicit Dense(std::size_t num_spins)
                        : Graph(num_spins), _J(num_spins*(num_spins+1)/2, 0), _list_adj_nodes(num_spins){

                            //initialize list_adj_nodes
                            for(auto& elem : _list_adj_nodes){
                                elem.reserve(num_spins);
                            }
                        }

                    /**
                     * @brief Dense constructor (from nlohmann::json)
                     *
                     * @param j JSON object
                     */
                    Dense(const json& j) : Dense(j["num_variables"]){
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
                     * @brief Dense copy constructor
                     */
                    Dense(const Dense<FloatType>&) = default;

                    /**
                     * @brief Dense move constructor
                     */
                    Dense(Dense<FloatType>&&) = default;

                    /**
                     * @brief returns list of adjacent nodes 
                     *
                     * @param ind Node index
                     *
                     * @return list of adjacent nodes
                     */
                    const Nodes& adj_nodes(Index ind) const{
                        return _list_adj_nodes[ind];
                    }

                    //DEPRECATED
                    //TODO: calc_energy should not be the member function.

                    /**
                     * @brief calculate total energy 
                     *
                     * @param spins
                     *
                     * @return corresponding energy
                     */
                    FloatType calc_energy(const Spins& spins) const{
                        if(!(spins.size() == get_num_spins())){
                            throw std::runtime_error("invalid length of spins in Dense");
                        }
                        FloatType ret = 0;
                        for(std::size_t ind=0; ind<this->get_num_spins(); ind++){
                            for(auto& adj_ind : _list_adj_nodes[ind]){
                                if(ind != adj_ind)
                                    ret += (1./2) * this->J(ind, adj_ind) * spins[ind] * spins[adj_ind];
                                else
                                    ret += this->h(ind) * spins[ind];
                            }
                        }

                        return ret;
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

                        set_adj_node(i, j);
                        return _J[convert_index(std::min(i, j), std::max(i, j))];
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

                        return _J[convert_index(std::min(i, j), std::max(i, j))];
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
                        //add node if it does not exist
                        set_adj_node(i, i);
                        return _J[convert_index(i, i)];
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
                        return _J[convert_index(i, i)];
                    }

            };
    } // namespace graph 
} // namespace openjij

#endif
