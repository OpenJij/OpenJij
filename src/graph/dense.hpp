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

#include <graph/graph.hpp>

namespace openjij {
    namespace graph {

        /**
         * two-body all-to-all interactions 
         *
         * @tparam FloatType float type of Sparse class (double or float)
         */
        template<typename FloatType>
            class Dense : public Graph{
                static_assert(std::is_floating_point<FloatType>::value, "FloatType must be floating-point type.");
                public:

                    /**
                     * interaction type
                     */
                    using Interactions = std::vector<FloatType>;
                private:

                    /**
                     * interactions (the number of intereactions is num_spins*(num_spins+1)/2) 
                     */
                    Interactions _J;


                    /**
                     * convert index from pair (i,j) to unique value
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
                     * the list of the indices of adjacent nodes
                     */
                    std::vector<Nodes> _list_adj_nodes;

                    /**
                     * add adjacent node from "from" Index to "to" Index
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
                     * Dense constructor
                     *
                     * @param num_spins the number of spins
                     */
                    explicit Dense(std::size_t num_spins)
                        : Graph(num_spins), _J(num_spins*(num_spins+1)/2), _list_adj_nodes(num_spins){


                            //initialize list_adj_nodes
                            for(auto& elem : _list_adj_nodes){
                                elem.reserve(num_spins);
                            }
                        }


                    /**
                     * Dense copy constructor
                     */
                    Dense(const Dense<FloatType>&) = default;

                    /**
                     * Dense move constructor
                     */
                    Dense(Dense<FloatType>&&) = default;

                    /**
                     * returns list of adjacent nodes 
                     *
                     * @param ind Node index
                     *
                     * @return list of adjacent nodes
                     */
                    const Nodes& adj_nodes(Index ind) const{
                        return _list_adj_nodes[ind];
                    }

                    //TODO: calc_energy should not be the member function.

                    /**
                     * calculate total energy 
                     *
                     * @param spins
                     *
                     * @return corresponding energy
                     */
                    FloatType calc_energy(const Spins& spins) const{
                        assert(spins.size() == get_num_spins());
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
                     * access J_{ij}
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
                     * access J_{ij}
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

                    //access to h (local field)

                    /**
                     * access h_{i} (local field)
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
                     * access h_{i} (local field)
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
