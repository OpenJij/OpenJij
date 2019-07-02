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

#ifndef OPENJIJ_GRAPH_SPARSE_HPP__
#define OPENJIJ_GRAPH_SPARSE_HPP__

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <unordered_map>

#include <graph/graph.hpp>
#include <utility/pairhash.hpp>

namespace openjij {
    namespace graph {

        /**
         * @brief Sparse graph: two-body intereactions with O(1) connectivity
         *
         * @tparam FloatType floating-point type
         */
        template<typename FloatType>
            class Sparse : public Graph{
                static_assert(std::is_floating_point<FloatType>::value, "FloatType must be floating-point type.");
                public:

                    /**
                     * interaction type
                     */
                    using Interactions = std::unordered_map<std::pair<Index, Index>, FloatType, utility::PairHash>;
                private:

                    /**
                     * interactions (the number of intereactions is num_spins*(num_spins+1)/2)
                     */
                    Interactions _J;

                    /**
                     * the uppder limit of the number of edges per site 
                     */
                    std::size_t _num_edges;

                    /**
                     * the list of the indices of adjacent nodes 
                     */
                    std::vector<Nodes> _list_adj_nodes;

                    /**
                     * add adjacent node from "from" Index to "to" Index 
                     *
                     * @param from "from" Index
                     * @param to "to" Index
                     *
                     * @return true if nodes is successfully added, false if this function failed.
                     */
                    inline bool set_adj_node(Index from, Index to){
                        assert(from < this->get_num_spins());
                        assert(to < this->get_num_spins());

                        //get adjacent nodes of node "from"
                        Nodes& nodes = _list_adj_nodes[from];
                        //check if the node "to" exists in the nodes 
                        if(std::find(nodes.begin(), nodes.end(), to)==nodes.end()){
                            //nodes size must be smaller than num_edges

#ifndef NDEBUG
                            assert(nodes.size() < _num_edges);
#else
                            //return false
                            if(nodes.size() >= _num_edges) return false;
#endif
                            //add node
                            nodes.push_back(to);
                            //add node from "to" to "from"
                            set_adj_node(to, from);
                        }
                        return true;
                    }

                public:

                    /**
                     * Sparse constructor
                     *
                     * @param num_spins number of spins
                     * @param num_edges number of edges
                     */
                    Sparse(std::size_t num_spins, std::size_t num_edges)
                        : Graph(num_spins), _num_edges(std::min(num_spins, num_edges)), _list_adj_nodes(num_spins){
                            //initialize list_adj_nodes
                            for(auto& elem : _list_adj_nodes){
                                elem.reserve(_num_edges); //not resize()
                            }
                        }

                    /**
                     * Sparse delegate constructor
                     *
                     * @param num_spins number of spins
                     */
                    explicit Sparse(std::size_t num_spins) : Sparse(num_spins, num_spins){}

                    /**
                     * Sparse copy constructor
                     *
                     */
                    Sparse(const Sparse<FloatType>&) = default;

                    /**
                     * Sparse move constructor
                     *
                     */
                    Sparse(Sparse<FloatType>&&) = default;

                    /**
                     * list of adjacent nodes 
                     *
                     * @param ind Node index
                     *
                     * @return corresponding list of adjacent nodes
                     */
                    const Nodes& adj_nodes(Index ind) const{
                        return _list_adj_nodes[ind];
                    }

                    /**
                     * get number of edges
                     *
                     * @return number of edges
                     */
                    std::size_t get_num_edges() const{
                        return _num_edges;
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

                        //i <= j
                        //add node if it does not exist
                        set_adj_node(i, j);
                        return _J[std::make_pair(std::min(i, j), std::max(i, j))];
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
                        return _J.at(std::make_pair(std::min(i, j), std::max(i, j)));
                    }

                    /**
                     * access h_{i} (local field)
                     *
                     * @param i Index i
                     *
                     * @return h_{i}
                     */
                    FloatType& h(Index i){
                        assert(i < get_num_spins());
                        set_adj_node(i, i);
                        return _J[std::make_pair(i, i)];
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
                        return _J.at(std::make_pair(i, i));
                    }

            };
    } // namespace graph
} // namespace openjij

#endif
