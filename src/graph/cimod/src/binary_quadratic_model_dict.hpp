//    Copyright 2021 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.


/**
 * @file binary_quadratic_model.hpp
 * @author Fumiya Watanabe
 * @brief BinaryQuadraticModel class
 * @version 1.0.0
 * @date 2020-03-24
 * 
 * @copyright Copyright (c) Jij Inc. 2020
 * 
 */

#ifndef BINARY_QUADRATIC_MODEL_DICT_HPP__
#define BINARY_QUADRATIC_MODEL_DICT_HPP__

#include "disable_eigen_warning.hpp"

#include "vartypes.hpp"
#include "hash.hpp"
#include "utilities.hpp"
#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <functional>
#include <Eigen/Dense>

#include "binary_quadratic_model.hpp"

namespace cimod
{

    struct Dict{};
    
    /**
     * @brief Class for legacy binary quadratic model (dict datastructure).
     */
    
    template <typename IndexType, typename FloatType>
    class BinaryQuadraticModel<IndexType, FloatType, Dict>
    {
        using DataType = Dict;
    public:

        using DenseMatrix   = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using SparseMatrix  = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;

        /**
         * @brief Eigen Matrix
         */
        using Matrix = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    protected:
        /**
         * @brief Linear biases as a dictionary.
         * 
         */
        Linear<IndexType, FloatType> m_linear;
    
        /**
         * @brief Quadratic biases as a dictionary.
         * 
         */
        Quadratic<IndexType, FloatType> m_quadratic;
    
        /**
         * @brief The energy offset associated with the model.
         * 
         */
        FloatType m_offset;
    
        /**
         * @brief The model's type.
         * 
         */
        Vartype m_vartype = Vartype::NONE;
    
    public:
        /**
         * @brief BinaryQuadraticModel constructor.
         * 
         * @param linear
         * @param quadratic
         * @param offset
         * @param vartype
         */
        BinaryQuadraticModel
        (
            const Linear<IndexType, FloatType> &linear,
            const Quadratic<IndexType, FloatType> &quadratic,
            const FloatType &offset,
            const Vartype vartype
        ):
            m_offset(offset),
            m_vartype(vartype)
        {
            add_variables_from(linear);
            add_interactions_from(quadratic);
        }

        /**
         * @brief BinaryQuadraticModel constructor.
         * 
         * @param linear
         * @param quadratic
         * @param vartype
         */
        BinaryQuadraticModel
        (
            const Linear<IndexType, FloatType> &linear,
            const Quadratic<IndexType, FloatType> &quadratic,
            const Vartype vartype
        ): BinaryQuadraticModel(linear, quadratic, 0.0, vartype){}

        /**
         * @brief BinaryQuadraticModel constructor (with matrix);
         * This constructor is not be implemented.
         *
         * @param mat
         * @param labels_vec
         * @param offset
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const Eigen::Ref<const DenseMatrix>&,
            const std::vector<IndexType>&,
            const FloatType&,
            const Vartype,
            bool
        )
        {
            throw std::runtime_error("Initialization from matrix is not implemented on dict-type BQM");
        }

        /**
         * @brief BinaryQuadraticModel constructor (with matrix);
         * This constructor is not be implemented.
         *
         * @param mat
         * @param labels_vec
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const Eigen::Ref<const DenseMatrix>&,
            const std::vector<IndexType>&,
            const Vartype,
            bool
        )
        {
            throw std::runtime_error("Initialization from matrix is not implemented on dict-type BQM");
        }

        /**
         * @brief BinaryQuadraticModel constructor (with sparse matrix);
         * this constructor is for developers.
         *
         * @param mat
         * @param labels_vec
         * @param offset
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const SparseMatrix&,
            const std::vector<IndexType>&,
            const FloatType&,
            const Vartype
        )
        {
            throw std::runtime_error("Initialization from matrix is not implemented on dict-type BQM");
        }
    
        /**
         * @brief BinaryQuadraticModel constructor (with sparse matrix);
         * this constructor is for developers.
         *
         * @param mat
         * @param labels_vec
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const SparseMatrix&,
            const std::vector<IndexType>&,
            const Vartype
        )
        {
            throw std::runtime_error("Initialization from matrix is not implemented on dict-type BQM");
        }
    
        /**
         * @brief Copy constructor of BinaryQuadraticModel
         * 
         * @param bqm 
         */
        BinaryQuadraticModel(const BinaryQuadraticModel&) = default;
    
        /**
         * @brief generate indices
         *
         * @return generated indices
         */
        std::vector<IndexType> _generate_indices() const{
            std::unordered_set<IndexType> index_set;
            for(auto elem : m_linear){
                index_set.insert(elem.first);
            }
    
            for(auto elem : m_quadratic){
                index_set.insert(elem.first.first);
                index_set.insert(elem.first.second);
            }
    
            auto ret = std::vector<IndexType>(index_set.begin(), index_set.end());
            std::sort(ret.begin(), ret.end());
            return ret;
        }
    
        /**
         * @brief Return the number of variables.
         * 
         * @deprecated use get_num_variables.
         * @return The number of variables.
         */
        size_t length() const
        {
            return get_num_variables();
        }

        /**
         * @brief Return the number of variables.
         * 
         * @deprecated use get_num_variables.
         * @return The number of variables.
         */
        size_t get_num_variables() const{
            return m_linear.size();
        }
    
        /**
         * @brief Return true if the variable contains v.
         * 
         * @return Return true if the variable contains v.
         * @param v
         */
        bool contains
        (const IndexType &v) const
        {
            if(m_linear.count(v)!=0)
            {
                return true;
            }
            else
            {
                return false;
            }
            
        }

        /**
         * @brief Get the element of linear object
         *
         * @param label_i
         *
         * @return 
         */
        FloatType get_linear(IndexType label_i) const{
            return this->m_linear.at(label_i);
        }
        
        /**
         * @brief Get the linear object
         * 
         * @return A linear bias.
         */
        const Linear<IndexType, FloatType>& get_linear() const
        {
            return this->m_linear;
        }

        /**
         * @brief Get the element of quadratic object
         * 
         * @return A quadratic bias.
         */
        FloatType get_quadratic(IndexType label_i, IndexType label_j) const
        {
            return this->m_quadratic.at(std::make_pair(std::min(label_i, label_j), std::max(label_i, label_j)));
        }
    
        /**
         * @brief Get the quadratic object
         * 
         * @return A quadratic bias.
         */
        const Quadratic<IndexType, FloatType>& get_quadratic() const
        {
            return this->m_quadratic;
        }
    
        /**
         * @brief Get the offset
         * 
         * @return An offset.
         */
        const FloatType& get_offset() const
        {
            return this->m_offset;
        }
    
        /**
         * @brief Get the vartype object
         * 
         * @return Type of the model.
         */
        const Vartype& get_vartype() const
        {
            return this->m_vartype;
        }

        /**
         * @brief Get variables
         *
         * @return variables
         */
        const std::vector<IndexType> get_variables() const{
            std::vector<IndexType> variables;
            variables.reserve(m_linear.size());
            for(auto&& elem : m_linear)
            {
                variables.push_back(elem.first);
            }
    
            std::sort(variables.begin(), variables.end());

            return variables;
        }
    
        /**
         * @brief Create an empty BinaryQuadraticModel
         * 
         */
        BinaryQuadraticModel<IndexType, FloatType, DataType> empty(Vartype vartype)
        {
            return BinaryQuadraticModel<IndexType, FloatType, DataType>(
                    Linear<IndexType, FloatType>(),
                    Quadratic<IndexType, FloatType>(),
                    0.0,
                    vartype
                    );
        }
    
        /* Update methods */
    
        /**
         * @brief Add variable v and/or its bias to a binary quadratic model.
         * 
         * @param v
         * @param bias
         */
        void add_variable
        (
            const IndexType &v,
            const FloatType &bias
        )
        {
            FloatType b = bias;

            Vartype vartype = this->get_vartype();
    
            // handle the case that a different vartype is provided
            if((vartype!=Vartype::NONE)&&(vartype!=m_vartype))
            {
                if((m_vartype == Vartype::SPIN)&&(vartype == Vartype::BINARY))
                {
                    b /= 2;
                    m_offset += b;
                }
                else if((m_vartype == Vartype::BINARY)&&(vartype == Vartype::SPIN))
                {
                    m_offset -= b;
                    b *= 2;
                }
                else
                {
                    throw std::runtime_error("Unknown vartype");
                }
            }
    
            // Insert or assign the bias
            FloatType value = 0;
            if(m_linear.count(v) != 0)
            {
                value = m_linear[v];
            }
            insert_or_assign(m_linear, v, value + b);
        }
    
        /**
         * @brief Add variables and/or linear biases to a binary quadratic model.
         * 
         * @param linear
         */
        void add_variables_from
        (
            const Linear<IndexType, FloatType> &linear
        )
        {
            for(auto &it : linear)
            {
                add_variable(it.first, it.second);
            }
        }
    
        /**
         * @brief Add an interaction and/or quadratic bias to a binary quadratic model.
         * 
         * @param arg_u
         * @param arg_v
         * @param bias
         */
        void add_interaction
        (
            const IndexType &arg_u,
            const IndexType &arg_v,
            const FloatType &bias
        )
        {
            IndexType u = std::min(arg_u, arg_v);
            IndexType v = std::max(arg_u, arg_v);

            Vartype vartype = this->get_vartype();

            if(u == v)
            {
                throw std::runtime_error("No self-loops allowed");
            }
            else
            {
                //Check vartype when m_linear.empty() is true
                if (m_linear.empty() && m_vartype == Vartype::NONE) {
                   if (vartype != Vartype::NONE) {
                      m_vartype = vartype;
                   }
                   else {
                      throw std::runtime_error("Binary quadratic model is empty. Please set vartype to Vartype::SPIN or Vartype::BINARY");
                   }
                }
               
                FloatType b = bias;
                if((vartype!=Vartype::NONE)&&(vartype!=m_vartype))
                {
                    if((m_vartype == Vartype::SPIN)&&(vartype == Vartype::BINARY))
                    {
                        //convert from binary to spin
                        b /= 4;
                        add_offset(b);
                        add_variable(u, b);
                        add_variable(v, b);
                    }
                    else if((m_vartype == Vartype::BINARY)&&(vartype == Vartype::SPIN))
                    {
                        //convert from spin to binary
                        add_offset(b);
                        add_variable(u, -2 * b);
                        add_variable(v, -2 * b);
                        b *= 4;
                    }
                    else
                    {
                        throw std::runtime_error("Unknown vartype");
                    }
                }
                else
                {
                    if(m_linear.count(u) == 0)
                    {
                        add_variable(u, 0);
                    }
                    if(m_linear.count(v) == 0)
                    {
                        add_variable(v, 0);
                    }
                }
                
                FloatType value = 0;
                std::pair<IndexType, IndexType> p1 = std::make_pair(u, v);
                if(m_quadratic.count(p1) != 0)
                {
                    value = m_quadratic[p1];
                }
                insert_or_assign(m_quadratic, p1, value + b);
            }
        }
    
        /**
         * @brief Add interactions and/or quadratic biases to a binary quadratic model.
         * 
         * @param quadratic
         */
        void add_interactions_from
        (
            const Quadratic<IndexType, FloatType> &quadratic
        )
        {
            for(auto &it : quadratic)
            {
                add_interaction(it.first.first, it.first.second, it.second);
            }
        }
    
        
    
        /**
         * @brief Remove variable v and all its interactions from a binary quadratic model.
         * 
         * @param v
         */
        void remove_variable
        (
            const IndexType &v
        )
        {
            std::vector<std::pair<IndexType, IndexType>> interactions;
            for(auto &it : m_quadratic)
            {
                if(it.first.first == v || it.first.second == v)
                {
                    interactions.push_back(it.first);
                }         
            }
            remove_interactions_from(interactions);
            m_linear.erase(v);
        }
    
        /**
         * @brief Remove specified variables and all of their interactions from a binary quadratic model.
         * 
         * @param variables
         */
        void remove_variables_from
        (
            const std::vector<IndexType> &variables
        )
        {
            for(auto &it : variables)
            {
                remove_variable(it);
            }
        }
    
        /**
         * @brief Remove interaction of variables u, v from a binary quadratic model.
         * 
         * @param arg_u
         * @param arg_v
         */
        void remove_interaction
        (
            const IndexType &arg_u,
            const IndexType &arg_v      
        )
        {
            IndexType u = std::min(arg_u, arg_v);
            IndexType v = std::max(arg_u, arg_v);
            auto p = std::make_pair(u, v);
            if(m_quadratic.count(p)!=0)
            {
                m_quadratic.erase(p);
            }

            bool u_flag = true;
            for(auto &it : m_quadratic)
            {
                if((it.first.first == u) || (it.first.second == u)){
                    u_flag = false;
                    break;
                }
            }

            if(u_flag && (m_linear[u] == 0)){ 
                remove_variable(u);
            }

            bool v_flag = true;
            for(auto &it : m_quadratic)
            {
                if((it.first.first == v) || (it.first.second == v)){
                    v_flag = false;
                    break;
                }
            }

            if(v_flag && (m_linear[v] == 0)){ 
                remove_variable(v);
            }
        }
    
        /**
         * @brief Remove all specified interactions from the binary quadratic model.
         * 
         * @param interactions
         */
        void remove_interactions_from
        (
            const std::vector<std::pair<IndexType, IndexType>> &interactions
        )
        {
            for(auto &it : interactions)
            {
                remove_interaction(it.first, it.second);
            }
        }
    
        /**
         * @brief Add specified value to the offset of a binary quadratic model.
         * 
         * @param offset
         */
        void add_offset
        (
            const FloatType &offset
        )
        {
            m_offset += offset;
        }
    
        /**
         * @brief Set the binary quadratic model's offset to zero.
         */
        void remove_offset()
        {
            add_offset(-m_offset);
        }
    
        /**
         * @brief Multiply by the specified scalar all the biases and offset of a binary quadratic model.
         * 
         * @param scalar
         * @param ignored_variables
         * @param ignored_interactions
         * @param ignored_offset
         */
        void scale
        (
            const FloatType &scalar,
            const std::vector<IndexType> &ignored_variables = {},
            const std::vector<std::pair<IndexType, IndexType>> &ignored_interactions = {},
            const bool ignored_offset = false
        )
        {
            // scaling linear
            for(auto &it : m_linear)
            {
                if(std::find(ignored_variables.begin(), ignored_variables.end(), it.first) != ignored_variables.end() || ignored_variables.empty())
                {
                    it.second *= scalar;
                }
            }
    
            // scaling quadratic
            for(auto &it : m_quadratic)
            {
                if(std::find(ignored_interactions.begin(), ignored_interactions.end(), it.first) != ignored_interactions.end() || ignored_variables.empty())
                {
                    it.second *= scalar;
                }
            }
    
            // scaling offset
            if(!ignored_offset)
            {
                m_offset *= scalar;
            }
        }
    
        /**
         * @brief Normalizes the biases of the binary quadratic model such that they fall in the provided range(s), and adjusts the offset appropriately.
         * 
         * @param bias_range
         * @param use_quadratic_range
         * @param quadratic_range
         * @param ignored_variables
         * @param ignored_interactions
         * @param ignored_offset
         * 
         */
        void normalize
        (
            const std::pair<FloatType, FloatType> &bias_range = {1.0, 1.0},
            const bool use_quadratic_range = false,
            const std::pair<FloatType, FloatType> &quadratic_range = {1.0, 1.0},
            const std::vector<IndexType> &ignored_variables = {},
            const std::vector<std::pair<IndexType, IndexType>> &ignored_interactions = {},
            const bool ignored_offset = false
        )
        {
            if (m_linear.empty()) {
               return;
            }
            // parse range
            std::pair<FloatType, FloatType> l_range = bias_range;
            std::pair<FloatType, FloatType> q_range;
            if(!use_quadratic_range)
            {
                q_range = bias_range;
            }
            else
            {
                q_range = quadratic_range;
            }
    
            // calculate scaling value
            auto comp = [](const auto &a, const auto &b) { return a.second < b.second; };
            auto it_lin_min = std::min_element(m_linear.begin(), m_linear.end(), comp);
            auto it_lin_max = std::max_element(m_linear.begin(), m_linear.end(), comp);
            auto it_quad_min = std::min_element(m_quadratic.begin(), m_quadratic.end(), comp);
            auto it_quad_max = std::max_element(m_quadratic.begin(), m_quadratic.end(), comp);
    
            std::vector<FloatType> v_scale =
            {
                it_lin_min->second / l_range.first,
                it_lin_max->second / l_range.second,
                it_quad_min->second / q_range.first,
                it_quad_max->second / q_range.second
            };
            FloatType inv_scale = *std::max_element(v_scale.begin(), v_scale.end());
    
            // scaling
            if(inv_scale != 0.0)
            {
                scale(1.0 / inv_scale, ignored_variables, ignored_interactions, ignored_offset);
            }
        }
    
        /**
         * @brief Fix the value of a variable and remove it from a binary quadratic model.
         * 
         * @param v
         * @param value
         */
        void fix_variable
        (
            const IndexType &v,
            const int32_t &value
        )
        {
            std::vector<std::pair<IndexType, IndexType>> interactions;
            for(auto &it : m_quadratic)
            {
                if(it.first.first == v)
                {
                    add_variable(it.first.second, value*it.second);
                    interactions.push_back(it.first);
                }
                else if(it.first.second == v)
                {
                    add_variable(it.first.first, value*it.second);
                    interactions.push_back(it.first);
                }
            }
            remove_interactions_from(interactions);
            add_offset(m_linear[v]*value);
            remove_variable(v);
        }
    
        /**
         * @brief Fix the value of the variables and remove it from a binary quadratic model.
         * 
         * @param fixed
         */
        void fix_variables
        (
            const std::vector<std::pair<IndexType, int32_t>> &fixed
        )
        {
            for(auto &it : fixed)
            {
                fix_variable(it.first, it.second);
            }
        }
    
        /**
         * @brief Flip variable v in a binary quadratic model.
         * 
         * @param v
         */
        void flip_variable
        (
            const IndexType &v
        )
        {
            // check variable
            if(m_linear.count(v)==0)
            {
                throw std::runtime_error("not a variable in the binary quadratic model.");
            }
    
            if(m_vartype == Vartype::SPIN)
            {
                m_linear[v] *= -1.0;
                for(auto &it : m_quadratic)
                {
                    if(it.first.first == v || it.first.second == v)
                    {
                        it.second *= -1.0;
                    }
                }
            }
            else if(m_vartype == Vartype::BINARY)
            {
                add_offset(m_linear[v]);
                m_linear[v] *= -1.0;
    
                for(auto &it : m_quadratic)
                {
                    if(it.first.first == v)
                    {
                        m_linear[it.first.second] += it.second;
                        it.second *= -1.0;
                    }
                    else if(it.first.second == v)
                    {
                        m_linear[it.first.first] += it.second;
                        it.second *= -1.0;
                    }
                }
            }
        }
    
        /**
         * @brief Enforce u, v being the same variable in a binary quadratic model.
         * 
         * @param u
         * @param v
         */
        void contract_variables
        (
            const IndexType &u,
            const IndexType &v
        )
        {
            // check variable
            if(m_linear.count(v)==0)
            {
                throw std::runtime_error("not a variable in the binary quadratic model.");
            }
            if(m_linear.count(u)==0)
            {
                throw std::runtime_error("not a variable in the binary quadratic model.");
            }
    
            auto p1 = std::make_pair(u, v);
            auto p2 = std::make_pair(v, u);
            if(m_quadratic.count(p1) != 0)
            {
                if(m_vartype == Vartype::BINARY)
                {
                    add_variable(u, m_quadratic[p1]);
                }
                else if(m_vartype == Vartype::SPIN)
                {
                    add_offset(m_quadratic[p1]);
                }
                remove_interaction(u, v);
            }
            if(m_quadratic.count(p2) != 0)
            {
                if(m_vartype == Vartype::BINARY)
                {
                    add_variable(u, m_quadratic[p2]);
                }
                else if(m_vartype == Vartype::SPIN)
                {
                    add_offset(m_quadratic[p2]);
                }
                remove_interaction(v, u);
            }
    
            std::vector<std::pair<IndexType, IndexType>> interactions;
            for(auto &it : m_quadratic)
            {
                if(it.first.first == v)
                {
                    add_interaction(u, it.first.second, it.second);
                    interactions.push_back(it.first);
                }
                else if(it.first.second == v)
                {
                    add_interaction(it.first.first, u, it.second);
                    interactions.push_back(it.first);
                }
            }
            remove_interactions_from(interactions);
    
            add_variable(u, m_linear[v]);
            remove_variable(v);
        }
    
        /* Transformations */
    
        /**
         * @brief Create a binary quadratic model with the specified vartype.
         * This function does not return any object.
         * 
         * @param vartype
         */
        void change_vartype
        (
            const Vartype &vartype
        )
        {
            Linear<IndexType, FloatType> linear;
            Quadratic<IndexType, FloatType> quadratic;
            FloatType offset = 0.0;
    
            if(m_vartype == Vartype::BINARY && vartype == Vartype::SPIN) // binary -> spin
            {
                std::tie(linear, quadratic, offset) = binary_to_spin(m_linear, m_quadratic, m_offset);                
            }
            else if(m_vartype == Vartype::SPIN && vartype == Vartype::BINARY) // spin -> binary
            {
                std::tie(linear, quadratic, offset) = spin_to_binary(m_linear, m_quadratic, m_offset);
            }
            else
            {
                std::tie(linear, quadratic, offset) = std::tie(m_linear, m_quadratic, m_offset);
            }
    
            //inplace
            m_linear = linear;
            m_quadratic = quadratic;
            m_offset = offset;
            m_vartype = vartype;
        }

        /**
         * @brief Create a binary quadratic model with the specified vartype.
         * 
         * @param vartype
         * @param inplace
         * @return A new instance of the BinaryQuadraticModel class.
         */
        BinaryQuadraticModel change_vartype
        (
            const Vartype &vartype,
            bool inplace
        )
        {
            Linear<IndexType, FloatType> linear;
            Quadratic<IndexType, FloatType> quadratic;
            FloatType offset = 0.0;
    
            if(m_vartype == Vartype::BINARY && vartype == Vartype::SPIN) // binary -> spin
            {
                std::tie(linear, quadratic, offset) = binary_to_spin(m_linear, m_quadratic, m_offset);                
            }
            else if(m_vartype == Vartype::SPIN && vartype == Vartype::BINARY) // spin -> binary
            {
                std::tie(linear, quadratic, offset) = spin_to_binary(m_linear, m_quadratic, m_offset);
            }
            else
            {
                std::tie(linear, quadratic, offset) = std::tie(m_linear, m_quadratic, m_offset);
            }
    
            BinaryQuadraticModel bqm(linear, quadratic, offset, vartype);
            
            if(inplace == true)
            {
                //inplace
                m_linear = bqm.get_linear();
                m_quadratic = bqm.get_quadratic();
                m_offset = bqm.get_offset();
                m_vartype = bqm.get_vartype();
            }
    
            return bqm;
        }
        
        /* Static methods */
        /**
         * @brief Convert linear, quadratic, and offset from spin to binary. Does no checking of vartype. Copies all of the values into new objects.
         * 
         * @param linear
         * @param quadratic
         * @param offset
         * 
         * @return A tuple including a linear bias, a quadratic bias and an offset.
         */
        static std::tuple<Linear<IndexType, FloatType>, Quadratic<IndexType, FloatType>, FloatType> spin_to_binary
        (
            const Linear<IndexType, FloatType> &linear,
            const Quadratic<IndexType, FloatType> &quadratic,
            const FloatType &offset
        )
        {
            Linear<IndexType, FloatType> new_linear;
            Quadratic<IndexType, FloatType> new_quadratic;
            FloatType new_offset = offset;
            FloatType linear_offset = 0.0;
            FloatType quadratic_offset = 0.0;
    
            for(auto &it : linear)
            {
                insert_or_assign(new_linear, it.first, static_cast<FloatType>(2.0 * it.second));
                linear_offset += it.second;
            }
    
            for(auto &it : quadratic)
            {
                insert_or_assign(new_quadratic, it.first, static_cast<FloatType>(4.0 * it.second));
                new_linear[it.first.first] -= 2.0 * it.second;
                new_linear[it.first.second] -= 2.0 * it.second;
                quadratic_offset += it.second; 
            }
    
            new_offset += quadratic_offset - linear_offset;
    
            return std::make_tuple(new_linear, new_quadratic, new_offset);
        }
    
        /**
         * @brief Convert linear, quadratic and offset from binary to spin. Does no checking of vartype. Copies all of the values into new objects.
         * 
         * @param linear
         * @param quadratic
         * @param offset
         * 
         * @return A tuple including a linear bias, a quadratic bias and an offset.
         */
        static std::tuple<Linear<IndexType, FloatType>, Quadratic<IndexType, FloatType>, FloatType> binary_to_spin
        (
            const Linear<IndexType, FloatType> &linear,
            const Quadratic<IndexType, FloatType> &quadratic,
            const FloatType &offset
        )
        {
            Linear<IndexType, FloatType> h;
            Quadratic<IndexType, FloatType> J;
            FloatType new_offset = offset;
            FloatType linear_offset = 0.0;
            FloatType quadratic_offset = 0.0;
    
            for(auto &it : linear)
            {
                insert_or_assign(h, it.first, static_cast<FloatType>(0.5 * it.second));
                linear_offset += it.second;
            }
    
            for(auto &it : quadratic)
            {
                insert_or_assign(J, it.first, static_cast<FloatType>(0.25 * it.second));
                h[it.first.first] += 0.25 * it.second;
                h[it.first.second] += 0.25 * it.second;
                quadratic_offset += it.second;
            }
    
            new_offset += 0.5 * linear_offset + 0.25 * quadratic_offset;
    
            return std::make_tuple(h, J, new_offset);
        }
    
        /* Methods */
    
        /**
         * @brief Determine the energy of the specified sample of a binary quadratic model.
         * 
         * @param sample
         * @return An energy with respect to the sample.
         */
        FloatType energy(const Sample<IndexType> &sample) const
        {
            FloatType en = m_offset;
            for(auto &&it : m_linear)
            {
                if(check_vartype(sample.at(it.first), m_vartype))
                {
                    en += static_cast<FloatType>(sample.at(it.first)) * it.second;
                }
            }
            for(auto &it : m_quadratic)
            {
                if(check_vartype(sample.at(it.first.first), m_vartype)&&check_vartype(sample.at(it.first.second), m_vartype))
                {
                    en += static_cast<FloatType>(sample.at(it.first.first)) * static_cast<FloatType>(sample.at(it.first.second)) * it.second;
                }
            }
            return en;
        }
        
        /**
         * @brief Determine the energies of the given samples.
         * 
         * @param samples_like
         * @return A vector including energies with respect to the samples.
         */
        std::vector<FloatType> energies(const std::vector<Sample<IndexType>> &samples_like) const
        {
            std::vector<FloatType> en_vec;
            for(auto &it : samples_like)
            {
                en_vec.push_back(energy(it));
            }
            return en_vec;
        }
        
        /* Conversions */
        /**
         * @brief Convert a binary quadratic model to QUBO format.
         * 
         * @return A tuple including a quadratic bias and an offset.
         */
        std::tuple<Quadratic<IndexType, FloatType>, FloatType> to_qubo()
        {
            // change vartype to binary
            BinaryQuadraticModel bqm = change_vartype(Vartype::BINARY, false);
    
            Linear<IndexType, FloatType> linear = bqm.get_linear();
            Quadratic<IndexType, FloatType> Q = bqm.get_quadratic();
            FloatType offset = bqm.get_offset();
            for(auto &it : linear)
            {
                insert_or_assign(Q, std::make_pair(it.first, it.first), it.second);
            }
            return std::make_tuple(Q, offset);
        }
    
        /**
         * @brief Create a binary quadratic model from a QUBO model.
         *
         * @param Q
         * @param offset
         *
         * @return Binary quadratic model with vartype set to `.Vartype.BINARY`.
         */
        static BinaryQuadraticModel from_qubo(const Quadratic<IndexType, FloatType>& Q, FloatType offset=0.0)
        {
            Linear<IndexType, FloatType> linear;
            Quadratic<IndexType, FloatType> quadratic;
    
            for(auto&& elem : Q){
                const auto& key = elem.first;
                const auto& value = elem.second;
                if(key.first == key.second){
                    linear[key.first] = value;
                }
                else{
                    quadratic[std::make_pair(key.first, key.second)] = value;
                }
            }
    
            return BinaryQuadraticModel<IndexType, FloatType, DataType>(linear, quadratic, offset, Vartype::BINARY);
        }
    
        /**
         * @brief Convert a binary quadratic model to Ising format.
         * 
         * @return A tuple including a linear bias, a quadratic bias and an offset.
         */
        std::tuple<Linear<IndexType, FloatType>, Quadratic<IndexType, FloatType>, FloatType> to_ising()
        {
            // change vartype to spin
            BinaryQuadraticModel bqm = change_vartype(Vartype::SPIN, false);
    
            Linear<IndexType, FloatType> linear = bqm.get_linear();
            Quadratic<IndexType, FloatType> quadratic = bqm.get_quadratic();
            FloatType offset = bqm.get_offset();
            return std::make_tuple(linear, quadratic, offset);
        }
    
        /**
         * @brief Create a binary quadratic model from an Ising problem.
         *
         * @param linear
         * @param quadratic
         * @param offset
         *
         * @return Binary quadratic model with vartype set to `.Vartype.SPIN`.
         */
        static BinaryQuadraticModel from_ising(const Linear<IndexType, FloatType>& linear, const Quadratic<IndexType, FloatType>& quadratic, FloatType offset=0.0)
        {
            return BinaryQuadraticModel<IndexType, FloatType, DataType>(linear, quadratic, offset, Vartype::SPIN);
        }
    
    
        /**
         * @brief generate interaction matrix with given list of indices
         * The generated matrix will be the following symmetric matrix:
         * \f[
         * \begin{pmatrix}
         * \tilde{J}_{0,0} & \tilde{J}_{0,1} & \tilde{J}_{0,2} & \cdots \\
         * \tilde{J}_{1,0} & \tilde{J}_{1,1} & \tilde{J}_{1,2} & \cdots \\
         * \tilde{J}_{2,0} & \tilde{J}_{2,1} & \tilde{J}_{2,2} & \cdots \\
         * \vdots & \vdots & \vdots & \ddots \\
         * \end{pmatrix}
         * \f]
         *
         * where in the Ising case,
         * \f[
         * \tilde{J}_{f(i),f(j)} = J_{ij} + J_{ji}, \\
         * \tilde{J}_{f(i),f(i)} = h_{i},
         * \f]
         * and the QUBO case,
         * \f[
         * \tilde{J}_{f(i),f(j)} = Q_{ij} + Q_{ji}, \\
         * \tilde{J}_{f(i),f(i)} = Q_{ii},
         * \f]
         * and the function \f$f\f$ denotes a mapping from index to the corresponding number specified by the argument `indices`.
         * For instance, if `indices` is ['a', 'b', 'c'], The following equations, \f$f(a) = 0, f(b)=1, \mathrm{and} f(c)=2\f$ hold.
         *
         * The original Hamiltonian can be rewritten with \f$\tilde{J_{ij}}\f$ as
         * \f[
         * E_{\mathrm{Ising}} = \sum_{i} \tilde{J}_{f(i),f(i)} s_i + \sum_{i < j} \tilde{J}_{f(i), f(j)} s_i s_j + \delta_{\mathrm{Ising}},
         * \f]
         * and
         * \f[
         * E_{\mathrm{QUBO}} = \sum_{i} \tilde{J}_{f(i), f(i)}x_i + \sum_{i < j} \tilde{J}_{f(i), f(j)} x_i x_j + \delta_{\mathrm{QUBO}}.
         * \f]
         * @deprecated use interaction_matrix()
         *
         * @param indices
         *
         * @return corresponding interaction matrix (Eigen)
         */
        Matrix interaction_matrix(const std::vector<IndexType>& indices) const {
            // generate matrix
            size_t system_size = indices.size();
            Matrix _interaction_matrix = Matrix::Zero(system_size, system_size);
            const Linear<IndexType, FloatType>& linear = m_linear; 
            const Quadratic<IndexType, FloatType>& quadratic = m_quadratic; 
    
            for(size_t i=0; i<indices.size(); i++){
                const IndexType& i_index = indices[i];
                _interaction_matrix(i, i) = (linear.find(i_index) != linear.end()) ? linear.at(i_index): 0;
                for(size_t j=i+1; j<indices.size(); j++){
                    const IndexType& j_index = indices[j];
                    FloatType jval = 0.0;
    
                    if(quadratic.find(std::make_pair(i_index, j_index)) != quadratic.end()){
                        jval += quadratic.at(std::make_pair(i_index, j_index));
                    }
                    if(quadratic.find(std::make_pair(j_index, i_index)) != quadratic.end()){
                        jval += quadratic.at(std::make_pair(j_index, i_index));
                    }
    
                    _interaction_matrix(i, j) = jval;
                    _interaction_matrix(j, i) = jval;
                }
            }
    
            return _interaction_matrix;
        }

        /**
         * @brief generate interaction matrix with given list of indices
         * The generated matrix will be the following symmetric matrix:
         * \f[
         * \begin{pmatrix}
         * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
         * 0 & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
         * \vdots & \vdots & \vdots & \vdots & \vdots \\
         * 0 & 0 & \cdots & J_{N-1,N-1} & h_{N-1}\\
         * 0 & 0 & \cdots & 0 & 1 \\
         * \end{pmatrix}
         * \f]
         *
         */
        Matrix interaction_matrix() const {
            // generate matrix
            auto indices = this->get_variables();
            size_t system_size = this->get_num_variables();
            Matrix _interaction_matrix = Matrix::Zero(system_size+1, system_size+1);
            const Linear<IndexType, FloatType>& linear = m_linear; 
            const Quadratic<IndexType, FloatType>& quadratic = m_quadratic; 
    
            for(size_t i=0; i<indices.size(); i++){
                const IndexType& i_index = indices[i];
                _interaction_matrix(i, system_size) = (linear.find(i_index) != linear.end()) ? linear.at(i_index): 0;
                for(size_t j=i+1; j<indices.size(); j++){
                    const IndexType& j_index = indices[j];
                    FloatType jval = 0.0;
    
                    if(quadratic.find(std::make_pair(i_index, j_index)) != quadratic.end()){
                        jval += quadratic.at(std::make_pair(i_index, j_index));
                    }
                    if(quadratic.find(std::make_pair(j_index, i_index)) != quadratic.end()){
                        jval += quadratic.at(std::make_pair(j_index, i_index));
                    }
    
                    _interaction_matrix(i, j) = jval;
                }
            }
    
            return _interaction_matrix;
        }
    
    
        using json = nlohmann::json;
    
        /**
         * @brief Convert the binary quadratic model to a serializable object
         * user_bytes is assume to be set to False
         *
         * @return An object that can be serialized (nlohmann::json)
         */
        json to_serializable() const
        {
            std::string schema_version = "3.0.0";
            //all variables are contained in the keys of m_linear.
            //All we need is to traverse the keys of m_linear.
            /*
             * output sample
             * >>> bqm = dimod.BinaryQuadraticModel({'c': -1, 'd': 1}, {('a', 'd'): 2, ('b', 'e'): 5, ('a', 'c'): 3}, 0.0, dimod.BINARY)
             *
             * >>> bqm.to_serializable()
             * {'type': 'BinaryQuadraticModel', 'version': {'bqm_schema': '3.0.0'}, 'use_bytes': False, 'index_type': 'uint16', 'bias_type': 'float32', 'num_variables': 5, 'num_interactions': 3, 'variable_labels': ['a', 'b', 'c', 'd', 'e'], 'variable_type': 'BINARY', 'offset': 0.0, 'info': {}, 'linear_biases': [0.0, 0.0, -1.0, 1.0, 0.0], 'quadratic_biases': [3.0, 2.0, 5.0], 'quadratic_head': [0, 0, 1], 'quadratic_tail': [2, 3, 4]}
             */
    
            //set variables (sorted)
            std::vector<IndexType> variables;
            variables.reserve(m_linear.size());
            for(auto&& elem : m_linear)
            {
                variables.push_back(elem.first);
            }
    
            std::sort(variables.begin(), variables.end());
    
            size_t num_variables = variables.size();
            
            //set sorted linear biases
            std::vector<FloatType> l_bias;
            for(auto&& key : variables)
            {
                l_bias.push_back(m_linear.at(key));
            }
    
            //set quadratic head, tail and biases
            std::vector<size_t> q_head, q_tail;
            std::vector<FloatType> q_bias;
            for(auto&& elem : m_quadratic)
            {
                auto it_head = std::find(variables.begin(), variables.end(), elem.first.first);
                auto it_tail = std::find(variables.begin(), variables.end(), elem.first.second);
                size_t idx_head = std::distance(variables.begin(), it_head);
                size_t idx_tail = std::distance(variables.begin(), it_tail);
                q_head.push_back(idx_head);
                q_tail.push_back(idx_tail);
                q_bias.push_back(elem.second);
            }
    
            size_t num_interactions =  m_quadratic.size();
    
            //set index_dtype
            std::string index_dtype = num_variables <= 65536UL ? "uint16" : "uint32";
    
            //set bias_type
            std::string bias_type;
            if(typeid(m_offset) == typeid(float))
            {
                bias_type = "float32";
            }
            else if(typeid(m_offset) == typeid(double))
            {
                bias_type = "float64";
            }
            else
            {
                throw std::runtime_error("FloatType must be float or double.");
            }
    
            //set variable type
            std::string variable_type;
            if(m_vartype == Vartype::SPIN)
            {
                variable_type = "SPIN";
            }
            else if(m_vartype == Vartype::BINARY)
            {
                variable_type = "BINARY";
            }
            else
            {
                throw std::runtime_error("Variable type must be SPIN or BINARY.");
            }
    
            json output;
            output["type"] = "BinaryQuadraticModel";
            output["version"] = {{"bqm_schema", "3.0.0"}};
            output["variable_labels"] = variables;
            output["use_bytes"] = false;
            output["index_type"] = index_dtype;
            output["bias_type"] = bias_type;
            output["num_variables"] = num_variables;
            output["num_interactions"] = num_interactions;
            output["variable_labels"] = variables;
            output["variable_type"] = variable_type;
            output["offset"] = m_offset;
            output["info"] = "";
            output["linear_biases"] = l_bias;
            output["quadratic_biases"] = q_bias;
            output["quadratic_head"] = q_head;
            output["quadratic_tail"] = q_tail;
    
            return output;
        }
    
        /**
         * @brief Create a BinaryQuadraticModel instance from a serializable object.
         * 
         * @tparam IndexType_serial
         * @tparam FloatType_serial
         * @param input
         * @return BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> 
         */
        template <typename IndexType_serial = IndexType, typename FloatType_serial = FloatType>
        static BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> from_serializable(const json &input)
        {
            //extract type and version
            std::string type = input["type"];
            if(type != "BinaryQuadraticModel")
            {
                throw std::runtime_error("Type must be \"BinaryQuadraticModel\".\n");
            }
            std::string version = input["version"]["bqm_schema"];
            if(version != "3.0.0")
            {
                throw std::runtime_error("bqm_schema must be 3.0.0.\n");
            }
    
            //extract variable_type
            Vartype vartype;
            std::string variable_type = input["variable_type"];
            if(variable_type == "SPIN")
            {
                vartype = Vartype::SPIN;
            }
            else if(variable_type == "BINARY")
            {
                vartype = Vartype::BINARY;
            }
            else
            {
                throw std::runtime_error("variable_type must be SPIN or BINARY.");
            }
    
            //extract linear biases
            Linear<IndexType_serial, FloatType_serial> linear;
            std::vector<IndexType_serial> variables = input["variable_labels"];
            std::vector<FloatType_serial> l_bias = input["linear_biases"];
            for(size_t i = 0; i < variables.size(); ++i)
            {
                insert_or_assign(linear, variables[i], l_bias[i]);
            }
    
            //extract quadratic biases
            Quadratic<IndexType_serial, FloatType_serial> quadratic;
            std::vector<size_t> q_head = input["quadratic_head"];
            std::vector<size_t> q_tail = input["quadratic_tail"];
            std::vector<FloatType_serial> q_bias = input["quadratic_biases"];
            for(size_t i = 0; i < q_head.size(); ++i)
            {
                insert_or_assign(quadratic, std::make_pair(variables[q_head[i]], variables[q_tail[i]]), q_bias[i]);
            }
    
            //extract offset and info
            FloatType_serial offset = input["offset"];
    
            //construct a BinaryQuadraticModel instance
            BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> bqm(linear, quadratic, offset, vartype);
            return bqm;
        }
    
    };
}
#endif
