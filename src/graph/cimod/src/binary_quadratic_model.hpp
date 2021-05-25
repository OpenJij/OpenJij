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
 * @mainpage cimod
 * 
 * @section s_overview Overview
 * cimod is a C++ library for a binary quadratic model.
 * This library provides a binary quadratic model class which contains an Ising model or a quadratic unconstrained binary optimization (QUBO) model.
 * It also provides utilities for constructing a model and transforming to some other interfaces.
 * This library is created based on dimod (https://github.com/dwavesystems/dimod).
 * 
 * @section s_bqm Binary quadratic model
 * A binary quadratic model class can contain an Ising model or a QUBO model.
 * 
 * @subsection ss_ising Ising model
 * An energy of an Ising model \f$E_{\mathrm{Ising}}\f$ is represented by
 * \f[
 * E_{\mathrm{Ising}} = \sum_{i} h_i s_i + \sum_{i \neq j} J_{ij} s_i s_j + \delta_{\mathrm{Ising}},
 * \f]
 * where \f$s_i \in \{+1, -1\}\f$ denotes a spin at the site \f$i\f$, \f$h_i\f$ denotes an external magnetic field parameter, \f$J_{ij}\f$ denotes an interaction parameter and \f$\delta_{\mathrm{Ising}}\f$ denotes an offset.
 * Note that this library assumes that the interaction is not symmetric, i.e., \f$J_{ij} \neq J_{ji}\f$.
 * 
 * @subsection ss_qubo QUBO model
 * An evaluation function of a QUBO model \f$E_{\mathrm{QUBO}}\f$ is represented by
 * \f[
 * E_{\mathrm{QUBO}} = \sum_{i, j} Q_{ij} x_i x_j + \delta_{\mathrm{QUBO}},
 * \f]
 * where \f$x_i \in \{0, 1\}\f$ denotes a decision variable, \f$Q_{ij}\f$ denotes a quadratic bias and \f$\delta_{\mathrm{QUBO}}\f$ denotes an offset.
 * Note that this library assumes that the quadratic bias is not symmetric, i.e., \f$Q_{ij} \neq Q_{ji}\f$ if \f$i \neq j\f$.
 * 
 * @section s_example Example
 * @code
 * #include "src/binary_quadratic_model.hpp"
 * 
 * using namespace cimod;
 * int main()
 * {
 * // Set linear biases and quadratic biases
 * Linear<uint32_t, double> linear{ {1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0} };
 * Quadratic<uint32_t, double> quadratic
 * {
 *      {std::make_pair(1, 2), 12.0}, {std::make_pair(1, 3), 13.0}, {std::make_pair(1, 4), 14.0},
 *      {std::make_pair(2, 3), 23.0}, {std::make_pair(2, 4), 24.0},
 *      {std::make_pair(3, 4), 34.0}
 *  };
 * 
 * // Set offset
 * double offset = 0.0;
 * 
 * // Set variable type
 * Vartype vartype = Vartype::BINARY;
 * // Create a BinaryQuadraticModel instance
 * BinaryQuadraticModel<uint32_t, double> bqm(linear, quadratic, offset, vartype);
 * 
 * // Print informations of bqm
 * bqm.print();
 * 
 * return 0;
 * }
 * @endcode
 */

/**
 * @file binary_quadratic_model.hpp
 * @author Kohji Nishimura
 * @brief Dense BinaryQuadraticModel class
 * @version 1.0.0
 * @date 2020-03-24
 * 
 * @copyright Copyright (c) Jij Inc. 2020
 * 
 */

#ifndef BINARY_QUADRATIC_MODEL_HPP__
#define BINARY_QUADRATIC_MODEL_HPP__

#include "disable_eigen_warning.hpp"

#include "vartypes.hpp"
#include "hash.hpp"
#include "utilities.hpp"
#include "json.hpp"

#include <algorithm>
#include <type_traits>
#include <limits>
#include <cstdint>
#include <cassert>
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
#include <Eigen/Sparse>

namespace cimod
{
    /**
     * @brief Type alias for linear bias
     * 
     * @tparam IndexType 
     */
    template <typename IndexType, typename FloatType>
    using Linear = std::unordered_map<IndexType, FloatType>;
    
    /**
     * @brief Type alias for quadratic bias
     * 
     * @tparam IndexType 
     */
    template <typename IndexType, typename FloatType>
    using Quadratic = std::unordered_map<std::pair<IndexType, IndexType>, FloatType, pair_hash>;
    
    /**
     * @brief Type alias for adjacency list
     * 
     * @tparam IndexType 
     */
    template <typename IndexType, typename FloatType>
    using Adjacency = std::unordered_map<IndexType, std::unordered_map<IndexType, FloatType>>;
    
    /**
     * @brief Type alias for sample
     * 
     * @tparam IndexType 
     */
    template <typename IndexType>
    using Sample = std::unordered_map<IndexType, int32_t>;


    struct Dense{};
    struct Sparse{};

    /**
     * @brief Class for dense binary quadratic model.
     * @tparam IndexType index type. type must be hashable and comparable.
     * @tparam FloatType 
     * @tparam IndexType 
     */
    
    template <typename IndexType, typename FloatType, typename DataType>
    class BinaryQuadraticModel
    {
    private:

        /**
         * @brief template type for dispatch
         * used for SFINAE
         *
         * @tparam T
         * @tparam U
         */
        template<typename T, typename U>
        using dispatch_t = std::enable_if_t<std::is_same_v<T, U>, std::nullptr_t>;
    public:
    /**
     * @brief Eigen Matrix
     * if DataType is Dense , Matrix is equal to Eigen::Matrix
     * if DataType is Sparse, Matrix is equal to Eigen::SparseMatrix
     */

    using DenseMatrix   = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using SparseMatrix  = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
    using SpIter        = typename SparseMatrix::InnerIterator;

    using Matrix = std::conditional_t<
            std::is_same_v<DataType, Dense>,
            DenseMatrix,
            SparseMatrix
          >;

    using Vector = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;
    
    protected:
        
        /**
         * @brief quadratic dense-type matrix
         * The stored matrix has the following triangular form:
         *
         * \f[
         * \begin{pmatrix}
         * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
         * 0 & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
         * \vdots & \vdots & \vdots & \vdots & \vdots \\
         * 0 & 0 & \cdots & J_{N-1,N-1} & h_{N-1}\\
         * 0 & 0 & \cdots & 0 & 1 \\
         * \end{pmatrix}
         * \f]
         */
        Matrix _quadmat;
    
        /**
         * @brief vector for converting index to label
         * the list is asssumed to be sorted
         */
        std::vector<IndexType> _idx_to_label;
    
        /**
         * @brief dict for converting label to index
         */
        std::unordered_map<IndexType, size_t> _label_to_idx;
    
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
    
        /**
         * @brief set _label_to_idx from _idx_to_label
         */
        inline void _set_label_to_idx(){
            //reset
            _label_to_idx.clear();
            //initialize
            for(size_t i=0; i<_idx_to_label.size(); i++){
                _label_to_idx[_idx_to_label[i]] = i;
            }
        }

        /**
         * @brief access elements for dense matrix
         *
         * @tparam T
         * @param i
         * @param j
         * @param dispatch_t
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType& _quadmat_get(size_t i, size_t j, dispatch_t<T, Dense> = nullptr){
            return _quadmat(i, j);
        }

        /**
         * @brief access elements for dense matrix
         *
         * @tparam T
         * @param i
         * @param j
         * @param dispatch_t
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _quadmat_get(size_t i, size_t j, dispatch_t<T, Dense> = nullptr) const{
            return _quadmat(i, j);
        }

        /**
         * @brief access elements for sparse matrix
         *
         * @tparam T
         * @param i
         * @param j
         * @param dispatch_t
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType& _quadmat_get(size_t i, size_t j, dispatch_t<T, Sparse> = nullptr){
            return _quadmat.coeffRef(i, j);
        }

        /**
         * @brief access elements for sparse matrix
         *
         * @tparam T
         * @param i
         * @param j
         * @param dispatch_t
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _quadmat_get(size_t i, size_t j, dispatch_t<T, Sparse> = nullptr) const{
            return _quadmat.coeff(i, j);
        }

    
        /**
         * @brief get reference of _quadmat(i,j)
         *
         * @param label_i
         * @param label_j
         *
         * @return reference of _quadmat(i,j)
         */
        inline FloatType& _mat(IndexType label_i, IndexType label_j){
            size_t i = _label_to_idx.at(label_i);
            size_t j = _label_to_idx.at(label_j);
            if(i != j)
                return _quadmat_get(std::min(i, j), std::max(i, j));
            else
                throw std::runtime_error("No self-loop (mat(i,i)) allowed");
        }
    
        /**
         * @brief get reference of _quadmat(i,i)
         *
         * @param label_i
         *
         * @return reference of _quadmat(i,i)
         */
        inline FloatType& _mat(IndexType label_i){
            size_t i = _label_to_idx.at(label_i);
            return _quadmat_get(i, _quadmat.rows()-1);
        }
    
        /**
         * @brief get reference of _quadmat(i,j)
         *
         * @param label_i
         * @param label_j
         *
         * @return reference of _quadmat(i,j)
         */
        inline FloatType _mat(IndexType label_i, IndexType label_j) const{
            size_t i = _label_to_idx.at(label_i);
            size_t j = _label_to_idx.at(label_j);
    
            if(i != j)
                return _quadmat_get(std::min(i, j), std::max(i, j));
            else
                throw std::runtime_error("No self-loop (mat(i,i)) allowed");
        }
    
        /**
         * @brief get reference of _quadmat(i,i)
         *
         * @param label_i
         *
         * @return reference of _quadmat(i,i)
         */
        inline FloatType _mat(IndexType label_i) const{
            size_t i = _label_to_idx.at(label_i);
            return _quadmat_get(i, _quadmat.rows()-1);
        }
    
        /**
         * @brief calculate maximum element in linear term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _max_linear(dispatch_t<T, Dense> = nullptr) const{
            size_t N = _quadmat.rows();
            return _quadmat.block(0,N-1,N-1,1).maxCoeff();
        }
    
        /**
         * @brief calculate maximum element in quadratic term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _max_quadratic(dispatch_t<T, Dense> = nullptr) const{
            size_t N = _quadmat.rows();
            return _quadmat.block(0,0,N-1,N-1).maxCoeff();
        }
    
        /**
         * @brief calculate minimum element in linear term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _min_linear(dispatch_t<T, Dense> = nullptr) const{
            size_t N = _quadmat.rows();
            return _quadmat.block(0,N-1,N-1,1).minCoeff();
        }
    
        /**
         * @brief calculate minimum element in quadratic term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _min_quadratic(dispatch_t<T, Dense> = nullptr) const{
            size_t N = _quadmat.rows();
            return _quadmat.block(0,0,N-1,N-1).minCoeff();
        }

        /**
         * @brief calculate maximum element in linear term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _max_linear(dispatch_t<T, Sparse> = nullptr) const{
            size_t N = _quadmat.rows();
            SparseMatrix mat(N-1,1);
            mat = _quadmat.block(0,N-1,N-1,1);
            return mat.coeffs().maxCoeff();
        }
    
        /**
         * @brief calculate maximum element in quadratic term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _max_quadratic(dispatch_t<T, Sparse> = nullptr) const{
            size_t N = _quadmat.rows();
            SparseMatrix mat(N-1,N-1);
            mat = _quadmat.block(0,0,N-1,N-1);
            return mat.coeffs().maxCoeff();
        }
    
        /**
         * @brief calculate minimum element in linear term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _min_linear(dispatch_t<T, Sparse> = nullptr) const{
            size_t N = _quadmat.rows();
            SparseMatrix mat(N-1,1);
            mat = _quadmat.block(0,N-1,N-1,1);
            return mat.coeffs().minCoeff();
        }
    
        /**
         * @brief calculate minimum element in quadratic term for dense graph
         *
         * @return 
         */
        template<typename T=DataType>
        inline FloatType _min_quadratic(dispatch_t<T, Sparse> = nullptr) const{
            size_t N = _quadmat.rows();
            SparseMatrix mat(N-1,N-1);
            mat = _quadmat.block(0,0,N-1,N-1);
            return mat.coeffs().minCoeff();
        }
    
        /**
         * @brief insert row and column that corresponds to added label into _quadmat for dense matrix
         *
         * @param label_i
         */
        template<typename T=DataType>
        inline void _insert_label_into_mat(IndexType label_i, dispatch_t<T, Dense> = nullptr){
            size_t i = _label_to_idx.at(label_i);
            //define temp mat
            size_t N = _quadmat.rows()+1;
            Matrix tempmat = Matrix(N, N);
            tempmat.setZero();
            //copy elements to new matrix
            tempmat.block(0,0,i,i)              = _quadmat.block(0,0,i,i);
            tempmat.block(0,i+1,i,N-i-1)        = _quadmat.block(0,i,i,N-i-1);
            tempmat.block(i+1,i+1,N-i-1,N-i-1)  = _quadmat.block(i,i,N-i-1,N-i-1);
    
            _quadmat = tempmat;
        }

        /**
         * @brief insert row and column that corresponds to added label into _quadmat for sparse matrix
         *
         * @param label_i
         */
        template<typename T=DataType>
        inline void _insert_label_into_mat(IndexType label_i, dispatch_t<T, Sparse> = nullptr){
            size_t i = _label_to_idx.at(label_i);
            //define temp mat
            size_t N = _quadmat.rows()+1;
            //Matrix tempmat = Matrix(N, N);
            //copy elements to new matrix
            //tempmat.block(0,0,i,i)              = _quadmat.block(0,0,i,i);
            //tempmat.block(0,i+1,i,N-i-1)        = _quadmat.block(0,i,i,N-i-1);
            //tempmat.block(i+1,i+1,N-i-1,N-i-1)  = _quadmat.block(i,i,N-i-1,N-i-1);

            std::vector<Eigen::Triplet<FloatType>> triplets;
            triplets.reserve(_quadmat.nonZeros());

            for(int k=0; k<_quadmat.outerSize(); k++){
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if(r >= i && c >= i){
                        triplets.emplace_back(r+1, c+1, val);
                    }
                    else if(r >= i){
                        triplets.emplace_back(r+1, c  , val);
                    }
                    else if(c >= i){
                        triplets.emplace_back(r  , c+1, val);
                    }
                    else{
                        triplets.emplace_back(r  , c  , val);
                    }
                }
            }

            _quadmat.resize(N, N);
            _quadmat.setFromTriplets(triplets.begin(), triplets.end());
        }
    
        /**
         * @brief delete row and column that corresponds to existing label from _quadmat for dense matrix
         *
         * @param label_i
         */
        template<typename T=DataType>
        inline void _delete_label_from_mat(IndexType label_i, dispatch_t<T, Dense> = nullptr){
            size_t i = _label_to_idx.at(label_i);
            //define temp mat
            size_t N = _quadmat.rows();
            Matrix tempmat = Matrix(N-1, N-1);
            tempmat.setZero();
            //copy elements to new matrix
            tempmat.block(0,0,i,i)              = _quadmat.block(0,0,i,i);
            tempmat.block(0,i,i,N-i-1)          = _quadmat.block(0,i+1,i,N-i-1);
            tempmat.block(i,i,N-i-1,N-i-1)      = _quadmat.block(i+1,i+1,N-i-1,N-i-1);
    
            _quadmat = tempmat;
        }

        /**
         * @brief delete row and column that corresponds to existing label from _quadmat for sparse matrix
         *
         * @param label_i
         */
        template<typename T=DataType>
        inline void _delete_label_from_mat(IndexType label_i, dispatch_t<T, Sparse> = nullptr){
            size_t i = _label_to_idx.at(label_i);
            //define temp mat
            size_t N = _quadmat.rows();
            //copy elements to new matrix
            //tempmat.block(0,0,i,i)              = _quadmat.block(0,0,i,i);
            //tempmat.block(0,i,i,N-i-1)          = _quadmat.block(0,i+1,i,N-i-1);
            //tempmat.block(i,i,N-i-1,N-i-1)      = _quadmat.block(i+1,i+1,N-i-1,N-i-1);
            
            std::vector<Eigen::Triplet<FloatType>> triplets;
            triplets.reserve(_quadmat.nonZeros());

            for(int k=0; k<_quadmat.outerSize(); k++){
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if(r == i || c == i)
                        continue;

                    if(r > i && c > i){
                        triplets.emplace_back(r-1, c-1, val);
                    }
                    else if(r > i){
                        triplets.emplace_back(r-1, c  , val);
                    }
                    else if(c > i){
                        triplets.emplace_back(r  , c-1, val);
                    }
                    else{
                        triplets.emplace_back(r  , c  , val);
                    }
                }
            }

            _quadmat.resize(N-1, N-1);
            _quadmat.setFromTriplets(triplets.begin(), triplets.end());
        }
    
        /**
         * @brief add new label
         * if label_i already exists, this process is skipped.
         *
         * @param label_i
         */
        inline void _add_new_label(IndexType label_i){
            if(_label_to_idx.find(label_i) == _label_to_idx.end()){
                //add label_i
                _idx_to_label.push_back(label_i);
                std::sort(_idx_to_label.begin(), _idx_to_label.end());
                _set_label_to_idx();
    
                _insert_label_into_mat(label_i);
            }
        }
    
        /**
         * @brief delete label
         * if label_i does not exist, this process is skipped.
         *
         * @param label_i
         * @param force_delete if true, delete label whenever there exists corresponding nonzero elements in the matrix.
         * otherwise, delete label only if there are no corresponding nonzero elements in the matrix.
         */
        inline void _delete_label(IndexType label_i, bool force_delete = true){
            auto position = std::find(_idx_to_label.begin(), _idx_to_label.end(), label_i);
            if(position != _idx_to_label.end()){
                if(force_delete == false){
                    //check if there are corresponding nonzero elements
                    size_t i = std::distance(_idx_to_label.begin(), position);
                    if(_quadmat.col(i).squaredNorm() > std::numeric_limits<FloatType>::epsilon() ||
                       _quadmat.row(i).squaredNorm() > std::numeric_limits<FloatType>::epsilon() ){
                        // exists nonzero elements
                        return;
                    }
                }
                //delete from matrix first
                _delete_label_from_mat(label_i);
                //add label_i
                _idx_to_label.erase(position);
                // already sorted
                //std::sort(_idx_to_label.begin(), _idx_to_label.end());
                _set_label_to_idx();
            }
        }
    
        /**
         * @brief initialize matrix with linear and quadratic dicts (for dense matrix)
         *
         * @param linear
         * @param quadratic
         */
        template<typename T=DataType>
        inline void _initialize_quadmat(const Linear<IndexType, FloatType> &linear, const Quadratic<IndexType, FloatType> &quadratic,
                dispatch_t<T, Dense> = nullptr){
            //gather labels
            std::unordered_set<IndexType> labels;
            
            for(const auto& kv : linear){
                labels.insert(kv.first);
            }
    
            for(const auto& kv : quadratic){
                labels.insert(kv.first.first);
                labels.insert(kv.first.second);
            }
    
            // init label <-> index conversion variables
            _idx_to_label = std::vector<IndexType>(labels.begin(), labels.end());
            std::sort(_idx_to_label.begin(), _idx_to_label.end());
            _set_label_to_idx();
    
            //initialize _quadmat
            size_t mat_size = _idx_to_label.size() + 1;
            _quadmat = Matrix(mat_size, mat_size);
            _quadmat.fill(0);
            _quadmat_get(mat_size-1, mat_size-1) = 1;
    
            //copy linear and quadratic to _quadmat
            for(const auto& kv : linear){
                IndexType key = kv.first;
                FloatType val = kv.second;
                _mat(key) += val;
            }
    
            for(const auto& kv : quadratic){
                std::pair<IndexType, IndexType> key = kv.first;
                FloatType val = kv.second;
                _mat(key.first, key.second) += val;
            }
        }

        /**
         * @brief initialize matrix with linear and quadratic dicts (for sparse matrix)
         *
         * @param linear
         * @param quadratic
         */
        template<typename T=DataType>
        inline void _initialize_quadmat(const Linear<IndexType, FloatType> &linear, const Quadratic<IndexType, FloatType> &quadratic,
                dispatch_t<T, Sparse> = nullptr){
            //gather labels
            std::unordered_set<IndexType> labels;
            
            for(const auto& kv : linear){
                labels.insert(kv.first);
            }
    
            for(const auto& kv : quadratic){
                labels.insert(kv.first.first);
                labels.insert(kv.first.second);
            }
    
            // init label <-> index conversion variables
            _idx_to_label = std::vector<IndexType>(labels.begin(), labels.end());
            std::sort(_idx_to_label.begin(), _idx_to_label.end());
            _set_label_to_idx();
    
            //initialize _quadmat
            size_t mat_size = _idx_to_label.size() + 1;
            _quadmat = Matrix(mat_size, mat_size);

            std::vector<Eigen::Triplet<FloatType>> triplets;
            const size_t buffer = 5;
            triplets.reserve(linear.size() + quadratic.size() + buffer);
    
            //set triplets to _quadmat
            for(const auto& kv : linear){
                size_t idx1 = _label_to_idx.at(kv.first);
                size_t idx2 = mat_size-1;
                FloatType val = kv.second;

                //NOTE: duplicated elements are summed up.
                triplets.emplace_back(std::min(idx1, idx2), std::max(idx1, idx2), val);
            }
    
            for(const auto& kv : quadratic){
                size_t idx1 = _label_to_idx.at(kv.first.first);
                size_t idx2 = _label_to_idx.at(kv.first.second);
                FloatType val = kv.second;

                //NOTE: duplicated elements are summed up.
                triplets.emplace_back(std::min(idx1, idx2), std::max(idx1, idx2), val);
            }

            triplets.emplace_back(mat_size-1, mat_size-1, 1);

            _quadmat.setFromTriplets(triplets.begin(), triplets.end());
        }

        /**
         * @brief add non-diagonal elements to upper triangular components for dense matrix
         *
         * @tparam T
         * @param mat
         * @param fix_format
         */
        template<typename T=DataType>
        inline void _add_triangular_elements(const DenseMatrix& mat, bool fix_format, dispatch_t<T, Dense> = nullptr){

            size_t mat_size = _idx_to_label.size() + 1;

            if((size_t)mat.rows() == _idx_to_label.size() + 1){
                if(fix_format == false){
                    _quadmat = mat;
                }
                else{
                    _quadmat += mat.template triangularView<Eigen::StrictlyUpper>();
                    _quadmat += mat.template triangularView<Eigen::StrictlyLower>().transpose();
                }
            }
            else if((size_t)mat.rows() == _idx_to_label.size()){
                _quadmat.block(0,0,mat_size-1,mat_size-1) += mat.template triangularView<Eigen::StrictlyUpper>();
                _quadmat.block(0,0,mat_size-1,mat_size-1) += mat.template triangularView<Eigen::StrictlyLower>().transpose();
                //local fields
                Vector loc = mat.diagonal();
                _quadmat.block(0,mat_size-1,mat_size-1,1) += loc;
            }
            else{
                throw std::runtime_error("the number of variables and dimension do not match.");
            }
        }

        /**
         * @brief add non-diagonal elements to upper triangular components for sparse matrix
         *
         * @tparam T
         * @param mat
         * @param fix_format
         */
        template<typename T=DataType>
        inline void _add_triangular_elements(const DenseMatrix& mat, bool fix_format, dispatch_t<T, Sparse> = nullptr){

            //generate sparse matrix
            SparseMatrix sparse_mat;

            size_t mat_size = _idx_to_label.size() + 1;

            if((size_t)mat.rows() == _idx_to_label.size() + 1){
                if(fix_format == false){
                    _quadmat = mat.sparseView();
                }
                else{
                    sparse_mat = mat.sparseView();
                    _quadmat += sparse_mat.template triangularView<Eigen::StrictlyUpper>();
                    sparse_mat = mat.sparseView().transpose();
                    _quadmat += sparse_mat.template triangularView<Eigen::StrictlyUpper>();
                }
            }
            else if((size_t)mat.rows() == _idx_to_label.size()){
                //generate Dense Matrix
                DenseMatrix temp_mat = DenseMatrix::Zero(mat_size, mat_size);

                temp_mat.block(0,0,mat_size-1,mat_size-1) += mat.template triangularView<Eigen::StrictlyUpper>();
                temp_mat.block(0,0,mat_size-1,mat_size-1) += mat.template triangularView<Eigen::StrictlyLower>().transpose();
                //local fields
                Vector loc = mat.diagonal();
                temp_mat.block(0,mat_size-1,mat_size-1,1) += loc;

                //insert in sparsematrix
                _quadmat += temp_mat.sparseView();
            }
            else{
                throw std::runtime_error("the number of variables and dimension do not match.");
            }
        }
    
        /**
         * @brief initialize matrix with matrix and labels
         * the form of matrix is assumed to be the following two forms:
         *
         * \f[
         * \begin{pmatrix}
         * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
         * J_{1,0} & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
         * \vdots & \vdots & \vdots & \vdots & \vdots \\
         * J_{N-1,0} & J_{N-1,1} & \cdots & J_{N-1,N-1} & h_{N-1}\\
         * h_{0} & h_{1} & \cdots & h_{N-1} & 1 \\
         * \end{pmatrix}
         * \f]
         *
         * or
         *
         * \f[
         * \begin{pmatrix}
         * h_{0} & J_{0,1} & \cdots & J_{0,N-1}\\
         * J_{1,0} & h_{1} & \cdots & J_{1,N-1}\\
         * \vdots & \vdots & \vdots & \vdots\\
         * J_{N-1,0} & J_{N-1,1} & \cdots & h_{N-1}\\
         * \end{pmatrix}
         * \f]
         *
         * if fix_format is set to false, the following shape is assumed:
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
         * @param mat
         * @param labels
         * @param fix_format
         */
        inline void _initialize_quadmat(const DenseMatrix& mat, const std::vector<IndexType>& labels_vec, bool fix_format){
    
            //initlaize label <-> index dict
            std::unordered_set<IndexType> labels(labels_vec.begin(), labels_vec.end());
            _idx_to_label = std::vector<IndexType>(labels.begin(), labels.end());
            std::sort(_idx_to_label.begin(), _idx_to_label.end());
            _set_label_to_idx();
    
            if(mat.rows() != mat.cols()){
                throw std::runtime_error("matrix must be a square matrix");
            }
    
            size_t mat_size = _idx_to_label.size() + 1;
            _quadmat = Matrix(mat_size, mat_size);
            _quadmat.setZero();
            _add_triangular_elements(mat, fix_format);
            _quadmat_get(mat_size-1, mat_size-1) = 1;
        }
    
    
        inline Linear<IndexType, FloatType> _generate_linear() const{
            Linear<IndexType, FloatType> ret_linear;
            for(size_t i=0; i<_idx_to_label.size(); i++){
                FloatType val = _quadmat_get(i, _idx_to_label.size());
                if(val != 0)
                    ret_linear[_idx_to_label[i]] = val;
            }
    
            return ret_linear;
        }
    
        inline Quadratic<IndexType, FloatType> _generate_quadratic() const{
            Quadratic<IndexType, FloatType> ret_quadratic;
            for(size_t i=0; i<_idx_to_label.size(); i++){
                for(size_t j=i+1; j<_idx_to_label.size(); j++){
                    FloatType val = _quadmat_get(i, j);
                    if(val != 0)
                        ret_quadratic[std::make_pair(_idx_to_label[i], _idx_to_label[j])] = val;
                }
            }
    
            return ret_quadratic;
        }

        /**
         * @brief initialize matrix with matrix and labels
         * the form of matrix is assumed to be the following form:
         *
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
         * @param mat
         * @param labels_vec
         */
        inline void _initialize_quadmat(const SparseMatrix& mat, const std::vector<IndexType>& labels_vec){
            this->_quadmat = mat;
            _idx_to_label = std::vector<IndexType>(labels_vec.begin(), labels_vec.end());
            std::sort(_idx_to_label.begin(), _idx_to_label.end());
            _set_label_to_idx();
        }
    
    
        /**
         * @brief change internal variable from Ising to QUBO ones for dense matrix
         * The following conversion is applied:
         *
         * \f[
         * \mathrm{offset} += \sum_{i<j} J_{ij} - \sum_{i}h_{i}
         * \f]
         * \f[
         * Q_ii += -2\left(\sum_{j}J_{ji}+\sum_{j}J_{ij}\right) + 2h_{i}
         * \f]
         * \f[
         * Q_{ij} = 4J_{ij}
         * \f]
         */
        template<typename T=DataType>
        inline void _spin_to_binary(dispatch_t<T, Dense> = nullptr){
            size_t num_variables = _idx_to_label.size();
            m_vartype = Vartype::BINARY;
            //calc col(row)wise-sum ((num_variables, 1))
            //Vector colwise_sum = _quadmat.block(0,0,num_variables,num_variables).colwise().sum();
            Vector colwise_sum(num_variables);
            for(size_t i=0; i<num_variables; i++){
                colwise_sum(i) = _quadmat.block(0,0,i,num_variables).col(i).sum();
            }

            Vector rowwise_sum = _quadmat.block(0,0,num_variables,num_variables).rowwise().sum();
            
            Vector local_field = _quadmat.block(0,num_variables,num_variables,1);

            //offset
            m_offset += colwise_sum.sum() - local_field.sum();

            //local field
            _quadmat.block(0,num_variables,num_variables,1)
                = 2 * local_field - 2 * (colwise_sum + rowwise_sum);
    
            //quadratic
            _quadmat.block(0,0,num_variables,num_variables) *= 4;
        }

        /**
         * @brief change internal variable from Ising to QUBO ones for sparse matrix
         * The following conversion is applied:
         *
         * \f[
         * \mathrm{offset} += \sum_{i<j} J_{ij} - \sum_{i}h_{i}
         * \f]
         * \f[
         * Q_ii += -2\left(\sum_{j}J_{ji}+\sum_{j}J_{ij}\right) + 2h_{i}
         * \f]
         * \f[
         * Q_{ij} = 4J_{ij}
         * \f]
         */
        template<typename T=DataType>
        inline void _spin_to_binary(dispatch_t<T, Sparse> = nullptr){
            size_t num_variables = _idx_to_label.size();
            m_vartype = Vartype::BINARY;
            //calc col(row)wise-sum ((num_variables, 1))
            //Vector colwise_sum = _quadmat.block(0,0,num_variables,num_variables).colwise().sum();
            //Vector rowwise_sum = _quadmat.block(0,0,num_variables,num_variables).rowwise().sum();
            
            Vector colwise_sum(num_variables);
            Vector rowwise_sum(num_variables);         
            colwise_sum.setZero();
            rowwise_sum.setZero();

            for(int k=0; k<_quadmat.outerSize(); k++){
                //k -> row index
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if((r < num_variables) && (c < num_variables)){
                        colwise_sum(c) += val;
                        rowwise_sum(r) += val;
                    }
                }
            }

            Vector local_field = _quadmat.block(0,num_variables,num_variables,1);

            //offset
            m_offset += colwise_sum.sum() - local_field.sum();

            //local field
            //_quadmat.block(0,num_variables,num_variables,1)
            //    = 2 * local_field - 2 * (colwise_sum + rowwise_sum);

            //quadratic
            //_quadmat.block(0,0,num_variables,num_variables) *= 4;


            Vector new_local_field = 2 * local_field - 2 * (colwise_sum + rowwise_sum);

            std::vector<Eigen::Triplet<FloatType>> triplets;
            triplets.reserve(_quadmat.nonZeros());
            for(size_t r=0; r<num_variables; r++){
                triplets.emplace_back(r, num_variables, new_local_field(r));
            }

            for(int k=0; k<_quadmat.outerSize(); k++){
                //k -> row index
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if((r < num_variables) && (c < num_variables)){
                        triplets.emplace_back(r, c, 4*val);
                    }
                }
            }

            triplets.emplace_back(num_variables, num_variables, 1);

            _quadmat = SparseMatrix(num_variables+1, num_variables+1);
            _quadmat.setFromTriplets(triplets.begin(), triplets.end());
        }
    
        /**
         * @brief change internal variable from QUBO to Ising ones for dense matrix
         * The following conversion is applied:
         *
         * \f[
         * \mathrm{offset} += \frac{1}{4}\sum_{i<j} Q_{ij} + \frac{1}{2}\sum_{i}Q_{ii}
         * \f]
         * \f[
         * h_i += \frac{1}{4}\left(\sum_{j}Q_{ji}+\sum_{j}Q_{ij}\right) + \frac{1}{2}Q_{ii}
         * \f]
         * \f[
         * J_{ij} = \frac{1}{4}Q_{ij}
         * \f]
         */
        template<typename T=DataType>
        inline void _binary_to_spin(dispatch_t<T, Dense> = nullptr){
            size_t num_variables = _idx_to_label.size();
            m_vartype = Vartype::SPIN;
            //calc col(row)wise-sum ((num_variables, 1))
            //Vector colwise_sum = _quadmat.block(0,0,num_variables,num_variables).colwise().sum();
            Vector colwise_sum(num_variables);
            for(size_t i=0; i<num_variables; i++){
                colwise_sum(i) = _quadmat.block(0,0,i,num_variables).col(i).sum();
            }
            Vector rowwise_sum = _quadmat.block(0,0,num_variables,num_variables).rowwise().sum();
            
            Vector local_field = _quadmat.block(0,num_variables,num_variables,1);
    
            //offset
            m_offset += 0.25 * colwise_sum.sum() + 0.5 * local_field.sum();
    
            //local field
            _quadmat.block(0,num_variables,num_variables,1)
                = 0.5 * local_field + 0.25 * (colwise_sum + rowwise_sum);
    
            //quadratic
            _quadmat.block(0,0,num_variables,num_variables) *= 0.25;
        }

        /**
         * @brief change internal variable from QUBO to Ising ones for sparse matrix
         * The following conversion is applied:
         *
         * \f[
         * \mathrm{offset} += \frac{1}{4}\sum_{i<j} Q_{ij} + \frac{1}{2}\sum_{i}Q_{ii}
         * \f]
         * \f[
         * h_i += \frac{1}{4}\left(\sum_{j}Q_{ji}+\sum_{j}Q_{ij}\right) + \frac{1}{2}Q_{ii}
         * \f]
         * \f[
         * J_{ij} = \frac{1}{4}Q_{ij}
         * \f]
         */
        template<typename T=DataType>
        inline void _binary_to_spin(dispatch_t<T, Sparse> = nullptr){
            size_t num_variables = _idx_to_label.size();
            m_vartype = Vartype::SPIN;
            //calc col(row)wise-sum ((num_variables, 1))
            //Vector colwise_sum = _quadmat.block(0,0,num_variables,num_variables).colwise().sum();
            //Vector rowwise_sum = _quadmat.block(0,0,num_variables,num_variables).rowwise().sum();

            Vector colwise_sum(num_variables);
            Vector rowwise_sum(num_variables);         
            colwise_sum.setZero();
            rowwise_sum.setZero();

            for(int k=0; k<_quadmat.outerSize(); k++){
                //k -> row index
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if((r < num_variables) && (c < num_variables)){
                        colwise_sum(c) += val;
                        rowwise_sum(r) += val;
                    }
                }
            }

            Vector local_field = _quadmat.block(0,num_variables,num_variables,1);
    
            //offset
            m_offset += 0.25 * colwise_sum.sum() + 0.5 * local_field.sum();
    
            //local field
            //_quadmat.block(0,num_variables,num_variables,1)
            //    = 0.5 * local_field + 0.25 * (colwise_sum + rowwise_sum);
    
            //quadratic
            //quadmat.block(0,0,num_variables,num_variables) *= 0.25;
            
            Vector new_local_field = 0.5 * local_field + 0.25 * (colwise_sum + rowwise_sum);

            std::vector<Eigen::Triplet<FloatType>> triplets;
            triplets.reserve(_quadmat.nonZeros());
            for(size_t r=0; r<num_variables; r++){
                triplets.emplace_back(r, num_variables, new_local_field(r));
            }

            for(int k=0; k<_quadmat.outerSize(); k++){
                //k -> row index
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if((r < num_variables) && (c < num_variables)){
                        triplets.emplace_back(r, c, 0.25*val);
                    }
                }
            }

            triplets.emplace_back(num_variables, num_variables, 1);

            _quadmat.setFromTriplets(triplets.begin(), triplets.end());
        }
    
    
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
            _initialize_quadmat(linear, quadratic);
        }
    
        /**
         * @brief BinaryQuadraticModel constructor.
         *
         * @param linear
         * @param quadratic
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const Linear<IndexType, FloatType> &linear,
            const Quadratic<IndexType, FloatType> &quadratic,
            const Vartype vartype
        ): BinaryQuadraticModel(linear, quadratic, 0.0, vartype){}
    
    
        /**
         * @brief BinaryQuadraticModel constructor (with matrix);
         *
         * @param mat
         * @param labels_vec
         * @param offset
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const Eigen::Ref<const DenseMatrix>& mat,
            const std::vector<IndexType>& labels_vec,
            const FloatType &offset,
            const Vartype vartype,
            bool fix_format = true
        ):
            m_offset(offset),
            m_vartype(vartype)
        {
            _initialize_quadmat(mat, labels_vec, fix_format);
        }
    
        /**
         * @brief BinaryQuadraticModel constructor (with matrix);
         *
         * @param mat
         * @param labels_vec
         * @param vartype
         *
         */
        BinaryQuadraticModel
        (
            const Eigen::Ref<const DenseMatrix>& mat,
            const std::vector<IndexType>& labels_vec,
            const Vartype vartype,
            bool fix_format = true
        ): BinaryQuadraticModel(mat, labels_vec, 0.0, vartype, fix_format){}

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
            const SparseMatrix& mat,
            const std::vector<IndexType>& labels_vec,
            const FloatType &offset,
            const Vartype vartype
        ):
            m_offset(offset),
            m_vartype(vartype)
        {
            _initialize_quadmat(mat, labels_vec);
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
            const SparseMatrix& mat,
            const std::vector<IndexType>& labels_vec,
            const Vartype vartype
        ): BinaryQuadraticModel(mat, labels_vec, 0.0, vartype){}
    
    
        BinaryQuadraticModel(const BinaryQuadraticModel&) = default;
    
        /**
         * @brief get the number of variables
         *
         * @return The number of variables.
         */
        size_t get_num_variables() const{
            return _idx_to_label.size();
        }
    
        /**
         * @brief Return the number of variables.
         * @deprecated use get_num_variables instead.
         * 
         * @return The number of variables.
         */
        size_t length() const
        {
            return get_num_variables();
        }
    
        /**
         * @brief Return true if the variable contains v.
         * 
         * @return Return true if the variable contains v.
         * @param v
         */
        bool contains(const IndexType &v) const
        {
            if(_label_to_idx.find(v) != _label_to_idx.end())
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
         * @return A linear bias.
         */
        FloatType get_linear(IndexType label_i) const{
            return _mat(label_i);
        }
    
        /**
         * @brief Get linear object
         *
         * @return A linear object
         */
        Linear<IndexType, FloatType> get_linear() const{
            return _generate_linear();
        }
    
        /**
         * @brief Get the element of quadratic object
         * 
         * @return A quadratic bias.
         */
        FloatType get_quadratic(IndexType label_i, IndexType label_j) const
        {
            return _mat(label_i, label_j);
        }
    
        /**
         * @brief Get uadratic object
         * 
         * @return A quadratic object.
         */
        Quadratic<IndexType, FloatType> get_quadratic() const
        {
            return _generate_quadratic();
        }
    
        /**
         * @brief Get the offset
         * 
         * @return An offset.
         */
        FloatType get_offset() const
        {
            return this->m_offset;
        }
    
        /**
         * @brief Get the vartype object
         * 
         * @return Type of the model.
         */
        Vartype get_vartype() const
        {
            return this->m_vartype;
        }
    
        /**
         * @brief Get variables
         *
         * @return variables
         */
        const std::vector<IndexType>& get_variables() const{
            return this->_idx_to_label;
        }
    
    
        /**
         * @brief Create an empty BinaryQuadraticModel
         * 
         * @return empty object
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
            //add new label if not exist
            _add_new_label(v);
            _mat(v) += bias;
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
         * @param u
         * @param v
         * @param bias
         */
        void add_interaction
        (
            const IndexType &u,
            const IndexType &v,
            const FloatType &bias
        )
        {
            //add labels u and v
            _add_new_label(u);
            _add_new_label(v);
            _mat(u, v) += bias;
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
            _delete_label(v);
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
         * @param u
         * @param v
         */
        void remove_interaction
        (
            const IndexType &u,
            const IndexType &v      
        )
        {
            _mat(u, v) = 0;
            _delete_label(u, false);
            _delete_label(v, false);
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
            if(scalar == 0.0)
                throw std::runtime_error("scalar must not be zero");
    
            // scale
            _quadmat *= scalar;
    
            // revert scale of linear
            for(const auto &it : ignored_variables)
            {
                _mat(it) *= 1.0/scalar;
            }
    
    
            // revert scale of quadratic
            for(const auto &it : ignored_interactions)
            {
                _mat(it.first, it.second) *= 1.0/scalar;
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
            FloatType lin_min = _min_linear();
            FloatType lin_max = _max_linear();
            FloatType quad_min = _min_quadratic();
            FloatType quad_max = _max_quadratic();
    
            std::vector<FloatType> v_scale =
            {
                lin_min / l_range.first,
                lin_max / l_range.second,
                quad_min / q_range.first,
                quad_max / q_range.second
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
            const Quadratic<IndexType, FloatType>& quadratic = this->get_quadratic();
            for(const auto &it : quadratic)
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
            add_offset(_mat(v)*value);
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
    
            if(m_vartype == Vartype::SPIN)
            {
                size_t i = _label_to_idx.at(v);
                _quadmat.row(i) *= -1;
                _quadmat.col(i) *= -1;
            }
            else if(m_vartype == Vartype::BINARY)
            {
                //change vartype to spin
                this->change_vartype(Vartype::SPIN);
    
                size_t i = _label_to_idx.at(v);
                _quadmat.row(i) *= -1;
                _quadmat.col(i) *= -1;
    
    
                //change vartype to binary
                this->change_vartype(Vartype::BINARY);
            }
        }
    
        ///**
        // * @brief Enforce u, v being the same variable in a binary quadratic model. (currently disabled)
        // * 
        // * @param u
        // * @param v
        // */
        //void contract_variables
        //(
        //    const IndexType &u,
        //    const IndexType &v
        //)
        //{
    
        //    if(this->m_vartype == Vartype::BINARY){
        //        // the quadratic bias becomes linear
        //        _mat(u) += _mat(v) + _mat(u,v);
        //    }
        //    else if(this->m_vartype == Vartype::SPIN){
        //        _mat(u) += _mat(v);
        //        m_offset += _mat(u,v);
        //    }
        //    this->remove_interaction(u,v);
    
    
    
        //}
    
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
            if(m_vartype == Vartype::BINARY && vartype == Vartype::SPIN) // binary -> spin
            {
                _binary_to_spin();                
            }
            else if(m_vartype == Vartype::SPIN && vartype == Vartype::BINARY) // spin -> binary
            {
                _spin_to_binary();
            }
        }
    
        /**
         * @brief Create a binary quadratic model with the specified vartype.
         * This function generates and returns a new object.
         *
         * @param vartype
         * @param inplace if set true, the current object is converted.
         *
         * @return created object
         */
        BinaryQuadraticModel<IndexType, FloatType, DataType> change_vartype
        (
            const Vartype &vartype,
            bool inplace
        )
        {
            BinaryQuadraticModel<IndexType, FloatType, DataType> new_bqm = *this;
            if(inplace == true){
                this->change_vartype(vartype);
            }
            new_bqm.change_vartype(vartype);
    
            return new_bqm;
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
            //initialize vector
            Vector s = Vector::Zero(_quadmat.rows());
            for(const auto& elem : sample){
                s[_label_to_idx.at(elem.first)] = elem.second;
            }
            s[_quadmat.rows()-1] = 1;
            
            return en + (s.transpose() * _quadmat * s) - 1;
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
            BinaryQuadraticModel<IndexType, FloatType, DataType> bqm = change_vartype(Vartype::BINARY, false);
    
            const Linear<IndexType, FloatType>& linear = bqm.get_linear();
            Quadratic<IndexType, FloatType> Q = bqm.get_quadratic();
            FloatType offset = bqm.get_offset();
            for(const auto &it : linear)
            {
                Q[std::make_pair(it.first, it.first)] = it.second;
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
        static BinaryQuadraticModel<IndexType, FloatType, DataType> from_qubo(const Quadratic<IndexType, FloatType>& Q, FloatType offset=0.0)
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
            BinaryQuadraticModel<IndexType, FloatType, DataType> bqm = change_vartype(Vartype::SPIN, false);
    
            const Linear<IndexType, FloatType>& linear = bqm.get_linear();
            const Quadratic<IndexType, FloatType>& quadratic = bqm.get_quadratic();
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
        static BinaryQuadraticModel<IndexType, FloatType, DataType> from_ising(const Linear<IndexType, FloatType>& linear, const Quadratic<IndexType, FloatType>& quadratic, FloatType offset=0.0)
        {
            return BinaryQuadraticModel<IndexType, FloatType, DataType>(linear, quadratic, offset, Vartype::SPIN);
        }
    
    
        /**
         * @brief generate (Dense or Sparse) interaction matrix with given list of indices
         * The generated matrix will be the following triangular matrix:
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
         * @param indices
         *
         * @return corresponding interaction matrix (Eigen)
         */
        Matrix interaction_matrix() const {
            return this->_quadmat;
        }
    
    
        using json = nlohmann::json;
    
        /**
         * @brief Convert the binary quadratic model to a dense-version serializable object
         *
         * @return An object that can be serialized (nlohmann::json)
         */
        template<typename T=DataType>
        json to_serializable(dispatch_t<T, Dense> = nullptr) const
        {
            std::string schema_version = "3.0.0-dense";
            /*
             * output sample
             * >>> bqm = dimod.BinaryQuadraticModel({'c': -1, 'd': 1}, {('a', 'd'): 2, ('b', 'e'): 5, ('a', 'c'): 3}, 0.0, dimod.BINARY)
             *
             * >>> bqm.to_serializable()
             * {'type': 'BinaryQuadraticModel', 'version': {'bqm_schema': '3.0.0-dense'}, 'use_bytes': False, 'index_type': 'uint16', 'bias_type': 'float32', 'num_variables': 5, 'variable_labels': ['a', 'b', 'c', 'd', 'e'], 'variable_type': 'BINARY', 'offset': 0.0, 'info': {}, 'biases': [0.0, 0.0, -1.0, 1.0, 0.0,...]}
             */
    
            //set index_dtype
            std::string index_dtype = this->get_num_variables() <= 65536UL ? "uint16" : "uint32";
    
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
    
            //copy matrix to std::vector
            std::vector<FloatType> biases(_quadmat.data(), _quadmat.data() + _quadmat.size());
    
            json output;
            output["type"] = "BinaryQuadraticModel";
            output["version"] = {{"bqm_schema", schema_version}};
            output["variable_labels"] = this->get_variables();
            output["use_bytes"] = false;
            output["index_type"] = index_dtype;
            output["bias_type"] = bias_type;
            output["num_variables"] = this->get_num_variables();
            output["variable_type"] = variable_type;
            output["offset"] = m_offset;
            output["info"] = {};
            output["biases"] = biases;
    
            return output;
        }

        /**
         * @brief Convert the binary quadratic model to a serializable object
         *
         * @return An object that can be serialized (nlohmann::json)
         */
        template<typename T=DataType>
        json to_serializable(dispatch_t<T, Sparse> = nullptr) const
        {
            std::string schema_version = "3.0.0";
            /*
             * output sample
             * >>> bqm = dimod.BinaryQuadraticModel({'c': -1, 'd': 1}, {('a', 'd'): 2, ('b', 'e'): 5, ('a', 'c'): 3}, 0.0, dimod.BINARY)
             *
             * >>> bqm.to_serializable()
             * {'type': 'BinaryQuadraticModel', 'version': {'bqm_schema': '3.0.0'}, 'use_bytes': False, 'index_type': 'uint16', 'bias_type': 'float32', 'num_variables': 5, 'num_interactions': 3, 'variable_labels': ['a', 'b', 'c', 'd', 'e'], 'variable_type': 'BINARY', 'offset': 0.0, 'info': {}, 'linear_biases': [0.0, 0.0, -1.0, 1.0, 0.0], 'quadratic_biases': [3.0, 2.0, 5.0], 'quadratic_head': [0, 0, 1], 'quadratic_tail': [2, 3, 4]}
             */
    
            //set index_dtype
            std::string index_dtype = this->get_num_variables() <= 65536UL ? "uint16" : "uint32";
    
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
            output["version"] = {{"bqm_schema", schema_version}};
            output["variable_labels"] = this->get_variables();
            output["use_bytes"] = false;
            output["index_type"] = index_dtype;
            output["bias_type"] = bias_type;
            output["num_variables"] = this->get_num_variables();
            output["variable_type"] = variable_type;
            output["offset"] = m_offset;
            output["info"] = "";

            //biases
            size_t mat_size = this->get_num_variables() + 1;
            Vector l_bias_vec = _quadmat.block(0, mat_size-1, mat_size-1, 1);
            std::vector<FloatType> l_bias(l_bias_vec.size());
            for(int i=0; i<l_bias_vec.size(); i++){
                l_bias[i] = l_bias_vec(i);
            }

            std::vector<FloatType> q_bias;
            std::vector<size_t> q_head;
            std::vector<size_t> q_tail;

            q_bias.reserve(_quadmat.nonZeros());
            q_head.reserve(_quadmat.nonZeros());
            q_tail.reserve(_quadmat.nonZeros());

            for(int k=0; k<_quadmat.outerSize(); k++){
                for(SpIter it(_quadmat, k); it; ++it){
                    size_t r        = it.row();
                    size_t c        = it.col();
                    FloatType val   = it.value();

                    if((r < mat_size-1) && (c < mat_size-1)){
                        q_bias.push_back(val);
                        q_head.push_back(r);
                        q_tail.push_back(c);
                    }

                }
            }

            output["linear_biases"] = l_bias;
            output["quadratic_biases"] = q_bias;
            output["quadratic_head"] = q_head;
            output["quadratic_tail"] = q_tail;
            output["num_interactions"] = q_bias.size();

            return output;
        }
    
        /**
         * @brief Create a BinaryQuadraticModel instance from a serializable object.
         * 
         * @tparam IndexType_serial
         * @tparam FloatType_serial
         * @param input
         * @return BinaryQuadraticModel<IndexType_serial, FloatType_serial> 
         */
        template <typename IndexType_serial = IndexType, typename FloatType_serial = FloatType, typename T=DataType>
        static BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> from_serializable(const json &input, dispatch_t<T, Dense> = nullptr)
        {
            //extract type and version
            std::string type = input["type"];
            if(type != "BinaryQuadraticModel")
            {
                throw std::runtime_error("Type must be \"BinaryQuadraticModel\".\n");
            }
            std::string version = input["version"]["bqm_schema"];
            if(version != "3.0.0-dense")
            {
                throw std::runtime_error("bqm_schema must be 3.0.0-dense.\n");
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
    
            //extract biases
            std::vector<IndexType_serial> variables = input["variable_labels"];
            std::vector<FloatType_serial> biases = input["biases"];
            FloatType offset = input["offset"];
    
            size_t mat_size = variables.size() + 1;
            Eigen::Map<Matrix> mat(biases.data(), mat_size, mat_size);
    
            BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> bqm(mat, variables, offset, vartype);
            return bqm;
        }

        /**
         * @brief Create a BinaryQuadraticModel instance from a serializable object.
         * 
         * @tparam IndexType_serial
         * @tparam FloatType_serial
         * @param input
         * @return BinaryQuadraticModel<IndexType_serial, FloatType_serial> 
         */
        template <typename IndexType_serial = IndexType, typename FloatType_serial = FloatType, typename T=DataType>
        static BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> from_serializable(const json &input, dispatch_t<T, Sparse> = nullptr)
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
    
            //extract biases
            std::vector<IndexType_serial> variables = input["variable_labels"];
            FloatType offset = input["offset"];

            std::vector<FloatType_serial> l_bias = input["linear_biases"];
            std::vector<size_t> q_head = input["quadratic_head"];
            std::vector<size_t> q_tail = input["quadratic_tail"];
            std::vector<FloatType_serial> q_bias = input["quadratic_biases"];

            //make triplets
            std::vector<Eigen::Triplet<FloatType_serial>> triplets;
            triplets.reserve(q_bias.size() + l_bias.size());

            size_t mat_size = variables.size() + 1;

            for(size_t i=0; i<l_bias.size(); i++){
                if(l_bias[i] != 0)
                    triplets.emplace_back(i, mat_size-1, l_bias[i]);
            }

            for(size_t i=0; i<q_bias.size(); i++){
                triplets.emplace_back(q_head[i], q_tail[i], q_bias[i]);
            }

            triplets.emplace_back(mat_size-1, mat_size-1, 1);

    
            SparseMatrix mat(mat_size, mat_size);

            mat.setFromTriplets(triplets.begin(), triplets.end());

            BinaryQuadraticModel<IndexType_serial, FloatType_serial, DataType> bqm(mat, variables, offset, vartype);
            return bqm;
        }
    
    };
}

#endif
