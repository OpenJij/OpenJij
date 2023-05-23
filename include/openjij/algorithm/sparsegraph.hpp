//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once
#include <iostream>
#include <stdio.h>
#include <stddef.h>
#include <limits.h>
#include <Eigen/Sparse>
#include <vector>
#include "openjij/utility/random.hpp"


using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using RandomEngine = openjij::utility::Xorshift;


class VectorSizeNp1Error : public std::exception {
public:
    const char* what() const noexcept override {
        return "Vector size is n+1 not n, because we needs the last fixed variable (x=1) for diagonal terms.";
    }
};

class COOVectorSizeError : public std::exception {
public:
    const char* what() const noexcept override {
        return "COO vector size is not equal to the number of variables.";
    }
};


class SparseSymmetricGraph {
public:
    SparseMatrix interaction;
    int num_vars;
    explicit SparseSymmetricGraph(
        Eigen::Ref<const Eigen::VectorXi> row_indices,
        Eigen::Ref<const Eigen::VectorXi> col_indices,
        Eigen::Ref<const Eigen::VectorXd> values
    ){
        if ((row_indices.size() != col_indices.size()) || (row_indices.size() != values.size())){
            throw COOVectorSizeError();
        }

        int maxIdx = *std::max_element(row_indices.begin(), row_indices.end());
        maxIdx = std::max(maxIdx, *std::max_element(col_indices.begin(), col_indices.end()));
        num_vars = maxIdx+1;

        // We delete diagonal elements by expanding the last row and column
        // We assume that the last element of a vector is one.
        // This is a trick to ignore diagonal elements.
        // The last row and column represent diagonal elements of interactionrix.
        interaction = SparseMatrix(num_vars+1, num_vars+1);
        interaction.reserve(row_indices.size());
        for (size_t i=0; i < row_indices.size(); i++){
            if (row_indices[i] == col_indices[i]){
                // We delete diagonal elements by expanding the last row and column
                interaction.coeffRef(num_vars, row_indices[i]) = values[i]/2.0;
                interaction.coeffRef(row_indices[i], num_vars) = values[i]/2.0;
            }else{
                // For symmetric interactionrix, we need to add values to both (i,j) and (j,i)
                double current_value = interaction.coeffRef(row_indices[i], col_indices[i]);
                interaction.coeffRef(row_indices[i], col_indices[i]) = values[i]/2.0 + current_value;
                interaction.coeffRef(col_indices[i], row_indices[i]) = values[i]/2.0 + current_value;
            }
        }
        interaction.makeCompressed();
    }

    void calc_qubo_energy_gradient(
        Eigen::Ref<const Eigen::VectorXd> x, 
        Eigen::Ref<Eigen::VectorXd> grad)
    {
        // Check the size of x with message that is easy to understand
        if (x.size() != num_vars + 1){
            throw VectorSizeNp1Error();
        }
        if (grad.size() != num_vars + 1){
            throw VectorSizeNp1Error();
        }
        Eigen::VectorXd x_sign = Eigen::VectorXd::Ones(x.size()) - 2.0 * x;
        grad = 2.0 * (interaction * x).cwiseProduct(x_sign);
    }

    float calc_energy(Eigen::Ref<const Eigen::VectorXd> x){
        if (x.size() != num_vars + 1){
            throw VectorSizeNp1Error();
        }
        return x.transpose() * interaction * x;
    }
};



