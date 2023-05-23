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
#include "openjij/algorithm/sparsegraph.hpp"

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using RandomEngine = openjij::utility::Xorshift;



/**
 * @brief calculate energy gradient
 * @param[in] A QUBO matrix
 * @param[in] x state
 * @param[out] grad energy gradient
*/
void calc_energy_gradient(
    const SparseMatrix &A,
    const Eigen::VectorXd &x, 
    Eigen::VectorXd &grad)
{
    Eigen::VectorXd x_sign = Eigen::VectorXd::Ones(x.size()) - 2.0 * x;
    grad = 2.0 * (A * x).cwiseProduct(x_sign);
}


/**
 * @brief simulated annealing for QUBO problems with metropolis update using single spin flip (ssf)
 * @param[in] qubo QUBO matrix
 * @param[in] flip_energy energy gradient
 * @param[in,out] state state
 * @param[in] beta_schedule beta schedule
*/
void sa_sparse_qubo_ssf(
    const SparseSymmetricGraph &qubo,
    Eigen::Ref<Eigen::VectorXd> flip_energy,
    Eigen::Ref<Eigen::VectorXd> state,
    Eigen::Ref<Eigen::VectorXd> beta_schedule
){

    size_t n = qubo.num_vars;
    int beta_steps = beta_schedule.size();
    Eigen::VectorXd x_sign = Eigen::VectorXd::Ones(n+1) - 2.0 * state;
    auto random_engine = RandomEngine(0);
    auto urd = std::uniform_real_distribution<>(0, 1.0);
    for (size_t t=0; t < beta_steps; t++){
        double beta = beta_schedule[t];
        for (size_t i=0; i < n; i++){
            int flip_flag = 0;
            if (flip_energy[i] <= 0.0){
                flip_flag = 1;
            } else {
                if (urd(random_engine) < std::exp(-beta * flip_energy[i])){
                    flip_flag = 1;
                }
            }

            if (flip_flag) {
                double origin_dEk = flip_energy[i];
                flip_energy += 2 * x_sign[i] * qubo.interaction.row(i).transpose().cwiseProduct(x_sign);
                flip_energy[i] = -origin_dEk;
                x_sign[i] *= -1;
            }
        }
    }
    state = (Eigen::VectorXd::Ones(n+1) - x_sign) / 2.0;
}


