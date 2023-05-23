#pragma once
#include <Eigen/Sparse>
#include "openjij/algorithm/sparsegraph.hpp"

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;


/**
 * @brief local search for QUBO using single spin flip (ssf)
 * @param[in] qubo QUBO matrix
 * @param[in,out] state state
 * @param[out] grad energy gradient
 * @param[in] max_iter maximum number of iterations
*/
int local_search_sparse_qubo_ssf(
    const SparseSymmetricGraph &qubo,
    Eigen::Ref<Eigen::VectorXd> grad,
    Eigen::Ref<Eigen::VectorXd> state,
    const int max_iter
){
    size_t n = qubo.num_vars;
    Eigen::VectorXd x_sign = Eigen::VectorXd::Ones(n+1) - 2.0 * state;
    int iter_counter = 0;
    for (size_t t=0; t < max_iter; t++){
        int flip_flag = 0;
        iter_counter += 1;
        for (size_t i=0; i < n; i++){
            if (grad[i] < 0.0){
                flip_flag = 1;
                double origin_grad = grad[i];
                grad += 2 * x_sign[i] * qubo.interaction.row(i).transpose().cwiseProduct(x_sign);
                grad[i] = -origin_grad;
                x_sign[i] *= -1;
            }
        }
        if (flip_flag == 0) break;
    }
    state = (Eigen::VectorXd::Ones(n+1) - x_sign) / 2.0;
    return iter_counter;
}
