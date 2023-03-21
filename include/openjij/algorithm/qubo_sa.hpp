#include <iostream>
#include <stdio.h>
#include <stddef.h>
#include <limits.h>
#include <Eigen/Sparse>
#include <vector>
#include <openjij/utility/random.hpp>

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using RandomEngine = openjij::utility::Xorshift;

// calc_energy_gradient function calculates the energy gradient for a given state x and QUBO problem represented by matrix A
void calc_energy_gradient(
    const SparseMatrix &A,
    const Eigen::VectorXd &x, 
    Eigen::VectorXd &grad)
{
    Eigen::VectorXd x_sign = Eigen::VectorXd::Ones(x.size()) - 2.0 * x;
    grad = 2.0 * (A * x).cwiseProduct(x_sign);
}

// sa_qubo_ssf function performs simulated annealing for QUBO problems with metropolis update using single spin flip (ssf)
void sa_qubo_ssf(
    const SparseMatrix &qubo,
    Eigen::Ref<Eigen::VectorXd> flip_energy,
    Eigen::Ref<Eigen::VectorXd> state,
    int beta_steps,
    Eigen::Ref<Eigen::VectorXd> beta_schedule
){
    Eigen::setNbThreads(1);
    Eigen::initParallel();

    size_t n = qubo.rows();
    Eigen::VectorXd x_sign = Eigen::VectorXd::Ones(n) - 2.0 * state;
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
                flip_energy += 2 * x_sign[i] * qubo.row(i).transpose().cwiseProduct(x_sign);
                flip_energy[i] = -origin_dEk;
                x_sign[i] *= -1;
            }
        }
    }
    state = (Eigen::VectorXd::Ones(n) - x_sign) / 2.0;
}

void sa_qubo_ssf_from_coo(
    Eigen::Ref<const Eigen::VectorXi> row_indices,
    Eigen::Ref<const Eigen::VectorXi> col_indices,
    Eigen::Ref<const Eigen::VectorXd> values,
    Eigen::Ref<Eigen::VectorXd> state,
    Eigen::Ref<Eigen::VectorXd> beta_schedule
){
    assert (row_indices.size() == col_indices.size());
    assert (row_indices.size() == values.size());
    size_t num_rows = state.size();

    SparseMatrix qubo(num_rows, num_rows);
    qubo.reserve(row_indices.size());
    for (size_t i=0; i < row_indices.size(); i++){
        qubo.coeffRef(row_indices[i], col_indices[i]) = values[i];
    }
    qubo.makeCompressed();

    Eigen::VectorXd flip_energy(num_rows);
    calc_energy_gradient(qubo, state, flip_energy);

    int beta_steps = beta_schedule.size();
    sa_qubo_ssf(qubo, flip_energy, state, beta_steps, beta_schedule);
}
