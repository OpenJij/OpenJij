#pragma once
#include <Eigen/Sparse>
#include "openjij/algorithm/sparsegraph.hpp"


std::pair<std::vector<Eigen::VectorXd>, std::vector<double>> solve_qubo_bruteforce(const SparseSymmetricGraph &qubo, const double epsilon=1e-8){
    size_t n = qubo.num_vars;
    Eigen::VectorXd state = Eigen::VectorXd::Zero(n+1);
    state[n] = 1.0; // the last element is a constant term
    double best_energy = std::numeric_limits<double>::max();

    std::vector<Eigen::VectorXd> best_states;
    std::vector<double> best_energies;

    for (size_t i=0; i < (1 << n); i++){
        for (size_t j=0; j < n; j++){
            state[j] = (i >> j) & 1;
        }
        double energy = state.transpose() * qubo.interaction * state;
        if (energy < best_energy - epsilon){
            best_energy = energy;
            best_energies.clear();
            best_states.clear();
            best_states.push_back(state);
            best_energies.push_back(energy);
        }else if (std::abs(energy - best_energy) < epsilon){
            best_energy = std::min(energy, best_energy);
            best_states.push_back(state);
            best_energies.push_back(energy);
        }
    }
    return std::make_pair(best_states, best_energies);
}

