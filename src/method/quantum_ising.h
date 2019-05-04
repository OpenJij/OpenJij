#pragma once

#include <random>
#include <utility>
#include <vector>

#include "../graph/dense.h"
#include "method.h"
#include "../updater/quantum_updater.h"

namespace openjij {
    namespace method {

        //TODO: double -> FloatType (template)
        class QuantumIsing : public Method, public updater::QuantumUpdater{
            //general transverse-field ising model with discrete trotter slices
            public:
                using Schedule = std::vector<std::pair<double, size_t>>;

                QuantumIsing(const graph::Dense<double>& interaction, size_t num_trotter_slices);
                QuantumIsing(const graph::Dense<double>& interaction, size_t num_trotter_slices, graph::Spins& classical_spins);

                void initialize_spins();
                void set_spins(graph::Spins& initial_spin);

                virtual double update(const double beta, const double gamma, const double s, const std::string& algo = "") override;

                void simulated_quantum_annealing(const double beta, const double gamma, const size_t step_length, const size_t step_num, const std::string& algo = "");
                void simulated_quantum_annealing(const double beta, const double gamma, const Schedule& schedule, const std::string& algo = "");

                TrotterSpins get_spins() const;

            private:
                TrotterSpins spins; //spins with trotter slices
                graph::Dense<double> interaction; //original interaction

                //random number generators
                //TODO: use MT or xorshift?
                std::mt19937 mt;
                std::uniform_int_distribution<> uid;
                std::uniform_int_distribution<> uid_trotter;
                std::uniform_real_distribution<> urd;

                //mod function for dealing trotter indices
                inline size_t mod_t(int64_t a) const;
        };
    } // namespace method
} // namespace openjij

