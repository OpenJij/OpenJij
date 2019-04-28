#pragma once

#include <random>

#include "../graph/dense.h"
#include "method.h"
#include "../updater/classical_updater.h"

namespace openjij {
    namespace method {

        //TODO: double -> FloatType (template)
        class ClassicalIsing : public Method, public updater::ClassicalUpdater{
            //general classical ising model
            private:
                graph::Spins spins;
                graph::Dense<double> interaction;
                //random number generator
                //TODO: use MT or xorshift
                std::mt19937 mt;
                std::uniform_int_distribution<> uid;
                std::uniform_real_distribution<> urd;

            public:
                ClassicalIsing(const graph::Dense<double>& interaction);
                ClassicalIsing(const graph::Dense<double>& interaction, graph::Spins& spins);

                void initialize_spins();
                void set_spins(graph::Spins& initial_spins);

                virtual double update(const double beta, const std::string& algo = "") override;

                //do simulated annelaing
                void simulated_annealing(const double beta_min, const double beta_max, const double step_length, const size_t step_num, const std::string& algo = "");

                const graph::Spins get_spins() const;
        };
    } // namespace method
} // namespace openjij

