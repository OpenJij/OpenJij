#pragma once

#include <string>
#include <utility>
#include <vector>

#include "algorithm.h"
#include "../updater/quantum_updater.h"

namespace openjij{
    namespace algorithm{
        using namespace openjij::updater;
        using Schedule = std::vector<std::pair<double, size_t>>;

        //simulated quantum annealing
        class SQA : public Algorithm{
            public:
                SQA(const double beta, const Schedule& schedule);
                SQA(const double beta, const double gamma_min, const double gamma_max, const size_t step_length, const size_t step_num);

                // do SQA protocol
                void run(QuantumUpdater& updater, const std::string& algo = "");

            private:
                double beta;
                Schedule schedule;
        };
    }
}
