#pragma once

#include <string>
#include <utility>
#include <vector>

#include "algorithm.h"
#include "../updater/classical_updater.h"

namespace openjij{
    namespace algorithm{
        using namespace openjij::updater;
        using Schedule = std::vector<std::pair<double, size_t>>;

        //simulated annealing
        class SA : public Algorithm{
            public:
                SA(const Schedule& schedule);
                SA(const double beta_min, const double beta_max, const size_t step_length, const size_t step_num);

                // do SA protocol
                void run(ClassicalUpdater& updater, const std::string& algo = "");

            private:
                Schedule schedule;
        };
    }
}
