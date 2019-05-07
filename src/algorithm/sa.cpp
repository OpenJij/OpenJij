#include "sa.h"
#include <cmath>

namespace openjij {
    namespace algorithm {

        SA::SA(const Schedule& schedule):schedule(schedule){}

        SA::SA(const double beta_min, const double beta_max, const size_t step_length, const size_t step_num) {
            const double r_beta = std::pow(beta_max/beta_min, 1.0/step_num);
            double beta = beta_min;
            for (size_t i = 0; i < step_num; i++) {
                schedule.emplace_back(std::make_pair(beta, step_length));
                beta *= r_beta;
            }
        }

        void SA::run(updater::ClassicalUpdater& updater, const std::string& algo) {
            // anneal
            for (const auto& elem : schedule) {
                const double beta = elem.first;
                const size_t step_length = elem.second;

                // step length -> number of MCS for each step
                for (size_t i = 0; i < step_length; i++) {
                    updater.update(beta, algo);
                }
            }
        }
    } // namespace algorithm
} // namespace openjij
