#include "sqa.h"
#include <cmath>

namespace openjij {
    namespace algorithm {

        SQA::SQA(const double beta, const Schedule& schedule)
            :beta(beta), schedule(schedule) {
        }

        SQA::SQA(const double beta, const double gamma_min, const double gamma_max, const size_t step_length, const size_t step_num)
            :beta(beta) {
            const double r_gamma = std::pow(gamma_min/gamma_max, 1.0/step_num);
            double gamma = gamma_max;

            for (size_t i = 0; i < step_num; i++) {
                schedule.emplace_back(std::make_pair(gamma, step_length));
                gamma *= r_gamma;
            }
        }

        void SQA::run(updater::QuantumUpdater& updater, const std::string& algo) {
            // anneal
            for (const auto& elem : schedule) {
                const double gamma = elem.first;
                const size_t step_length = elem.second;

                // step length -> number of MCS for each step
                for (size_t i = 0; i < step_length; i++) {
                    // update
                    updater.update(beta, gamma, algo);
                }
            }
        }
    } // namespace algorithm
} // namespace openjij

