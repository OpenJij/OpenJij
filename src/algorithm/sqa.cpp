#include "sqa.h"
#include <cmath>

namespace openjij {
    namespace algorithm {

        SQA::SQA(const double beta, const double gamma, const Schedule& schedule)
            :beta(beta), gamma(gamma), schedule(schedule) {
        }

        SQA::SQA(const double beta, const double gamma, const size_t step_length, const size_t step_num)
            :beta(beta), gamma(gamma){
            // linear annealing schedule
            const double s_d = 1.0 / (step_num+1);
            double s = 0.0;

            for (size_t i = 0; i < step_num-1; i++) {
                schedule.emplace_back(std::make_pair(s, step_length));
                s += s_d;
            }
        }

        void SQA::run(updater::QuantumUpdater& updater, const std::string& algo) {
            // anneal
            for (const auto& elem : schedule) {
                const double s = elem.first;
                const size_t step_length = elem.second;

                // step length -> number of MCS for each step
                for (size_t i = 0; i < step_length; i++) {
                    // update
                    updater.update(beta, gamma, s, algo);
                }
            }
        }
    } // namespace algorithm
} // namespace openjij

