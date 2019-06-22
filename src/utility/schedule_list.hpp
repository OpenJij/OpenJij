#ifndef OPENJIJ_UTILITY_SCHEDULE_LIST_HPP__
#define OPENJIJ_UTILITY_SCHEDULE_LIST_HPP__

#include <cmath>
#include <vector>
#include <tuple>

namespace openjij {
    namespace utility {
        using ScheduleList = std::vector<std::pair<std::size_t, double>>;

        ScheduleList make_schedule_list(double beta_min, double beta_max, std::size_t one_mc_step, std::size_t num_call_update) noexcept {
            const double r_beta = std::pow(beta_max/beta_min, 1.0/static_cast<double>(num_call_update-1));
            double beta = beta_min;

            auto schedule_list = ScheduleList(num_call_update);
            for (auto first = schedule_list.begin(), last = schedule_list.end(); first != last; ++first) {
                *first = std::make_pair(one_mc_step, beta);
                beta *= r_beta;
            }

            return schedule_list;
        }
    } // namespace utility
} // namespace openjij

#endif
