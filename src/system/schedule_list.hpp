#ifndef OPENJIJ_SYSTEM_SCHEDULE_LIST_HPP__
#define OPENJIJ_SYSTEM_SCHEDULE_LIST_HPP__

#include <tuple>
#include <vector>

#include "system_traits.hpp"
#include "../utility/create_geometric_progression.hpp"

namespace openjij {
    namespace system {
        template<typename SystemType>
        struct schedule_list_traits;

        template<>
        struct schedule_list_traits<classical_system> {
            using parameter_type = double;
            using schedule_type = std::pair<size_t, parameter_type>;
            using schedule_list_type = std::vector<schedule_type>;
        };

        template<>
        struct schedule_list_traits<quantum_system> {
            using parameter_type = std::pair<double, double>;
            using schedule_type = std::pair<size_t, parameter_type>;
            using schedule_list_type = std::vector<schedule_type>;
        };

        template<typename System>
        using ScheduleList = typename schedule_list_traits<typename get_system_type<System>::type>::schedule_list_type;

        template<typename System>
        using ScheduleParameter = typename schedule_list_traits<typename get_system_type<System>::type>::parameter_type;

        using ClassicalScheduleList = typename schedule_list_traits<classical_system>::schedule_list_type;
        using QuantumScheduleList = typename schedule_list_traits<quantum_system>::schedule_list_type;

        // helper function
        ClassicalScheduleList create_sa_schedule_list(const size_t total_mc_step, const size_t one_mc_step, const double beta_min, const double beta_max) {
            const auto beta_list = [&total_mc_step, &beta_min, &beta_max](){
                auto beta_list = std::vector<double>(total_mc_step);
                const double ratio_beta = std::pow(beta_max/beta_min, 1.0/ static_cast<const double>(total_mc_step - 1));
                utility::make_geometric_progression(beta_list.begin(), beta_list.end(), beta_min, ratio_beta);
                return beta_list;
            }();

            auto schedule_list = ClassicalScheduleList(total_mc_step);
            for (size_t i = 0; i < total_mc_step; i++) {
                schedule_list[i] = std::make_pair(one_mc_step, beta_list[i]);
            }
            return schedule_list;
        }
    } // namespace system
} // namespace openjij

#endif
