#ifndef OPENJIJ_SYSTEM_SCHEDULE_LIST_HPP__
#define OPENJIJ_SYSTEM_SCHEDULE_LIST_HPP__

#include "system_traits.hpp"

#include <tuple>
#include <vector>

namespace openjij {
    namespace system {
        template<typename SystemType>
        struct schedule_list_traits;

        template<>
        struct schedule_list_traits<classical_system> {
            using parameter_type = std::pair<size_t, double>;
            using schedule_type = std::pair<size_t, parameter_type>;
            using schedule_list_type = std::vector<schedule_type>;
        };

        template<>
        struct schedule_list_traits<quantum_system> {
            using parameter_type = std::tuple<size_t, double, double>;
            using schedule_type = std::pair<size_t, parameter_type>;
            using schedule_list_type = std::vector<schedule_type>;
        };

        template<typename System>
        using ScheduleList = typename schedule_list_traits<typename get_system_type<System>::type>::schedule_list_type;

        template<typename System>
        using ScheduleParameter = typename schedule_list_traits<typename get_system_type<System>::type>::parameter_type;

        using ClassicalScheduleList = typename schedule_list_traits<classical_system>::schedule_list_type;
        using QuantumScheduleList = typename schedule_list_traits<quantum_system>::schedule_list_type;
    } // namespace system
} // namespace openjij

#endif
