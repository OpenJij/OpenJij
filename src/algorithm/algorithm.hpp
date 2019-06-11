#ifndef SYSTEM_ALGORITHM_ALGORITHM_HPP__
#define SYSTEM_ALGORITHM_ALGORITHM_HPP__

#include <iostream>
#include <tuple>
#include <vector>
// #include "../system/schedule_list.hpp"

namespace openjij {
    namespace algorithm {
        template<template<typename> class Updater>
        struct Algorithm {
            template<typename System>
            static void run(System& system, const std::vector<double>& schedule_list) {
                for (auto&& schedule : schedule_list) {
                    const auto parameter = schedule;
                    const auto one_mc_step = 5;

                    for (auto i = 0; i < one_mc_step; ++i) {
                    std::cout << "one_mc_step: " << one_mc_step << std::endl;
                        Updater<System>::update(system, parameter);
                    }
                    std::cout << std::endl;
                }
            }
        };
    } // namespace algorithm
} // namespace openjij

#endif
