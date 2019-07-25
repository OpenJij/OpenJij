//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef OPENJIJ_DECLARE_HPP__
#define OPENJIJ_DECLARE_HPP__

#include <graph/all.hpp>
#include <system/all.hpp>
#include <updater/all.hpp>
#include <algorithm/algorithm.hpp>
#include <result/get_solution.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using namespace py::literals;
using namespace openjij;

//system

//ClassicalIsing
template<typename GraphType, bool eigen_impl>
inline void declare_ClassicalIsing(py::module &m, const std::string& gtype_str, const std::string& eigen_str){
    //ClassicalIsing
    using ClassicalIsing = system::ClassicalIsing<GraphType, eigen_impl>;

    auto str = std::string("ClassicalIsing")+gtype_str+eigen_str;
    py::class_<ClassicalIsing>(m, str.c_str())
        .def(py::init<const graph::Spins&, const GraphType&>(), "init_spin"_a, "init_interaction"_a)
        .def_readwrite("spin", &ClassicalIsing::spin)
        .def_readonly("interaction", &ClassicalIsing::interaction)
        .def_readonly("num_spin", &ClassicalIsing::num_spins);

    //make_classical_ising
    auto mkci_str = std::string("make_classical_ising")+eigen_str;
    m.def(mkci_str.c_str(), [](const graph::Spins& init_spin, const GraphType& init_interaction){
            return system::make_classical_ising<eigen_impl>(init_spin, init_interaction);
            }, "init_spin"_a, "init_interaction"_a);
}


//TransverseIsing
template<typename GraphType, bool eigen_impl>
inline void declare_TransverseIsing(py::module &m, const std::string& gtype_str, const std::string& eigen_str){
    //TransverseIsing
    using TransverseIsing = system::TransverseIsing<GraphType, eigen_impl>;
    using FloatType = typename GraphType::value_type;

    auto str = std::string("TransverseIsing")+gtype_str+eigen_str;
    py::class_<TransverseIsing>(m, str.c_str())
        .def(py::init<const system::TrotterSpins&, const GraphType&, FloatType>(), "init_spin"_a, "init_interaction"_a, "gamma"_a)
        .def(py::init<const graph::Spins&, const GraphType&, FloatType, size_t>(), "init_classical_spins"_a, "init_interaction"_a, "gamma"_a, "num_trotter_slices"_a)
        .def_readwrite("trotter_spins", &TransverseIsing::trotter_spins)
        .def_readonly("interaction", &TransverseIsing::interaction)
        .def_readonly("num_classical_spins", &TransverseIsing::num_classical_spins)
        .def_readwrite("gamma", &TransverseIsing::gamma);

    //make_transverse_ising
    auto mkci_str = std::string("make_transverse_ising")+eigen_str;
    m.def(mkci_str.c_str(), [](const system::TrotterSpins& init_trotter_spins, const GraphType& init_interaction, double gamma){
            return system::make_transverse_ising<eigen_impl>(init_trotter_spins, init_interaction, gamma);
            }, "init_trotter_spins"_a, "init_interaction"_a, "gamma"_a);

    m.def(mkci_str.c_str(), [](const graph::Spins& classical_spins, const GraphType& init_interaction, double gamma, std::size_t num_trotter_slices){
            return system::make_transverse_ising<eigen_impl>(classical_spins, init_interaction, gamma, num_trotter_slices);
            }, "classical_spins"_a, "init_interaction"_a, "gamma"_a, "num_trotter_slices"_a);
}

#ifdef USE_CUDA

//ChimeraTransverseGPU
template<typename FloatType,
    std::size_t rows_per_block,
    std::size_t cols_per_block,
    std::size_t trotters_per_block>
    inline void declare_ChimeraTranseverseGPU(py::module &m){
        using ChimeraTransverseGPU = system::ChimeraTransverseGPU<FloatType, rows_per_block, cols_per_block, trotters_per_block>;
        py::class_<ChimeraTransverseGPU>(m, "ChimeraTransverseGPU")
            .def(py::init<const system::TrotterSpins&, const graph::Chimera<FloatType>&, FloatType, int>(), "init_trotter_spins"_a, "init_interaction"_a, "gamma"_a, "device_num"_a=0)
            .def(py::init<const graph::Spins&, const graph::Chimera<FloatType>&, FloatType, size_t, int>(), "classical_spins"_a, "init_interaction"_a, "gamma"_a, "num_trotter_slices"_a, "device_num"_a=0)
            .def_readwrite("gamma", &ChimeraTransverseGPU::gamma);

        //make_chimera_transverse_gpu
        m.def("make_chimera_transverse_gpu", [](const system::TrotterSpins& init_trotter_spins, const graph::Chimera<FloatType>& init_interaction, double gamma, int device_num){
                return system::make_chimera_transverse_gpu<rows_per_block, cols_per_block, trotters_per_block>(init_trotter_spins, init_interaction, gamma, device_num);
                }, "init_trotter_spins"_a, "init_interaction"_a, "gamma"_a, "device_num"_a=0);

        m.def("make_chimera_transverse_gpu", [](const graph::Spins& classical_spins, const graph::Chimera<FloatType>& init_interaction, double gamma, size_t num_trotter_slices, int device_num){
                return system::make_chimera_transverse_gpu<rows_per_block, cols_per_block, trotters_per_block>(classical_spins, init_interaction, gamma, num_trotter_slices, device_num);
                }, "classical_spins"_a, "init_interaction"_a, "gamma"_a, "num_trotter_slices"_a, "device_num"_a=0);
    }

#endif

//Algorithm
template<template<typename> class Updater, typename System, typename RandomNumberEngine>
inline void declare_Algorithm_run(py::module &m, const std::string& updater_str){
//with seed
    m.def(std::string("Algorithm_")+updater_str+std::string("_run"), [](System& system, std::size_t seed, const utility::ScheduleList<typename system::get_system_type<System>::type>& schedule_list){
            RandomEngine rng(seed);
            algorithm::Algorithm<Updater>::run(system, rng, schedule_list);
            }, "system"_a, "seed"_a, "schedule_list"_a);

//without seed
    m.def(std::string("Algorithm_")+updater_str+std::string("_run"), [](System& system, const utility::ScheduleList<typename system::get_system_type<System>::type>& schedule_list){
            RandomEngine rng(std::random_device{}());
            algorithm::Algorithm<Updater>::run(system, rng, schedule_list);
            }, "system"_a, "schedule_list"_a);
}

//result
//get_solution



#endif
