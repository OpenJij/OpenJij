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

#ifdef USE_CUDA
#endif

namespace openjij {

    namespace py = pybind11;

    using namespace py::literals;
    using namespace openjij;

    //system

    template<typename GraphType, bool eigen_impl>
        inline void declare_ClassicalIsing(py::module &m, const std::string& gtype_str, const std::string& eigen_str){
            //ClassicalIsing
            using ClassicalIsing = system::ClassicalIsing<GraphType, eigen_impl>;
            py::class_<ClassicalIsing>(m, "ClassicalIsing_"+gtype_str+"_"+eigen_str)
                .def(py::init<const graph::Spins&, const GraphType&>(), "init_spin"_a, "init_interaction"_a)
                .def_readwrite("spin", &ClassicalIsing_Dense::spin)
                .def_readonly("interaction", &ClassicalIsing_Dense::interaction);
        }
} // namespace openjij

#endif
