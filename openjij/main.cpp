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


#include "../src/graph/dense.h"
#include "../src/graph/sparse.h"
#include "../src/graph/square.h"
#include "../src/graph/chimera.h"
#include "../src/method/classical_ising.h"
#include "../src/method/quantum_ising.h"
#include "../src/updater/classical_updater.h"
#include "../src/updater/quantum_updater.h"
#include "../src/algorithm/sa.h"
#include "../src/algorithm/sqa.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

using namespace py::literals;
using namespace openjij;

PYBIND11_MODULE(cxxjij, m){
	m.doc() = "openjij is framework of ising and qubo";

	//graph 
	py::class_<graph::Graph>(m, "Graph")
		.def(py::init<size_t>(), "num_spins"_a)
		.def("gen_spin", &graph::Graph::gen_spin, "random_initialize"_a=true)
		.def("get_num_spins", &graph::Graph::get_num_spins);

	py::class_<graph::Dense<double>, graph::Graph>(m, "Dense")
		.def(py::init<size_t>())
		.def(py::init<const graph::Dense<double>&>(), "other"_a)
		.def(py::init<const graph::Sparse<double>&>(), "other"_a)
		.def("adj_nodes", &graph::Dense<double>::adj_nodes)
		.def("calc_energy", &graph::Dense<double>::calc_energy)
		.def("set_J", [](graph::Dense<double>& self, size_t i, size_t j, double val){ self.J(i, j) = val; return;}, "i"_a, "j"_a, "val"_a)
		.def("get_J", [](const graph::Dense<double>& self, size_t i, size_t j){ return self.J(i, j);}, "i"_a, "j"_a)
		.def("set_h", [](graph::Dense<double>& self, size_t i, double val){ self.h(i) = val; return;}, "i"_a, "val"_a)
		.def("get_h", [](const graph::Dense<double>& self, size_t i){ return self.h(i);}, "i"_a);

	py::class_<graph::Sparse<double>, graph::Graph>(m, "Sparse")
		.def(py::init<size_t, size_t>())
		.def(py::init<size_t>())
		.def(py::init<const graph::Sparse<double>&>(), "other"_a)
		.def(py::init<const graph::Dense<double>&>(), "other"_a)
		.def("adj_nodes", &graph::Sparse<double>::adj_nodes)
		.def("get_num_edges", &graph::Sparse<double>::get_num_edges)
		.def("calc_energy", &graph::Sparse<double>::calc_energy)
		.def("set_J", [](graph::Sparse<double>& self, size_t i, size_t j, double val){ self.J(i, j) = val; return;}, "i"_a, "j"_a, "val"_a)
		.def("get_J", [](const graph::Sparse<double>& self, size_t i, size_t j){ return self.J(i, j);}, "i"_a, "j"_a)
		.def("set_h", [](graph::Sparse<double>& self, size_t i, double val){ self.h(i) = val; return;}, "i"_a, "val"_a)
		.def("get_h", [](const graph::Sparse<double>& self, size_t i){ return self.h(i);}, "i"_a);

	//TODO: Square, Chimera
	
	//algorithm
	py::class_<method::ClassicalIsing>(m, "ClassicalIsing")
		.def(py::init<const graph::Dense<double>&>(), "other"_a)
		.def(py::init<const graph::Sparse<double>&>(), "other"_a)
		.def("simulated_annealing", &method::ClassicalIsing::simulated_annealing, "beta_min"_a, "beta_max"_a, "step_length"_a, "step_num"_a, "algo"_a="")
		.def("get_spins", &method::ClassicalIsing::get_spins);

	py::class_<method::QuantumIsing>(m, "QuantumIsing")
		.def(py::init<const graph::Dense<double>&, size_t>(), "other"_a, "num_trotter_slices"_a)
		.def(py::init<const graph::Sparse<double>&, size_t>(), "other"_a, "num_trotter_slices"_a)
		.def("simulated_quantum_annealing", &method::QuantumIsing::simulated_quantum_annealing, "beta"_a, "gamma_min"_a, "gamma_max"_a, "step_length"_a, "step_num"_a, "algo"_a="")
		.def("get_spins", &method::QuantumIsing::get_spins);
}


// #include <pybind11/pybind11.h>

// int add(int i, int j) {
//     return i + j;
// }

// namespace py = pybind11;

// PYBIND11_MODULE(cmake_example, m) {
//     m.doc() = R"pbdoc(
//         Pybind11 example plugin
//         -----------------------
//         .. currentmodule:: cmake_example
//         .. autosummary::
//            :toctree: _generate
//            add
//            subtract
//     )pbdoc";

//     m.def("add", &add, R"pbdoc(
//         Add two numbers
//         Some other explanation about the add function.
//     )pbdoc");

//     m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
//         Subtract two numbers
//         Some other explanation about the subtract function.
//     )pbdoc");

// #ifdef VERSION_INFO
//     m.attr("__version__") = VERSION_INFO;
// #else
//     m.attr("__version__") = "dev";
// #endif



