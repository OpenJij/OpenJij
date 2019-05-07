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
#include "../src/system/classical_ising.h"
#include "../src/system/quantum_ising.h"
#include "../src/updater/classical_updater.h"
#include "../src/updater/quantum_updater.h"
#include "../src/algorithm/sa.h"
#include "../src/algorithm/sqa.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include "../src/system/chimera_gpu/chimera_gpu_quantum.h"
#endif


namespace py = pybind11;

using namespace py::literals;
using namespace openjij;

PYBIND11_MODULE(cxxjij, m){
	m.doc() = "openjij is framework of ising and qubo";

	py::module m_graph = m.def_submodule("graph", "A submodule of cxxjij");

	//graph 
	py::class_<graph::Graph>(m_graph, "Graph")
		.def(py::init<size_t>(), "num_spins"_a)
		.def("gen_spin", [](const graph::Graph& self, uint_fast32_t seed){return self.gen_spin(seed);})
		.def("gen_spin", [](const graph::Graph& self){return self.gen_spin();})
		.def("get_num_spins", &graph::Graph::get_num_spins);

	//dense
	py::class_<graph::Dense<double>, graph::Graph>(m_graph, "Dense")
		.def(py::init<size_t>(), "num_spins"_a)
		.def(py::init<const graph::Dense<double>&>(), "other"_a)
		.def(py::init<const graph::Sparse<double>&>(), "other"_a)
		.def("adj_nodes", &graph::Dense<double>::adj_nodes)
		.def("calc_energy", &graph::Dense<double>::calc_energy)
		.def("__setitem__", [](graph::Dense<double>& self, const std::pair<size_t, size_t>& key, double val){self.J(key.first, key.second) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Dense<double>& self, const std::pair<size_t, size_t>& key){return self.J(key.first, key.second);}, "key"_a)
		.def("__setitem__", [](graph::Dense<double>& self, size_t key, double val){self.h(key) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Dense<double>& self, size_t key){return self.h(key);}, "key"_a);

	//sparse
	py::class_<graph::Sparse<double>, graph::Graph>(m_graph, "Sparse")
		.def(py::init<size_t, size_t>(), "num_spins"_a, "num_edges"_a)
		.def(py::init<size_t>(),  "num_spins"_a)
		.def(py::init<const graph::Sparse<double>&>(), "other"_a)
		.def(py::init<const graph::Dense<double>&>(), "other"_a)
		.def("adj_nodes", &graph::Sparse<double>::adj_nodes)
		.def("get_num_edges", &graph::Sparse<double>::get_num_edges)
		.def("calc_energy", &graph::Sparse<double>::calc_energy)
		.def("__setitem__", [](graph::Sparse<double>& self, const std::pair<size_t, size_t>& key, double val){self.J(key.first, key.second) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Sparse<double>& self, const std::pair<size_t, size_t>& key){return self.J(key.first, key.second);}, "key"_a)
		.def("__setitem__", [](graph::Sparse<double>& self, size_t key, double val){self.h(key) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Sparse<double>& self, size_t key){return self.h(key);}, "key"_a);

	//enum class Dir
	py::enum_<graph::Dir>(m_graph, "Dir")
		.value("PLUS_R", graph::Dir::PLUS_R)
		.value("MINUS_R", graph::Dir::MINUS_R)
		.value("PLUS_C", graph::Dir::PLUS_C)
		.value("MINUS_C", graph::Dir::MINUS_C);

	//square
	py::class_<graph::Square<double>, graph::Sparse<double>>(m_graph, "Square")
		.def(py::init<size_t, size_t, double>(), "num_row"_a, "num_column"_a, "init_val"_a=0)
		.def(py::init<const graph::Square<double>&>(), "other"_a)
		.def("to_ind", &graph::Square<double>::to_ind)
		.def("to_rc", &graph::Square<double>::to_rc)
		.def("get_num_row", &graph::Square<double>::get_num_row)
		.def("get_num_column", &graph::Square<double>::get_num_column)
		.def("__setitem__", [](graph::Square<double>& self, const std::tuple<size_t, size_t, graph::Dir>& key, double val){self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key)) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Square<double>& self, const std::tuple<size_t, size_t, graph::Dir>& key){return self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key));}, "key"_a)
		.def("__setitem__", [](graph::Square<double>& self, const std::pair<size_t, size_t>& key, double val){self.h(key.first, key.second) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Square<double>& self, const std::pair<size_t, size_t>& key){return self.h(key.first, key.second);}, "key"_a);

	//enum class ChimeraDir
	py::enum_<graph::ChimeraDir>(m_graph, "ChimeraDir")
		.value("PLUS_R", graph::ChimeraDir::PLUS_R)
		.value("MINUS_R", graph::ChimeraDir::MINUS_R)
		.value("PLUS_C", graph::ChimeraDir::PLUS_C)
		.value("MINUS_C", graph::ChimeraDir::MINUS_C)
		.value("IN_0or4", graph::ChimeraDir::IN_0or4)
		.value("IN_1or5", graph::ChimeraDir::IN_1or5)
		.value("IN_2or6", graph::ChimeraDir::IN_2or6)
		.value("IN_3or7", graph::ChimeraDir::IN_3or7);

	//chimera
	py::class_<graph::Chimera<double>, graph::Sparse<double>>(m_graph, "Chimera")
		.def(py::init<size_t, size_t, double>(), "num_row"_a, "num_column"_a, "init_val"_a=0)
		.def(py::init<const graph::Chimera<double>&>(), "other"_a)
		.def("to_ind", &graph::Chimera<double>::to_ind)
		.def("to_rci", &graph::Chimera<double>::to_rci)
		.def("get_num_row", &graph::Chimera<double>::get_num_row)
		.def("get_num_column", &graph::Chimera<double>::get_num_column)
		.def("get_num_in_chimera", &graph::Chimera<double>::get_num_in_chimera)
		.def("__setitem__", [](graph::Chimera<double>& self, const std::tuple<size_t, size_t, size_t, graph::ChimeraDir>& key, double val){self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key)) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Chimera<double>& self, const std::tuple<size_t, size_t, size_t, graph::ChimeraDir>& key){return self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));}, "key"_a)
		.def("__setitem__", [](graph::Chimera<double>& self, const std::tuple<size_t, size_t, size_t>& key, double val){self.h(std::get<0>(key), std::get<1>(key), std::get<2>(key)) = val;}, "key"_a, "val"_a)
		.def("__getitem__", [](const graph::Chimera<double>& self, const std::tuple<size_t, size_t, size_t>& key){return self.h(std::get<0>(key), std::get<1>(key), std::get<2>(key));}, "key"_a);


	//algorithm
	py::module m_system = m.def_submodule("system", "A submodule of cxxjij");
	
	//ClassicalIsing
	py::class_<system::ClassicalIsing>(m_system, "ClassicalIsing")
		.def(py::init<const graph::Dense<double>&>(), "other"_a)
		.def(py::init<const graph::Sparse<double>&>(), "other"_a)
		.def(py::init<const graph::Sparse<double>&, graph::Spins&>(), "other"_a, "init_state"_a)
		.def("simulated_annealing", (void (system::ClassicalIsing::*)(const double, const double, const size_t, const size_t, const std::string&)) &system::ClassicalIsing::simulated_annealing, "beta_min"_a, "beta_max"_a, "step_length"_a, "step_num"_a, "algo"_a="")
		.def("simulated_annealing", (void (system::ClassicalIsing::*)(const std::vector<std::pair<double, size_t>>&, const std::string&)) &system::ClassicalIsing::simulated_annealing, "schedule"_a, "algo"_a="")
		.def("get_spins", &system::ClassicalIsing::get_spins)
		.def("initialize_spins", &system::ClassicalIsing::initialize_spins)
		.def("set_spins", &system::ClassicalIsing::set_spins, "initial_state"_a);

	//QuantumIsing
	py::class_<system::QuantumIsing>(m_system, "QuantumIsing")
		.def(py::init<const graph::Dense<double>&, size_t>(), "other"_a, "num_trotter_slices"_a)
		.def(py::init<const graph::Sparse<double>&, size_t>(), "other"_a, "num_trotter_slices"_a)
		.def(py::init<const graph::Sparse<double>&, size_t, graph::Spins&>(), "other"_a, "num_trotter_slices"_a, "init_state"_a)
		.def("simulated_quantum_annealing", (void (system::QuantumIsing::*)(const double, const double, const size_t, const size_t, const std::string&)) &system::QuantumIsing::simulated_quantum_annealing, "beta"_a, "gamma"_a, "step_length"_a, "step_num"_a, "algo"_a="")
		.def("simulated_quantum_annealing", (void (system::QuantumIsing::*)(const double, const double, const std::vector<std::pair<double, size_t>>&, const std::string&)) &system::QuantumIsing::simulated_quantum_annealing, "beta"_a, "gamma"_a, "schedule"_a, "algo"_a="")
		.def("get_spins", &system::QuantumIsing::get_spins)
		.def("initialize_spins", &system::QuantumIsing::initialize_spins)
		.def("set_spins", &system::QuantumIsing::set_spins, "initial_state"_a);

#ifdef USE_CUDA
	py::class_<system::ChimeraGPUQuantum>(m_system, "ChimeraGPUQuantum")
		.def(py::init<const graph::Chimera<double>&, size_t, int>(), "other"_a, "num_trotter_slices"_a, "gpudevice"_a=0)
		.def("simulated_quantum_annealing", (void (system::ChimeraGPUQuantum::*)(const double, const double, const size_t, const size_t, const std::string&)) &system::ChimeraGPUQuantum::simulated_quantum_annealing, "beta"_a, "gamma"_a, "step_length"_a, "step_num"_a, "algo"_a="")
		.def("simulated_quantum_annealing", (void (system::ChimeraGPUQuantum::*)(const double, const double, const std::vector<std::pair<double, size_t>>&, const std::string&)) &system::ChimeraGPUQuantum::simulated_quantum_annealing, "beta"_a, "gamma"_a, "schedule"_a, "algo"_a="")
		.def("get_spins", &system::ChimeraGPUQuantum::get_spins);
#endif
}

