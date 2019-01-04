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


#include "../src/sampler/sampler.h"
#include "../src/model.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

PYBIND11_MODULE(cxxjij, m){
	m.doc() = "openjij is framework of ising and qubo";

	py::class_<openjij::SquareMatrix<double>>(m, "SquareMatrix")
		.def(py::init<int, double>())
		.def("__call__", [](openjij::SquareMatrix<double>& sqm, int x, int y){ return sqm(x, y); })
		.def("set", &openjij::SquareMatrix<double>::set);

	py::class_<openjij::sampler::Results>(m, "Results")
		.def(py::init<>())
		.def_readwrite("states", &openjij::sampler::Results::states)
		.def_readwrite("quantum_states", &openjij::sampler::Results::quantum_states)
		.def_readwrite("energies", &openjij::sampler::Results::energies);

	py::class_<openjij::sampler::Sampler>(m, "Sampler")
		.def(py::init<const openjij::SquareMatrix<double>&>())
		.def("sampling", &openjij::sampler::Sampler::sampling)
		.def("quantum_sampling", &openjij::sampler::Sampler::quantum_sampling)
		.def("simulated_annealing", &openjij::sampler::Sampler::simulated_annealing)
		.def("simulated_quantum_annealing", &openjij::sampler::Sampler::simulated_quantum_annealing);
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



