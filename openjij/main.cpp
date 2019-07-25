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


#include "./compile_config.hpp"
#include "./declare.hpp"


PYBIND11_MODULE(cxxjij, m){
	m.doc() = "openjij is a framework for ising and qubo";

    /**********************************************************
    //namespace graph
     **********************************************************/
 	py::module m_graph = m.def_submodule("graph", "cxxjij submodule for graph");
    
 	//graph 
 	py::class_<graph::Graph>(m_graph, "Graph")
 		.def(py::init<std::size_t>(), "num_spins"_a)
 		.def("gen_spin", [](const graph::Graph& self, std::size_t seed){
                RandomEngine rng(seed);
                return self.gen_spin(rng);
                }, "seed"_a)
 		.def("gen_spin", [](const graph::Graph& self){
                RandomEngine rng(std::random_device{}());
                return self.gen_spin(rng);
                })
 		.def("size", &graph::Graph::size);

 	//dense
 	py::class_<graph::Dense<FloatType>, graph::Graph>(m_graph, "Dense")
 		.def(py::init<std::size_t>(), "num_spins"_a)
 		.def(py::init<const graph::Dense<FloatType>&>(), "other"_a)
 		.def("adj_nodes", &graph::Dense<FloatType>::adj_nodes)
 		.def("calc_energy", &graph::Dense<FloatType>::calc_energy, "spins"_a)
 		.def("__setitem__", [](graph::Dense<FloatType>& self, const std::pair<std::size_t, std::size_t>& key, FloatType val){self.J(key.first, key.second) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Dense<FloatType>& self, const std::pair<std::size_t, std::size_t>& key){return self.J(key.first, key.second);}, "key"_a)
 		.def("__setitem__", [](graph::Dense<FloatType>& self, std::size_t key, FloatType val){self.h(key) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Dense<FloatType>& self, std::size_t key){return self.h(key);}, "key"_a);

 	//sparse
 	py::class_<graph::Sparse<FloatType>, graph::Graph>(m_graph, "Sparse")
 		.def(py::init<std::size_t, std::size_t>(), "num_spins"_a, "num_edges"_a)
 		.def(py::init<std::size_t>(),  "num_spins"_a)
 		.def(py::init<const graph::Sparse<FloatType>&>(), "other"_a)
 		.def("adj_nodes", &graph::Sparse<FloatType>::adj_nodes)
 		.def("get_num_edges", &graph::Sparse<FloatType>::get_num_edges)
 		.def("calc_energy", &graph::Sparse<FloatType>::calc_energy, "spins"_a)
 		.def("__setitem__", [](graph::Sparse<FloatType>& self, const std::pair<std::size_t, std::size_t>& key, FloatType val){self.J(key.first, key.second) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Sparse<FloatType>& self, const std::pair<std::size_t, std::size_t>& key){return self.J(key.first, key.second);}, "key"_a)
 		.def("__setitem__", [](graph::Sparse<FloatType>& self, std::size_t key, FloatType val){self.h(key) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Sparse<FloatType>& self, std::size_t key){return self.h(key);}, "key"_a);

 	//enum class Dir
 	py::enum_<graph::Dir>(m_graph, "Dir")
 		.value("PLUS_R", graph::Dir::PLUS_R)
 		.value("MINUS_R", graph::Dir::MINUS_R)
 		.value("PLUS_C", graph::Dir::PLUS_C)
 		.value("MINUS_C", graph::Dir::MINUS_C);

 	//square
 	py::class_<graph::Square<FloatType>, graph::Sparse<FloatType>>(m_graph, "Square")
 		.def(py::init<std::size_t, std::size_t, FloatType>(), "num_row"_a, "num_column"_a, "init_val"_a=0)
 		.def(py::init<const graph::Square<FloatType>&>(), "other"_a)
 		.def("to_ind", &graph::Square<FloatType>::to_ind)
 		.def("to_rc", &graph::Square<FloatType>::to_rc)
 		.def("get_num_row", &graph::Square<FloatType>::get_num_row)
 		.def("get_num_column", &graph::Square<FloatType>::get_num_column)
 		.def("__setitem__", [](graph::Square<FloatType>& self, const std::tuple<std::size_t, std::size_t, graph::Dir>& key, FloatType val){self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key)) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Square<FloatType>& self, const std::tuple<std::size_t, std::size_t, graph::Dir>& key){return self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key));}, "key"_a)
 		.def("__setitem__", [](graph::Square<FloatType>& self, const std::pair<std::size_t, std::size_t>& key, FloatType val){self.h(key.first, key.second) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Square<FloatType>& self, const std::pair<std::size_t, std::size_t>& key){return self.h(key.first, key.second);}, "key"_a);

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
 	py::class_<graph::Chimera<FloatType>, graph::Sparse<FloatType>>(m_graph, "Chimera")
 		.def(py::init<std::size_t, std::size_t, FloatType>(), "num_row"_a, "num_column"_a, "init_val"_a=0)
 		.def(py::init<const graph::Chimera<FloatType>&>(), "other"_a)
 		.def("to_ind", &graph::Chimera<FloatType>::to_ind)
 		.def("to_rci", &graph::Chimera<FloatType>::to_rci)
 		.def("get_num_row", &graph::Chimera<FloatType>::get_num_row)
 		.def("get_num_column", &graph::Chimera<FloatType>::get_num_column)
 		.def("get_num_in_chimera", &graph::Chimera<FloatType>::get_num_in_chimera)
 		.def("__setitem__", [](graph::Chimera<FloatType>& self, const std::tuple<std::size_t, std::size_t, std::size_t, graph::ChimeraDir>& key, FloatType val){self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key)) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Chimera<FloatType>& self, const std::tuple<std::size_t, std::size_t, std::size_t, graph::ChimeraDir>& key){return self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));}, "key"_a)
 		.def("__setitem__", [](graph::Chimera<FloatType>& self, const std::tuple<std::size_t, std::size_t, std::size_t>& key, FloatType val){self.h(std::get<0>(key), std::get<1>(key), std::get<2>(key)) = val;}, "key"_a, "val"_a)
 		.def("__getitem__", [](const graph::Chimera<FloatType>& self, const std::tuple<std::size_t, std::size_t, std::size_t>& key){return self.h(std::get<0>(key), std::get<1>(key), std::get<2>(key));}, "key"_a);


    /**********************************************************
 	//namespace system 
     **********************************************************/
 	py::module m_system = m.def_submodule("system", "cxxjij module for system");

 	//ClassicalIsing (Dense, NoEigenImpl)
    ::declare_ClassicalIsing<graph::Dense<FloatType>, false>(m_system, "_Dense", "");
    ::declare_ClassicalIsing<graph::Dense<FloatType>, true>(m_system, "_Dense", "_Eigen");
    ::declare_ClassicalIsing<graph::Sparse<FloatType>, false>(m_system, "_Sparse", "");
    ::declare_ClassicalIsing<graph::Sparse<FloatType>, true>(m_system, "_Sparse", "_Eigen");

    //TransverselIsing
    ::declare_TransverseIsing<graph::Dense<FloatType>, false>(m_system, "_Dense", "");
    ::declare_TransverseIsing<graph::Dense<FloatType>, true>(m_system, "_Dense", "_Eigen");
    ::declare_TransverseIsing<graph::Sparse<FloatType>, false>(m_system, "_Sparse", "");
    ::declare_TransverseIsing<graph::Sparse<FloatType>, true>(m_system, "_Sparse", "_Eigen");

    //ChimeraTransverseGPU
    ::declare_ChimeraTranseverseGPU<FloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>(m_system);

    /**********************************************************
 	//namespace algorithm
     **********************************************************/
 	py::module m_algorithm = m.def_submodule("algorithm", "cxxjij module for algorithm");

    //singlespinflip
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsing<graph::Dense<FloatType>, false>,   RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsing<graph::Dense<FloatType>, true>,    RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsing<graph::Sparse<FloatType>, false>,  RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsing<graph::Sparse<FloatType>, true>,   RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::TransverseIsing<graph::Dense<FloatType>, false>,  RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::TransverseIsing<graph::Dense<FloatType>, true>,   RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::TransverseIsing<graph::Sparse<FloatType>, false>, RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::TransverseIsing<graph::Sparse<FloatType>, true>,  RandomEngine>(m_algorithm, "SingleSpinFlip");

    //swendsen-wang
    ::declare_Algorithm_run<updater::SwendsenWang, system::ClassicalIsing<graph::Dense<FloatType>, false>,  RandomEngine>(m_algorithm, "SwendsenWang");
    ::declare_Algorithm_run<updater::SwendsenWang, system::ClassicalIsing<graph::Sparse<FloatType>, false>, RandomEngine>(m_algorithm, "SwendsenWang");

    //GPU
    ::declare_Algorithm_run<updater::GPU, system::ChimeraTransverseGPU<FloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>, GPUFloatType>(m_algorithm, "GPU");

}


    


























