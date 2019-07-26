//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software //    distributed under the License is distributed on an "AS IS" BASIS, //    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.


#include "./compile_config.hpp"
#include "./declare.hpp"
#include <utility/random.hpp>


PYBIND11_MODULE(cxxjij, m){
    m.doc() = "openjij is a framework for ising and qubo";

    /**********************************************************
    //namespace graph
     **********************************************************/
    py::module m_graph = m.def_submodule("graph", "cxxjij submodule for graph");

    ::declare_Graph(m_graph);

    ::declare_Dir(m_graph);
    ::declare_ChimeraDir(m_graph);

    //CPU version (FloatType)
    ::declare_Dense<FloatType>(m_graph, "");
    ::declare_Sparse<FloatType>(m_graph, "");
    ::declare_Square<FloatType>(m_graph, "");
    ::declare_Chimera<FloatType>(m_graph, "");

    //GPU version (GPUFloatType)
    ::declare_Dense<GPUFloatType>(m_graph, "GPU");
    ::declare_Sparse<GPUFloatType>(m_graph, "GPU");
    ::declare_Square<GPUFloatType>(m_graph, "GPU");
    ::declare_Chimera<GPUFloatType>(m_graph, "GPU");


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

#ifdef USE_CUDA
    //ChimeraTransverseGPU
    ::declare_ChimeraTranseverseGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>(m_system);
#endif

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

#ifdef USE_CUDA
    //GPU
    ::declare_Algorithm_run<updater::GPU, system::ChimeraTransverseGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>, utility::cuda::CurandWrapper<GPUFloatType, GPURandomEngine>>(m_algorithm, "GPU");
#endif

    /**********************************************************
    //namespace utlity
     **********************************************************/
    py::module m_utility = m.def_submodule("utility", "cxxjij module for utility");

    //schedule_list
    py::class_<utility::ClassicalUpdaterParameter>(m_utility, "ClassicalUpdaterParameter")
        .def(py::init<>())
        .def(py::init<double>(), "beta"_a)
        .def_readwrite("beta", &utility::ClassicalUpdaterParameter::beta)
        .def("__repr__", [](const utility::ClassicalUpdaterParameter& self){
                return repr_impl(self);
                });

    py::class_<utility::TransverseFieldUpdaterParameter>(m_utility, "TransverseFieldUpdaterParameter")
        .def(py::init<>())
        .def(py::init<double, double>(), "beta"_a, "s"_a)
        .def_readwrite("beta", &utility::TransverseFieldUpdaterParameter::beta)
        .def_readwrite("s", &utility::TransverseFieldUpdaterParameter::s)
        .def("__repr__", [](const utility::TransverseFieldUpdaterParameter& self){
                return repr_impl(self);
                });

    ::declare_Schedule<system::classical_system>(m_utility, "Classical");
    ::declare_Schedule<system::transverse_field_system>(m_utility, "TransverseField");

    m_utility.def("make_classical_schedule_list", &utility::make_classical_schedule_list,
            "beta_min"_a, "beta_max"_a, "one_mc_step"_a, "num_call_updater"_a);

    m_utility.def("make_transverse_field_schedule_list", &utility::make_transverse_field_schedule_list,
            "beta"_a, "one_mc_step"_a, "num_call_updater"_a);

    


    /**********************************************************
    //namespace result
     **********************************************************/
    py::module m_result = m.def_submodule("result", "cxxjij module for result");

    ::declare_get_solution<system::ClassicalIsing<graph::Dense<FloatType>, false>>(m_result);
    ::declare_get_solution<system::ClassicalIsing<graph::Dense<FloatType>, true>>(m_result);
    ::declare_get_solution<system::ClassicalIsing<graph::Sparse<FloatType>, false>>(m_result);
    ::declare_get_solution<system::ClassicalIsing<graph::Sparse<FloatType>, true>>(m_result);
    ::declare_get_solution<system::TransverseIsing<graph::Dense<FloatType>, false>>(m_result);
    ::declare_get_solution<system::TransverseIsing<graph::Dense<FloatType>, true>>(m_result);
    ::declare_get_solution<system::TransverseIsing<graph::Sparse<FloatType>, false>>(m_result);
    ::declare_get_solution<system::TransverseIsing<graph::Sparse<FloatType>, true>>(m_result);
    ::declare_get_solution<system::ChimeraTransverseGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>>(m_result);
}





























