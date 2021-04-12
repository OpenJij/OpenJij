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

//disable Eigen warning
#include <utility/disable_eigen_warning.hpp>

#include "./compile_config.hpp"
#include "./declare.hpp"
#include <pybind11/eval.h>
#include <utility/random.hpp>
#include <type_traits>
#include <pybind11_json/pybind11_json.hpp>


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
    ::declare_Polynomial<FloatType>(m_graph, "");

    //GPU version (GPUFloatType)
    if(!std::is_same<FloatType, GPUFloatType>::value){
        ::declare_Dense<GPUFloatType>(m_graph, "GPU");
        ::declare_Sparse<GPUFloatType>(m_graph, "GPU");
        ::declare_Square<GPUFloatType>(m_graph, "GPU");
        ::declare_Chimera<GPUFloatType>(m_graph, "GPU");
    }
    else{
        //raise warning
        std::cerr << "Warning: please use classes in Graph module without suffix \"GPU\" or define type aliases." << std::endl;
    }


    /**********************************************************
    //namespace system 
     **********************************************************/
    py::module m_system = m.def_submodule("system", "cxxjij module for system");

    //ClassicalIsing
    ::declare_ClassicalIsing<graph::Dense<FloatType>>(m_system, "_Dense");
    ::declare_ClassicalIsing<graph::Sparse<FloatType>>(m_system, "_Sparse");
    ::declare_ClassicalIsingPolynomial<graph::Polynomial<FloatType>>(m_system, "_Polynomial");

    //TransverselIsing
    ::declare_TransverseIsing<graph::Dense<FloatType>>(m_system, "_Dense");
    ::declare_TransverseIsing<graph::Sparse<FloatType>>(m_system, "_Sparse");

    //Continuous Time Transeverse Ising
    ::declare_ContinuousTimeIsing<graph::Sparse<FloatType>>(m_system, "_Sparse");

#ifdef USE_CUDA
    //ChimeraTransverseGPU
    ::declare_ChimeraTranseverseGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>(m_system);
    //ChimeraClassicalGPU
    ::declare_ChimeraClassicalGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL>(m_system);
#endif

    /**********************************************************
    //namespace algorithm
     **********************************************************/
    py::module m_algorithm = m.def_submodule("algorithm", "cxxjij module for algorithm");

    //singlespinflip
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsing<graph::Dense<FloatType>>,    RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsing<graph::Sparse<FloatType>>,   RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::ClassicalIsingPolynomial<graph::Polynomial<FloatType>>,   RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::TransverseIsing<graph::Dense<FloatType>>,   RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<updater::SingleSpinFlip, system::TransverseIsing<graph::Sparse<FloatType>>,  RandomEngine>(m_algorithm, "SingleSpinFlip");

    //swendsen-wang
    ::declare_Algorithm_run<updater::SwendsenWang, system::ClassicalIsing<graph::Sparse<FloatType>>, RandomEngine>(m_algorithm, "SwendsenWang");

    //Continuous time swendsen-wang
    ::declare_Algorithm_run<updater::ContinuousTimeSwendsenWang, system::ContinuousTimeIsing<graph::Sparse<FloatType>>, RandomEngine>(m_algorithm, "ContinuousTimeSwendsenWang");

#ifdef USE_CUDA
    //GPU
    ::declare_Algorithm_run<updater::GPU, system::ChimeraTransverseGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>, utility::cuda::CurandWrapper<GPUFloatType, GPURandomEngine>>(m_algorithm, "GPU");
    ::declare_Algorithm_run<updater::GPU, system::ChimeraClassicalGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL>, utility::cuda::CurandWrapper<GPUFloatType, GPURandomEngine>>(m_algorithm, "GPU");
#endif

    /**********************************************************
    //namespace utlity
     **********************************************************/
    py::module m_utility = m.def_submodule("utility", "cxxjij module for utility");

    //schedule_list
    ::declare_ClassicalUpdaterParameter(m_utility);
    ::declare_ClassicalConstraintUpdaterParameter(m_utility);
    ::declare_TransverseFieldUpdaterParameter(m_utility);

    ::declare_Schedule<system::classical_system>(m_utility, "Classical");
    ::declare_Schedule<system::classical_constraint_system>(m_utility, "ClassicalConstraint");
    ::declare_Schedule<system::transverse_field_system>(m_utility, "TransverseField");

    m_utility.def("make_classical_schedule_list", &utility::make_classical_schedule_list,
            "beta_min"_a, "beta_max"_a, "one_mc_step"_a, "num_call_updater"_a);

    m_utility.def("make_classical_constraint_schedule_list", &utility::make_classical_constraint_schedule_list,
            "lambda"_a, "beta_min"_a, "beta_max"_a, "one_mc_step"_a, "num_call_updater"_a);

    m_utility.def("make_transverse_field_schedule_list", &utility::make_transverse_field_schedule_list,
            "beta"_a, "one_mc_step"_a, "num_call_updater"_a);

    


    /**********************************************************
    //namespace result
     **********************************************************/
    py::module m_result = m.def_submodule("result", "cxxjij module for result");

    ::declare_get_solution<system::ClassicalIsing<graph::Dense<FloatType>>>(m_result);
    ::declare_get_solution<system::ClassicalIsing<graph::Sparse<FloatType>>>(m_result);
    ::declare_get_solution<system::ClassicalIsingPolynomial<graph::Polynomial<FloatType>>>(m_result);
    ::declare_get_solution<system::TransverseIsing<graph::Dense<FloatType>>>(m_result);
    ::declare_get_solution<system::TransverseIsing<graph::Sparse<FloatType>>>(m_result);
    ::declare_get_solution<system::ContinuousTimeIsing<graph::Sparse<FloatType>>>(m_result);
#ifdef USE_CUDA
    ::declare_get_solution<system::ChimeraTransverseGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>>(m_result);
    ::declare_get_solution<system::ChimeraClassicalGPU<GPUFloatType, BLOCK_ROW, BLOCK_COL>>(m_result);
#endif
}





























