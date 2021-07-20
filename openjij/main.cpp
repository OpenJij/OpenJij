//    Copyright 2021 Jij Inc.

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

    //CPU version (openjij::FloatType)
    ::declare_Dense<openjij::FloatType>(m_graph, "");
    ::declare_Sparse<openjij::FloatType>(m_graph, "");
    ::declare_Square<openjij::FloatType>(m_graph, "");
    ::declare_Chimera<openjij::FloatType>(m_graph, "");
    ::declare_Polynomial<openjij::FloatType>(m_graph, "");

    //GPU version (openjij::GPUFloatType)
    if(!std::is_same<openjij::FloatType, openjij::GPUFloatType>::value){
        ::declare_Dense<openjij::GPUFloatType>(m_graph, "GPU");
        ::declare_Sparse<openjij::GPUFloatType>(m_graph, "GPU");
        ::declare_Square<openjij::GPUFloatType>(m_graph, "GPU");
        ::declare_Chimera<openjij::GPUFloatType>(m_graph, "GPU");
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
    ::declare_ClassicalIsing<openjij::graph::Dense<openjij::FloatType>>(m_system, "_Dense");
    ::declare_ClassicalIsing<openjij::graph::Sparse<openjij::FloatType>>(m_system, "_Sparse");
    ::declare_ClassicalIsingPolynomial<openjij::graph::Polynomial<openjij::FloatType>>(m_system, "_Polynomial");
    ::declare_KLocalPolynomial<openjij::graph::Polynomial<openjij::FloatType>>(m_system, "_Polynomial");

    //TransverselIsing
    ::declare_TransverseIsing<openjij::graph::Dense<openjij::FloatType>>(m_system, "_Dense");
    ::declare_TransverseIsing<openjij::graph::Sparse<openjij::FloatType>>(m_system, "_Sparse");

    //Continuous Time Transeverse Ising
    ::declare_ContinuousTimeIsing<openjij::graph::Sparse<openjij::FloatType>>(m_system, "_Sparse");

#ifdef USE_CUDA
    //ChimeraTransverseGPU
    ::declare_ChimeraTranseverseGPU<openjij::GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>(m_system);
    //ChimeraClassicalGPU
    ::declare_ChimeraClassicalGPU<openjij::GPUFloatType, BLOCK_ROW, BLOCK_COL>(m_system);
#endif

    /**********************************************************
    //namespace algorithm
     **********************************************************/
    py::module m_algorithm = m.def_submodule("algorithm", "cxxjij module for algorithm");

    //singlespinflip
    ::declare_Algorithm_run<openjij::updater::SingleSpinFlip, openjij::system::ClassicalIsing<openjij::graph::Dense<openjij::FloatType>>,    openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<openjij::updater::SingleSpinFlip, openjij::system::ClassicalIsing<openjij::graph::Sparse<openjij::FloatType>>,   openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<openjij::updater::SingleSpinFlip, openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<openjij::FloatType>>,   openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<openjij::updater::KLocal, openjij::system::KLocalPolynomial<openjij::graph::Polynomial<openjij::FloatType>>, openjij::RandomEngine>(m_algorithm, "KLocal");
    ::declare_Algorithm_run<openjij::updater::SingleSpinFlip, openjij::system::TransverseIsing<openjij::graph::Dense<openjij::FloatType>>,   openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
    ::declare_Algorithm_run<openjij::updater::SingleSpinFlip, openjij::system::TransverseIsing<openjij::graph::Sparse<openjij::FloatType>>,  openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");

    //swendsen-wang
    ::declare_Algorithm_run<openjij::updater::SwendsenWang, openjij::system::ClassicalIsing<openjij::graph::Sparse<openjij::FloatType>>, openjij::RandomEngine>(m_algorithm, "SwendsenWang");

    //Continuous time swendsen-wang
    ::declare_Algorithm_run<openjij::updater::ContinuousTimeSwendsenWang, openjij::system::ContinuousTimeIsing<openjij::graph::Sparse<openjij::FloatType>>, openjij::RandomEngine>(m_algorithm, "ContinuousTimeSwendsenWang");

#ifdef USE_CUDA
    //GPU
    ::declare_Algorithm_run<openjij::updater::GPU, openjij::system::ChimeraTransverseGPU<openjij::GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>, openjij::utility::cuda::CurandWrapper<openjij::GPUFloatType, GPURandomEngine>>(m_algorithm, "GPU");
    ::declare_Algorithm_run<openjij::updater::GPU, openjij::system::ChimeraClassicalGPU<openjij::GPUFloatType, BLOCK_ROW, BLOCK_COL>, openjij::utility::cuda::CurandWrapper<openjij::GPUFloatType, GPURandomEngine>>(m_algorithm, "GPU");
#endif

    /**********************************************************
    //namespace utlity
     **********************************************************/
    py::module m_utility = m.def_submodule("utility", "cxxjij module for utility");

    //schedule_list
    ::declare_ClassicalUpdaterParameter(m_utility);
    ::declare_ClassicalConstraintUpdaterParameter(m_utility);
    ::declare_TransverseFieldUpdaterParameter(m_utility);

    ::declare_Schedule<openjij::system::classical_system>(m_utility, "Classical");
    ::declare_Schedule<openjij::system::classical_constraint_system>(m_utility, "ClassicalConstraint");
    ::declare_Schedule<openjij::system::transverse_field_system>(m_utility, "TransverseField");

    m_utility.def("make_classical_schedule_list", &openjij::utility::make_classical_schedule_list,
            "beta_min"_a, "beta_max"_a, "one_mc_step"_a, "num_call_updater"_a);

    m_utility.def("make_classical_constraint_schedule_list", &openjij::utility::make_classical_constraint_schedule_list,
            "lambda"_a, "beta_min"_a, "beta_max"_a, "one_mc_step"_a, "num_call_updater"_a);

    m_utility.def("make_transverse_field_schedule_list", &openjij::utility::make_transverse_field_schedule_list,
            "beta"_a, "one_mc_step"_a, "num_call_updater"_a);

    


    /**********************************************************
    //namespace result
     **********************************************************/
    py::module m_result = m.def_submodule("result", "cxxjij module for result");

    ::declare_get_solution<openjij::system::ClassicalIsing<openjij::graph::Dense<openjij::FloatType>>>(m_result);
    ::declare_get_solution<openjij::system::ClassicalIsing<openjij::graph::Sparse<openjij::FloatType>>>(m_result);
    ::declare_get_solution<openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<openjij::FloatType>>>(m_result);
    ::declare_get_solution<openjij::system::KLocalPolynomial<openjij::graph::Polynomial<openjij::FloatType>>>(m_result);
    ::declare_get_solution<openjij::system::TransverseIsing<openjij::graph::Dense<openjij::FloatType>>>(m_result);
    ::declare_get_solution<openjij::system::TransverseIsing<openjij::graph::Sparse<openjij::FloatType>>>(m_result);
    ::declare_get_solution<openjij::system::ContinuousTimeIsing<openjij::graph::Sparse<openjij::FloatType>>>(m_result);
#ifdef USE_CUDA
    ::declare_get_solution<openjij::system::ChimeraTransverseGPU<openjij::GPUFloatType, BLOCK_ROW, BLOCK_COL, BLOCK_TROT>>(m_result);
    ::declare_get_solution<openjij::system::ChimeraClassicalGPU<openjij::GPUFloatType, BLOCK_ROW, BLOCK_COL>>(m_result);
#endif
}





























