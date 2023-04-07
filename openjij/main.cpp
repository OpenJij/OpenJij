//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11_json/pybind11_json.hpp>

#include <type_traits>
// disable Eigen warning
#include <openjij/utility/disable_eigen_warning.hpp>
#include <openjij/utility/random.hpp>

#include "compile_config.hpp"
#include "declare.hpp"

PYBIND11_MODULE(cxxjij, m) {
  py::options options;
  //options.disable_function_signatures();
  
  m.doc() = "openjij is a framework for ising and qubo";

  /**********************************************************
   //namespace graph
   **********************************************************/
  py::module m_graph = m.def_submodule("graph", "cxxjij submodule for graph");

  openjij::declare_Graph(m_graph);

  openjij::declare_Dir(m_graph);
  openjij::declare_ChimeraDir(m_graph);

  // CPU version (openjij::FloatType)
  openjij::declare_Dense<openjij::FloatType>(m_graph, "");
  openjij::declare_Sparse<openjij::FloatType>(m_graph, "");
  openjij::declare_CSRSparse<openjij::FloatType>(m_graph, "");
  openjij::declare_Square<openjij::FloatType>(m_graph, "");
  openjij::declare_Chimera<openjij::FloatType>(m_graph, "");
  openjij::declare_Polynomial<openjij::FloatType>(m_graph, "");

  openjij::declare_BinaryPolynomialModel<openjij::FloatType>(m_graph);
  openjij::declare_IsingPolynomialModel<openjij::FloatType>(m_graph);

  py::module_ m_sampler = m.def_submodule("sampler");
  openjij::declare_SASampler<openjij::graph::BinaryPolynomialModel<openjij::FloatType>>(m_sampler, "BPM");
  openjij::declare_SASampler<openjij::graph::IsingPolynomialModel<openjij::FloatType>>(m_sampler, "IPM");

  // GPU version (openjij::GPUFloatType)
  if (!std::is_same<openjij::FloatType, openjij::GPUFloatType>::value) {
    openjij::declare_Dense<openjij::GPUFloatType>(m_graph, "GPU");
    openjij::declare_Sparse<openjij::GPUFloatType>(m_graph, "GPU");
    openjij::declare_CSRSparse<openjij::GPUFloatType>(m_graph, "GPU");
    openjij::declare_Square<openjij::GPUFloatType>(m_graph, "GPU");
    openjij::declare_Chimera<openjij::GPUFloatType>(m_graph, "GPU");
  } else {
    // raise warning
    std::cerr << "Warning: please use classes in Graph module without suffix "
                 "\"GPU\" or define type aliases."
              << std::endl;
  }

  /**********************************************************
   //namespace system
   **********************************************************/
  py::module m_system = m.def_submodule("system", "cxxjij module for system");

  // ClassicalIsing
  openjij::declare_ClassicalIsing<openjij::graph::Dense<openjij::FloatType>>(
      m_system, "_Dense");
  openjij::declare_ClassicalIsing<openjij::graph::Sparse<openjij::FloatType>>(
      m_system, "_Sparse");
  openjij::declare_ClassicalIsing<openjij::graph::CSRSparse<openjij::FloatType>>(
      m_system, "_CSRSparse");
  openjij::declare_ClassicalIsingPolynomial<
      openjij::graph::Polynomial<openjij::FloatType>>(m_system, "_Polynomial");
  openjij::declare_KLocalPolynomial<
      openjij::graph::Polynomial<openjij::FloatType>>(m_system, "_Polynomial");

  // TransverselIsing
  openjij::declare_TransverseIsing<openjij::graph::Dense<openjij::FloatType>>(
      m_system, "_Dense");
  openjij::declare_TransverseIsing<openjij::graph::Sparse<openjij::FloatType>>(
      m_system, "_Sparse");
  openjij::declare_TransverseIsing<openjij::graph::CSRSparse<openjij::FloatType>>(
      m_system, "_CSRSparse");

  // Continuous Time Transeverse Ising
  openjij::declare_ContinuousTimeIsing<
      openjij::graph::Sparse<openjij::FloatType>>(m_system, "_Sparse");
  openjij::declare_ContinuousTimeIsing<
      openjij::graph::CSRSparse<openjij::FloatType>>(m_system, "_CSRSparse");

#ifdef USE_CUDA
  // ChimeraTransverseGPU
  openjij::declare_ChimeraTranseverseGPU<openjij::GPUFloatType,
                                         openjij::BLOCK_ROW, openjij::BLOCK_COL,
                                         openjij::BLOCK_TROT>(m_system);
  // ChimeraClassicalGPU
  openjij::declare_ChimeraClassicalGPU<openjij::GPUFloatType,
                                       openjij::BLOCK_ROW, openjij::BLOCK_COL>(
      m_system);
#endif

  /**********************************************************
   //namespace algorithm
   **********************************************************/
  py::module m_algorithm =
      m.def_submodule("algorithm", "cxxjij module for algorithm");

  openjij::declare_UpdateMethod(m_algorithm);
  openjij::declare_RandomNumberEngine(m_algorithm);

  // singlespinflip
  openjij::declare_Algorithm_run<openjij::updater::SingleSpinFlip,
                                 openjij::system::ClassicalIsing<
                                     openjij::graph::Dense<openjij::FloatType>>,
                                 openjij::RandomEngine>(m_algorithm,
                                                        "SingleSpinFlip");
  openjij::declare_Algorithm_run<
      openjij::updater::SingleSpinFlip,
      openjij::system::ClassicalIsing<
          openjij::graph::Sparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
  openjij::declare_Algorithm_run<
      openjij::updater::SingleSpinFlip,
      openjij::system::ClassicalIsing<
          openjij::graph::CSRSparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
  openjij::declare_Algorithm_run<
      openjij::updater::SingleSpinFlip,
      openjij::system::ClassicalIsingPolynomial<
          openjij::graph::Polynomial<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
  openjij::declare_Algorithm_run<
      openjij::updater::KLocal,
      openjij::system::KLocalPolynomial<
          openjij::graph::Polynomial<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "KLocal");
  openjij::declare_Algorithm_run<openjij::updater::SingleSpinFlip,
                                 openjij::system::TransverseIsing<
                                     openjij::graph::Dense<openjij::FloatType>>,
                                 openjij::RandomEngine>(m_algorithm,
                                                        "SingleSpinFlip");
  openjij::declare_Algorithm_run<
      openjij::updater::SingleSpinFlip,
      openjij::system::TransverseIsing<
          openjij::graph::Sparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");
  openjij::declare_Algorithm_run<
      openjij::updater::SingleSpinFlip,
      openjij::system::TransverseIsing<
          openjij::graph::CSRSparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SingleSpinFlip");

  // swendsen-wang
  openjij::declare_Algorithm_run<
      openjij::updater::SwendsenWang,
      openjij::system::ClassicalIsing<
          openjij::graph::Sparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SwendsenWang");
  openjij::declare_Algorithm_run<
      openjij::updater::SwendsenWang,
      openjij::system::ClassicalIsing<
          openjij::graph::CSRSparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "SwendsenWang");

  // Continuous time swendsen-wang
  openjij::declare_Algorithm_run<
      openjij::updater::ContinuousTimeSwendsenWang,
      openjij::system::ContinuousTimeIsing<
          openjij::graph::Sparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "ContinuousTimeSwendsenWang");
  openjij::declare_Algorithm_run<
      openjij::updater::ContinuousTimeSwendsenWang,
      openjij::system::ContinuousTimeIsing<
          openjij::graph::CSRSparse<openjij::FloatType>>,
      openjij::RandomEngine>(m_algorithm, "ContinuousTimeSwendsenWang");

#ifdef USE_CUDA
  // GPU
  openjij::declare_Algorithm_run<
      openjij::updater::GPU,
      openjij::system::ChimeraTransverseGPU<
          openjij::GPUFloatType, openjij::BLOCK_ROW, openjij::BLOCK_COL,
          openjij::BLOCK_TROT>,
      openjij::utility::cuda::CurandWrapper<openjij::GPUFloatType,
                                            openjij::GPURandomEngine>>(
      m_algorithm, "GPU");
  openjij::declare_Algorithm_run<
      openjij::updater::GPU,
      openjij::system::ChimeraClassicalGPU<
          openjij::GPUFloatType, openjij::BLOCK_ROW, openjij::BLOCK_COL>,
      openjij::utility::cuda::CurandWrapper<openjij::GPUFloatType,
                                            openjij::GPURandomEngine>>(
      m_algorithm, "GPU");
#endif

  /**********************************************************
   //namespace utlity
   **********************************************************/
  py::module m_utility =
      m.def_submodule("utility", "cxxjij module for utility");

  openjij::declare_TemperatureSchedule(m_utility);

  // schedule_list
  openjij::declare_ClassicalUpdaterParameter(m_utility);
  openjij::declare_ClassicalConstraintUpdaterParameter(m_utility);
  openjij::declare_TransverseFieldUpdaterParameter(m_utility);

  openjij::declare_Schedule<openjij::system::classical_system>(m_utility,
                                                               "Classical");
  openjij::declare_Schedule<openjij::system::classical_constraint_system>(
      m_utility, "ClassicalConstraint");
  openjij::declare_Schedule<openjij::system::transverse_field_system>(
      m_utility, "TransverseField");

  m_utility.def("make_classical_schedule_list",
                &openjij::utility::make_classical_schedule_list, "beta_min"_a,
                "beta_max"_a, "one_mc_step"_a, "num_call_updater"_a);

  m_utility.def("make_classical_constraint_schedule_list",
                &openjij::utility::make_classical_constraint_schedule_list,
                "lambda"_a, "beta_min"_a, "beta_max"_a, "one_mc_step"_a,
                "num_call_updater"_a);

  m_utility.def("make_transverse_field_schedule_list",
                &openjij::utility::make_transverse_field_schedule_list,
                "beta"_a, "one_mc_step"_a, "num_call_updater"_a);

  /**********************************************************
   //namespace result
   **********************************************************/
  py::module m_result = m.def_submodule("result", "cxxjij module for result");

  openjij::declare_get_solution<openjij::system::ClassicalIsing<
      openjij::graph::Dense<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::ClassicalIsing<
      openjij::graph::Sparse<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::ClassicalIsing<
      openjij::graph::CSRSparse<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::ClassicalIsingPolynomial<
      openjij::graph::Polynomial<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::KLocalPolynomial<
      openjij::graph::Polynomial<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::TransverseIsing<
      openjij::graph::Dense<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::TransverseIsing<
      openjij::graph::Sparse<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::TransverseIsing<
      openjij::graph::CSRSparse<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::ContinuousTimeIsing<
      openjij::graph::Sparse<openjij::FloatType>>>(m_result);
  openjij::declare_get_solution<openjij::system::ContinuousTimeIsing<
      openjij::graph::CSRSparse<openjij::FloatType>>>(m_result);
#ifdef USE_CUDA
  openjij::declare_get_solution<openjij::system::ChimeraTransverseGPU<
      openjij::GPUFloatType, openjij::BLOCK_ROW, openjij::BLOCK_COL,
      openjij::BLOCK_TROT>>(m_result);
  openjij::declare_get_solution<openjij::system::ChimeraClassicalGPU<
      openjij::GPUFloatType, openjij::BLOCK_ROW, openjij::BLOCK_COL>>(m_result);
#endif
}
