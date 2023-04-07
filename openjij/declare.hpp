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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include <pybind11_json/pybind11_json.hpp>

#include <nlohmann/json.hpp>

#include <openjij/algorithm/all.hpp>
#include <openjij/graph/all.hpp>
#include <openjij/result/all.hpp>
#include <openjij/system/all.hpp>
#include <openjij/updater/all.hpp>
#include <openjij/sampler/sa_sampler.hpp>

namespace py = pybind11;

using namespace py::literals;

namespace openjij {

// NOTE: please add `py::module_local()` when defining `py::class_`

// graph
inline void declare_Graph(py::module &m) {
  py::class_<graph::Graph>(m, "Graph", py::module_local())
      .def(py::init<std::size_t>(), "num_spins"_a)
      .def(
          "gen_spin",
          [](const graph::Graph &self, std::size_t seed) {
            RandomEngine rng(seed);
            return self.gen_spin(rng);
          },
          "seed"_a)
      .def("gen_spin",
           [](const graph::Graph &self) {
             RandomEngine rng(std::random_device{}());
             return self.gen_spin(rng);
           })
      .def(
          "gen_binary",
          [](const graph::Graph &self, std::size_t seed) {
            RandomEngine rng(seed);
            return self.gen_binary(rng);
          },
          "seed"_a)
      .def("gen_binary",
           [](const graph::Graph &self) {
             RandomEngine rng(std::random_device{}());
             return self.gen_binary(rng);
           })
      .def("size", &graph::Graph::size);
}

// dense
template <typename FloatType>
inline void declare_Dense(py::module &m, const std::string &suffix) {

  using json = nlohmann::json;

  auto str = std::string("Dense") + suffix;
  py::class_<graph::Dense<FloatType>, graph::Graph>(m, str.c_str(),
                                                    py::module_local())
      .def(py::init<std::size_t>(), "num_spins"_a)
      .def(py::init([](py::object obj) {
             return std::unique_ptr<graph::Dense<FloatType>>(
                 new graph::Dense<FloatType>(static_cast<json>(obj)));
           }),
           "obj"_a)
      .def(py::init<const graph::Dense<FloatType> &>(), "other"_a)
      .def("set_interaction_matrix",
           &graph::Dense<FloatType>::set_interaction_matrix, "interaction"_a)
      .def(
          "calc_energy",
          [](const graph::Dense<FloatType> &self,
             const Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>
                 &spins) { return self.calc_energy(spins); },
          "spins"_a)
      .def(
          "calc_energy",
          [](const graph::Dense<FloatType> &self, const graph::Spins &spins) {
            return self.calc_energy(spins);
          },
          "spins"_a)
      .def(
          "__setitem__",
          [](graph::Dense<FloatType> &self,
             const std::pair<std::size_t, std::size_t> &key,
             FloatType val) { self.J(key.first, key.second) = val; },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Dense<FloatType> &self,
             const std::pair<std::size_t, std::size_t> &key) {
            return self.J(key.first, key.second);
          },
          "key"_a)
      .def(
          "__setitem__",
          [](graph::Dense<FloatType> &self, std::size_t key, FloatType val) {
            self.h(key) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Dense<FloatType> &self, std::size_t key) {
            return self.h(key);
          },
          "key"_a)
      .def("get_interactions", &graph::Dense<FloatType>::get_interactions);
}

// sparse
template <typename FloatType>
inline void declare_Sparse(py::module &m, const std::string &suffix) {

  using json = nlohmann::json;

  auto str = std::string("Sparse") + suffix;
  py::class_<graph::Sparse<FloatType>, graph::Graph>(m, str.c_str(),
                                                     py::module_local())
      .def(py::init<std::size_t, std::size_t>(), "num_spins"_a, "num_edges"_a)
      .def(py::init<std::size_t>(), "num_spins"_a)
      .def(py::init([](py::object obj, std::size_t num_edges) {
             return std::unique_ptr<graph::Sparse<FloatType>>(
                 new graph::Sparse<FloatType>(static_cast<json>(obj),
                                              num_edges));
           }),
           "obj"_a, "num_edges"_a)
      .def(py::init([](py::object obj) {
             return std::unique_ptr<graph::Sparse<FloatType>>(
                 new graph::Sparse<FloatType>(static_cast<json>(obj)));
           }),
           "obj"_a)
      .def(py::init<const graph::Sparse<FloatType> &>(), "other"_a)
      .def("adj_nodes", &graph::Sparse<FloatType>::adj_nodes)
      .def("get_num_edges", &graph::Sparse<FloatType>::get_num_edges)
      .def(
          "calc_energy",
          [](const graph::Sparse<FloatType> &self,
             const Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>
                 &spins) { return self.calc_energy(spins); },
          "spins"_a)
      .def(
          "calc_energy",
          [](const graph::Sparse<FloatType> &self, const graph::Spins &spins) {
            return self.calc_energy(spins);
          },
          "spins"_a)
      .def(
          "__setitem__",
          [](graph::Sparse<FloatType> &self,
             const std::pair<std::size_t, std::size_t> &key,
             FloatType val) { self.J(key.first, key.second) = val; },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Sparse<FloatType> &self,
             const std::pair<std::size_t, std::size_t> &key) {
            return self.J(key.first, key.second);
          },
          "key"_a)
      .def(
          "__setitem__",
          [](graph::Sparse<FloatType> &self, std::size_t key, FloatType val) {
            self.h(key) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Sparse<FloatType> &self, std::size_t key) {
            return self.h(key);
          },
          "key"_a);
}

// csr sparse
template <typename FloatType>
inline void declare_CSRSparse(py::module &m, const std::string &suffix) {

  auto str = std::string("CSRSparse") + suffix;
  py::class_<graph::CSRSparse<FloatType>, graph::Graph>(m, str.c_str(),
                                                     py::module_local())
      .def(py::init<const Eigen::SparseMatrix<FloatType, Eigen::RowMajor> &>(), "interaction"_a)
      .def(py::init<const graph::CSRSparse<FloatType> &>(), "other"_a)
      .def(
          "calc_energy",
          [](const graph::CSRSparse<FloatType> &self,
             const Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>
                 &spins) { return self.calc_energy(spins); },
          "spins"_a)
      .def(
          "calc_energy",
          [](const graph::CSRSparse<FloatType> &self, const graph::Spins &spins) {
            return self.calc_energy(spins);
          },
          "spins"_a)
      .def("get_interactions", &graph::CSRSparse<FloatType>::get_interactions);
}

// Polynomial
template <typename FloatType>
inline void declare_Polynomial(py::module &m, const std::string &suffix) {

  using json = nlohmann::json;
  using Poly = graph::Polynomial<FloatType>;
  auto str = std::string("Polynomial") + suffix;

  py::class_<Poly, graph::Graph>(m, str.c_str(), py::module_local())
      .def(py::init<const std::size_t>(), "num_variables"_a)
      .def(py::init([](const py::object &obj) {
             return std::unique_ptr<graph::Polynomial<FloatType>>(
                 new graph::Polynomial<FloatType>(static_cast<json>(obj)));
           }),
           "obj"_a)
      .def("get_num_interactions", &Poly::get_num_interactions)
      .def("calc_energy", &Poly::calc_energy, "spins"_a, "omp_flag"_a = true)
      .def("energy", &Poly::energy, "spins"_a, "omp_flag"_a = true)
      .def(
          "__setitem__",
          [](Poly &self, graph::Index key, FloatType val) {
            self.J(key) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__setitem__",
          [](Poly &self, std::vector<graph::Index> &key, FloatType val) {
            self.J(key) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const Poly &self, std::vector<graph::Index> &key) {
            return self.J(key);
          },
          "key"_a)
      .def(
          "__getitem__",
          [](const Poly &self, graph::Index key) { return self.J(key); },
          "key"_a)
      .def("get_polynomial", [](const Poly &self) {
        py::dict py_polynomial;
        for (std::size_t i = 0; i < self.get_keys().size(); ++i) {
          py::tuple temp;
          for (const auto &it : self.get_keys()[i]) {
            temp = temp + py::make_tuple(it);
          }
          py_polynomial[temp] = self.get_values()[i];
        }
        return py_polynomial;
      });
}

// enum class Dir
inline void declare_Dir(py::module &m) {
  py::enum_<graph::Dir>(m, "Dir", py::module_local())
      .value("PLUS_R", graph::Dir::PLUS_R)
      .value("MINUS_R", graph::Dir::MINUS_R)
      .value("PLUS_C", graph::Dir::PLUS_C)
      .value("MINUS_C", graph::Dir::MINUS_C);
}

// square
template <typename FloatType>
inline void declare_Square(py::module &m, const std::string &suffix) {

  using json = nlohmann::json;

  auto str = std::string("Square") + suffix;
  py::class_<graph::Square<FloatType>, graph::Sparse<FloatType>>(
      m, str.c_str(), py::module_local())
      .def(py::init<std::size_t, std::size_t, FloatType>(), "num_row"_a,
           "num_column"_a, "init_val"_a = 0)
      .def(py::init<const graph::Square<FloatType> &>(), "other"_a)
      .def(py::init([](py::object obj, std::size_t num_row,
                       std::size_t num_column, FloatType init_val) {
             return std::unique_ptr<graph::Square<FloatType>>(
                 new graph::Square<FloatType>(static_cast<json>(obj), num_row,
                                              num_column, init_val));
           }),
           "obj"_a, "num_row"_a, "num_column"_a, "init_val"_a = 0)
      .def("to_ind", &graph::Square<FloatType>::to_ind)
      .def("to_rc", &graph::Square<FloatType>::to_rc)
      .def("get_num_row", &graph::Square<FloatType>::get_num_row)
      .def("get_num_column", &graph::Square<FloatType>::get_num_column)
      .def(
          "__setitem__",
          [](graph::Square<FloatType> &self,
             const std::tuple<std::size_t, std::size_t, graph::Dir> &key,
             FloatType val) {
            self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key)) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Square<FloatType> &self,
             const std::tuple<std::size_t, std::size_t, graph::Dir> &key) {
            return self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key));
          },
          "key"_a)
      .def(
          "__setitem__",
          [](graph::Square<FloatType> &self,
             const std::pair<std::size_t, std::size_t> &key,
             FloatType val) { self.h(key.first, key.second) = val; },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Square<FloatType> &self,
             const std::pair<std::size_t, std::size_t> &key) {
            return self.h(key.first, key.second);
          },
          "key"_a);
}

// enum class ChimeraDir
inline void declare_ChimeraDir(py::module &m) {
  py::enum_<graph::ChimeraDir>(m, "ChimeraDir")
      .value("PLUS_R", graph::ChimeraDir::PLUS_R)
      .value("MINUS_R", graph::ChimeraDir::MINUS_R)
      .value("PLUS_C", graph::ChimeraDir::PLUS_C)
      .value("MINUS_C", graph::ChimeraDir::MINUS_C)
      .value("IN_0or4", graph::ChimeraDir::IN_0or4)
      .value("IN_1or5", graph::ChimeraDir::IN_1or5)
      .value("IN_2or6", graph::ChimeraDir::IN_2or6)
      .value("IN_3or7", graph::ChimeraDir::IN_3or7);
}

// chimera
template <typename FloatType>
inline void declare_Chimera(py::module &m, const std::string &suffix) {

  using json = nlohmann::json;

  auto str = std::string("Chimera") + suffix;
  py::class_<graph::Chimera<FloatType>, graph::Sparse<FloatType>>(
      m, str.c_str(), py::module_local())
      .def(py::init<std::size_t, std::size_t, FloatType>(), "num_row"_a,
           "num_column"_a, "init_val"_a = 0)
      .def(py::init<const graph::Chimera<FloatType> &>(), "other"_a)
      .def(py::init([](py::object obj, std::size_t num_row,
                       std::size_t num_column, FloatType init_val) {
             return std::unique_ptr<graph::Chimera<FloatType>>(
                 new graph::Chimera<FloatType>(static_cast<json>(obj), num_row,
                                               num_column, init_val));
           }),
           "obj"_a, "num_row"_a, "num_column"_a, "init_val"_a = 0)
      .def("to_ind", &graph::Chimera<FloatType>::to_ind)
      .def("to_rci", &graph::Chimera<FloatType>::to_rci)
      .def("get_num_row", &graph::Chimera<FloatType>::get_num_row)
      .def("get_num_column", &graph::Chimera<FloatType>::get_num_column)
      .def("get_num_in_chimera", &graph::Chimera<FloatType>::get_num_in_chimera)
      .def(
          "__setitem__",
          [](graph::Chimera<FloatType> &self,
             const std::tuple<std::size_t, std::size_t, std::size_t,
                              graph::ChimeraDir> &key,
             FloatType val) {
            self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                   std::get<3>(key)) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Chimera<FloatType> &self,
             const std::tuple<std::size_t, std::size_t, std::size_t,
                              graph::ChimeraDir> &key) {
            return self.J(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                          std::get<3>(key));
          },
          "key"_a)
      .def(
          "__setitem__",
          [](graph::Chimera<FloatType> &self,
             const std::tuple<std::size_t, std::size_t, std::size_t> &key,
             FloatType val) {
            self.h(std::get<0>(key), std::get<1>(key), std::get<2>(key)) = val;
          },
          "key"_a, "val"_a)
      .def(
          "__getitem__",
          [](const graph::Chimera<FloatType> &self,
             const std::tuple<std::size_t, std::size_t, std::size_t> &key) {
            return self.h(std::get<0>(key), std::get<1>(key), std::get<2>(key));
          },
          "key"_a);
}

// system

// ClassicalIsing
template <typename GraphType>
inline void declare_ClassicalIsing(py::module &m,
                                   const std::string &gtype_str) {
  // ClassicalIsing
  using ClassicalIsing = system::ClassicalIsing<GraphType>;

  auto str = std::string("ClassicalIsing") + gtype_str;
  py::class_<ClassicalIsing>(m, str.c_str(), py::module_local())
      .def(py::init<const graph::Spins &, const GraphType &>(), "init_spin"_a,
           "init_interaction"_a)
      .def(
          "reset_spins",
          [](ClassicalIsing &self, const graph::Spins &init_spin) {
            self.reset_spins(init_spin);
          },
          "init_spin"_a)
      .def_readwrite("spin", &ClassicalIsing::spin)
      .def_readonly("interaction", &ClassicalIsing::interaction)
      .def_readonly("num_spins", &ClassicalIsing::num_spins);

  // make_classical_ising
  auto mkci_str = std::string("make_classical_ising");
  m.def(
      mkci_str.c_str(),
      [](const graph::Spins &init_spin, const GraphType &init_interaction) {
        return system::make_classical_ising(init_spin, init_interaction);
      },
      "init_spin"_a, "init_interaction"_a);
}

// ClassicalIsingPolynomial
template <typename GraphType>
inline void declare_ClassicalIsingPolynomial(py::module &m,
                                             const std::string &gtype_str) {

  using CIP = system::ClassicalIsingPolynomial<GraphType>;
  auto str = std::string("ClassicalIsing") + gtype_str;

  py::class_<CIP>(m, str.c_str(), py::module_local())
      .def(py::init<const graph::Spins &, const GraphType &,
                    const cimod::Vartype>(),
           "init_variables"_a, "init_interaction"_a, "vartype"_a)
      .def(py::init<const graph::Spins &, const GraphType &,
                    const std::string>(),
           "init_variables"_a, "init_interaction"_a, "vartype"_a)
      .def(py::init([](const graph::Spins &init_spins, const py::object &obj) {
             return std::unique_ptr<CIP>(
                 new CIP(init_spins, static_cast<nlohmann::json>(obj)));
           }),
           "init_spin"_a, "obj"_a)
      .def_readonly("variables", &CIP::variables)
      .def_readonly("num_variables", &CIP::num_variables)
      .def("reset_variables", &CIP::reset_variables, "init_variables"_a)
      .def("reset_spins", &CIP::reset_variables, "init_spins"_a)
      .def("get_values", &CIP::get_values)
      .def("get_keys", &CIP::get_keys)
      .def("get_adj", &CIP::get_adj)
      .def("get_vartype_to_string", &CIP::get_vartype_string)
      .def("get_max_effective_dE", &CIP::get_max_effective_dE)
      .def("get_min_effective_dE", &CIP::get_min_effective_dE);

  // make_classical_ising_polynomial
  auto mkcip_str = std::string("make_classical_ising_polynomial");
  m.def(
      mkcip_str.c_str(),
      [](const graph::Spins &init_spin, const GraphType &init_interaction,
         const cimod::Vartype vartype) {
        return system::make_classical_ising_polynomial(
            init_spin, init_interaction, vartype);
      },
      "init_spin"_a, "init_interaction"_a, "vartype"_a);

  // make_classical_ising_polynomial
  m.def(
      mkcip_str.c_str(),
      [](const graph::Spins &init_spin, const GraphType &init_interaction,
         const std::string vartype) {
        return system::make_classical_ising_polynomial(
            init_spin, init_interaction, vartype);
      },
      "init_spin"_a, "init_interaction"_a, "vartype"_a);

  // make_classical_ising_polynomial
  m.def(
      mkcip_str.c_str(),
      [](const graph::Spins &init_spin, const py::object &obj) {
        return system::make_classical_ising_polynomial(
            init_spin, static_cast<nlohmann::json>(obj));
      },
      "init_spin"_a, "obj"_a);
}

template <typename GraphType>
inline void declare_KLocalPolynomial(py::module &m,
                                     const std::string &gtype_str) {

  using KLP = system::KLocalPolynomial<GraphType>;
  auto str = std::string("KLocal") + gtype_str;

  py::class_<KLP>(m, str.c_str(), py::module_local())
      .def(py::init<const graph::Binaries &, const GraphType &>(),
           "init_spin"_a, "init_interaction"_a)
      .def(py::init(
               [](const graph::Binaries &init_binaries, const py::object &obj) {
                 return std::unique_ptr<KLP>(
                     new KLP(init_binaries, static_cast<nlohmann::json>(obj)));
               }),
           "init_binaries"_a, "obj"_a)
      .def_readonly("binaries", &KLP::binaries)
      .def_readonly("num_binaries", &KLP::num_binaries)
      .def_readonly("count_call_updater", &KLP::count_call_updater)
      .def_property_readonly("num_interactions", &KLP::GetNumInteractions)
      .def_readwrite("rate_call_k_local", &KLP::rate_call_k_local)
      .def("reset_binaries", &KLP::reset_binaries, "init_binaries"_a)
      .def("reset_spins", &KLP::reset_binaries, "init_spins"_a)
      .def("reset_dE", &KLP::reset_dE)
      .def("get_active_binaries", &KLP::get_active_binaries)
      .def("get_max_effective_dE", &KLP::get_max_effective_dE)
      .def("get_min_effective_dE", &KLP::get_min_effective_dE)
      .def("get_keys", &KLP::get_keys)
      .def("get_values", &KLP::get_values)
      .def("get_vartype_to_string", &KLP::get_vartype_string)
      .def("get_polynomial",
           [](const KLP &self) {
             py::dict py_polynomial;
             const auto &poly_key_list = self.get_keys();
             const auto &poly_value_list = self.get_values();
             for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
               py::tuple tuple;
               for (const auto &index : poly_key_list[i]) {
                 tuple = tuple + py::make_tuple(index);
               }
               py_polynomial[tuple] = poly_value_list[i];
             }
             return py_polynomial;
           })
      .def("get_adj", [](const KLP &self) {
        const auto &adj = self.get_adj();
        const auto &poly_key_list = self.get_keys();
        const auto &poly_value_list = self.get_values();
        py::dict py_adj;
        for (int64_t i = 0; i < self.num_binaries; ++i) {
          py::dict dict;
          for (const auto &index_key : adj[i]) {
            py::tuple tuple;
            for (const auto &index_binary : poly_key_list[index_key]) {
              tuple = tuple + py::make_tuple(index_binary);
            }
            dict[tuple] = poly_value_list[index_key];
          }
          py_adj[py::int_(i)] = dict;
        }
        return py_adj;
      });

  // make_classical_ising_polynomial
  auto mkcip_str = std::string("make_k_local_polynomial");
  m.def(
      mkcip_str.c_str(),
      [](const graph::Spins &init_spin, const GraphType &init_interaction) {
        return system::make_k_local_polynomial(init_spin, init_interaction);
      },
      "init_spin"_a, "init_interaction"_a);

  // make_classical_ising_polynomial
  auto mkcip_json_str = std::string("make_k_local_polynomial");
  m.def(
      mkcip_json_str.c_str(),
      [](const graph::Spins &init_spin, const py::object &obj) {
        return system::make_k_local_polynomial(
            init_spin, static_cast<nlohmann::json>(obj));
      },
      "init_spin"_a, "obj"_a);
}

// TransverseIsing
template <typename GraphType>
inline void declare_TransverseIsing(py::module &m,
                                    const std::string &gtype_str) {
  // TransverseIsing
  using TransverseIsing = system::TransverseIsing<GraphType>;
  using FloatType = typename GraphType::value_type;

  auto str = std::string("TransverseIsing") + gtype_str;
  py::class_<TransverseIsing>(m, str.c_str(), py::module_local())
      .def(py::init<const system::TrotterSpins &, const GraphType &,
                    FloatType>(),
           "init_spin"_a, "init_interaction"_a, "gamma"_a)
      .def(py::init<const graph::Spins &, const GraphType &, FloatType,
                    size_t>(),
           "init_classical_spins"_a, "init_interaction"_a, "gamma"_a,
           "num_trotter_slices"_a)
      .def(
          "reset_spins",
          [](TransverseIsing &self,
             const system::TrotterSpins &init_trotter_spins) {
            self.reset_spins(init_trotter_spins);
          },
          "init_trotter_spins"_a)
      .def(
          "reset_spins",
          [](TransverseIsing &self, const graph::Spins &classical_spins) {
            self.reset_spins(classical_spins);
          },
          "classical_spins"_a)
      .def_readwrite("trotter_spins", &TransverseIsing::trotter_spins)
      .def_readonly("interaction", &TransverseIsing::interaction)
      .def_readonly("num_classical_spins",
                    &TransverseIsing::num_classical_spins)
      .def_readwrite("gamma", &TransverseIsing::gamma);

  // make_transverse_ising
  auto mkci_str = std::string("make_transverse_ising");
  m.def(
      mkci_str.c_str(),
      [](const system::TrotterSpins &init_trotter_spins,
         const GraphType &init_interaction, double gamma) {
        return system::make_transverse_ising(init_trotter_spins,
                                             init_interaction, gamma);
      },
      "init_trotter_spins"_a, "init_interaction"_a, "gamma"_a);

  m.def(
      mkci_str.c_str(),
      [](const graph::Spins &classical_spins, const GraphType &init_interaction,
         double gamma, std::size_t num_trotter_slices) {
        return system::make_transverse_ising(classical_spins, init_interaction,
                                             gamma, num_trotter_slices);
      },
      "classical_spins"_a, "init_interaction"_a, "gamma"_a,
      "num_trotter_slices"_a);
}

// Continuous Time Transverse Ising
template <typename GraphType>
inline void declare_ContinuousTimeIsing(py::module &m,
                                        const std::string &gtype_str) {
  // TransverseIsing
  using TransverseIsing = system::ContinuousTimeIsing<GraphType>;
  using FloatType = typename GraphType::value_type;
  using SpinConfiguration = typename TransverseIsing::SpinConfiguration;

  auto str = std::string("ContinuousTimeIsing") + gtype_str;
  py::class_<TransverseIsing>(m, str.c_str(), py::module_local())
      .def(py::init<const SpinConfiguration &, const GraphType &, FloatType>(),
           "init_spin_config"_a, "init_interaction"_a, "gamma"_a)
      .def(py::init<const graph::Spins &, const GraphType &, FloatType>(),
           "init_spins"_a, "init_interaction"_a, "gamma"_a)
      .def(
          "reset_spins",
          [](TransverseIsing &self, const SpinConfiguration &init_spin_config) {
            self.reset_spins(init_spin_config);
          },
          "init_spin_config"_a)
      .def(
          "reset_spins",
          [](TransverseIsing &self, const graph::Spins &classical_spins) {
            self.reset_spins(classical_spins);
          },
          "classical_spins"_a)
      .def_readwrite("spin_config", &TransverseIsing::spin_config)
      .def_readonly("interaction", &TransverseIsing::interaction)
      .def_readonly("num_spins", &TransverseIsing::num_spins)
      .def_readonly("gamma", &TransverseIsing::gamma);

  // make_continuous_ising
  auto mkci_str = std::string("make_continuous_time_ising");
  m.def(
      mkci_str.c_str(),
      [](const graph::Spins &classical_spins, const GraphType &init_interaction,
         double gamma) {
        return system::make_continuous_time_ising(classical_spins,
                                                  init_interaction, gamma);
      },
      "classical_spins"_a, "init_interaction"_a, "gamma"_a);
}

#ifdef USE_CUDA

// ChimeraTransverseGPU
template <typename FloatType, std::size_t rows_per_block,
          std::size_t cols_per_block, std::size_t trotters_per_block>
inline void declare_ChimeraTranseverseGPU(py::module &m) {
  using ChimeraTransverseGPU =
      system::ChimeraTransverseGPU<FloatType, rows_per_block, cols_per_block,
                                   trotters_per_block>;
  py::class_<ChimeraTransverseGPU>(m, "ChimeraTransverseGPU",
                                   py::module_local())
      .def(py::init<const system::TrotterSpins &,
                    const graph::Chimera<FloatType> &, FloatType, int>(),
           "init_trotter_spins"_a, "init_interaction"_a, "gamma"_a,
           "device_num"_a = 0)
      .def(py::init<const graph::Spins &, const graph::Chimera<FloatType> &,
                    FloatType, size_t, int>(),
           "classical_spins"_a, "init_interaction"_a, "gamma"_a,
           "num_trotter_slices"_a, "device_num"_a = 0)
      .def(
          "reset_spins",
          [](ChimeraTransverseGPU &self,
             const system::TrotterSpins &init_trotter_spins) {
            self.reset_spins(init_trotter_spins);
          },
          "init_trotter_spins"_a)
      .def(
          "reset_spins",
          [](ChimeraTransverseGPU &self, const graph::Spins &classical_spins) {
            self.reset_spins(classical_spins);
          },
          "classical_spins"_a)
      .def_readwrite("gamma", &ChimeraTransverseGPU::gamma);

  // make_chimera_transverse_gpu
  m.def(
      "make_chimera_transverse_gpu",
      [](const system::TrotterSpins &init_trotter_spins,
         const graph::Chimera<FloatType> &init_interaction, double gamma,
         int device_num) {
        return system::make_chimera_transverse_gpu<
            rows_per_block, cols_per_block, trotters_per_block>(
            init_trotter_spins, init_interaction, gamma, device_num);
      },
      "init_trotter_spins"_a, "init_interaction"_a, "gamma"_a,
      "device_num"_a = 0);

  m.def(
      "make_chimera_transverse_gpu",
      [](const graph::Spins &classical_spins,
         const graph::Chimera<FloatType> &init_interaction, double gamma,
         size_t num_trotter_slices, int device_num) {
        return system::make_chimera_transverse_gpu<
            rows_per_block, cols_per_block, trotters_per_block>(
            classical_spins, init_interaction, gamma, num_trotter_slices,
            device_num);
      },
      "classical_spins"_a, "init_interaction"_a, "gamma"_a,
      "num_trotter_slices"_a, "device_num"_a = 0);
}

// ChimeraClassicalGPU
template <typename FloatType, std::size_t rows_per_block,
          std::size_t cols_per_block>
inline void declare_ChimeraClassicalGPU(py::module &m) {
  using ChimeraClassicalGPU =
      system::ChimeraClassicalGPU<FloatType, rows_per_block, cols_per_block>;
  py::class_<ChimeraClassicalGPU, typename ChimeraClassicalGPU::Base>(
      m, "ChimeraClassicalGPU", py::module_local())
      .def(py::init<const graph::Spins &, const graph::Chimera<FloatType> &,
                    int>(),
           "init_spin"_a, "init_interaction"_a, "device_num"_a = 0)
      .def(
          "reset_spins",
          [](ChimeraClassicalGPU &self, const graph::Spins &init_spin) {
            self.reset_spins(init_spin);
          },
          "init_spin"_a);

  // make_chimera_transverse_gpu
  m.def(
      "make_chimera_classical_gpu",
      [](const graph::Spins &init_spin,
         const graph::Chimera<FloatType> &init_interaction, int device_num) {
        return system::make_chimera_classical_gpu<rows_per_block,
                                                  cols_per_block>(
            init_spin, init_interaction, device_num);
      },
      "init_spin"_a, "init_interaction"_a, "device_num"_a = 0);
}

#endif

// Algorithm
template <template <typename> class Updater, typename System,
          typename RandomNumberEngine>
inline void declare_Algorithm_run(py::module &m,
                                  const std::string &updater_str) {
  auto str = std::string("Algorithm_") + updater_str + std::string("_run");
  using SystemType = typename system::get_system_type<System>::type;
  // with seed
  m.def(
      str.c_str(),
      [](System &system, std::size_t seed,
         const utility::ScheduleList<SystemType> &schedule_list,
         const std::function<void(
             const System &,
             const typename utility::UpdaterParameter<SystemType>::Tuple &)>
             &callback) {
        py::gil_scoped_release release;

        using Callback = std::function<void(
            const System &, const utility::UpdaterParameter<SystemType> &)>;
        RandomNumberEngine rng(seed);
        algorithm::Algorithm<Updater>::run(
            system, rng, schedule_list,
            callback ? [=](const System &system,
                           const utility::UpdaterParameter<SystemType>
                               &param) { callback(system, param.get_tuple()); }
                     : Callback(nullptr));

        py::gil_scoped_acquire acquire;
      },
      "system"_a, "seed"_a, "schedule_list"_a, "callback"_a = nullptr);

  // without seed
  m.def(
      str.c_str(),
      [](System &system, const utility::ScheduleList<SystemType> &schedule_list,
         const std::function<void(
             const System &,
             const typename utility::UpdaterParameter<SystemType>::Tuple &)>
             &callback) {
        py::gil_scoped_release release;

        using Callback = std::function<void(
            const System &, const utility::UpdaterParameter<SystemType> &)>;
        RandomNumberEngine rng(std::random_device{}());
        algorithm::Algorithm<Updater>::run(
            system, rng, schedule_list,
            callback ? [=](const System &system,
                           const utility::UpdaterParameter<SystemType>
                               &param) { callback(system, param.get_tuple()); }
                     : Callback(nullptr));

        py::gil_scoped_acquire acquire;
      },
      "system"_a, "schedule_list"_a, "callback"_a = nullptr);

  // schedule_list can be a list of tuples
  using TupleList = std::vector<std::pair<
      typename utility::UpdaterParameter<SystemType>::Tuple, std::size_t>>;

  // with seed
  m.def(
      str.c_str(),
      [](System &system, std::size_t seed, const TupleList &tuplelist,
         const std::function<void(
             const System &,
             const typename utility::UpdaterParameter<SystemType>::Tuple &)>
             &callback) {
        py::gil_scoped_release release;

        using Callback = std::function<void(
            const System &, const utility::UpdaterParameter<SystemType> &)>;
        RandomNumberEngine rng(seed);
        algorithm::Algorithm<Updater>::run(
            system, rng, utility::make_schedule_list<SystemType>(tuplelist),
            callback ? [=](const System &system,
                           const utility::UpdaterParameter<SystemType>
                               &param) { callback(system, param.get_tuple()); }
                     : Callback(nullptr));

        py::gil_scoped_acquire acquire;
      },
      "system"_a, "seed"_a, "tuplelist"_a, "callback"_a = nullptr);

  // without seed
  m.def(
      str.c_str(),
      [](System &system, const TupleList &tuplelist,
         const std::function<void(
             const System &,
             const typename utility::UpdaterParameter<SystemType>::Tuple &)>
             &callback) {
        py::gil_scoped_release release;
        using Callback = std::function<void(
            const System &, const utility::UpdaterParameter<SystemType> &)>;
        RandomNumberEngine rng(std::random_device{}());
        algorithm::Algorithm<Updater>::run(
            system, rng, utility::make_schedule_list<SystemType>(tuplelist),
            callback ? [=](const System &system,
                           const utility::UpdaterParameter<SystemType>
                               &param) { callback(system, param.get_tuple()); }
                     : Callback(nullptr));

        py::gil_scoped_acquire acquire;
      },
      "system"_a, "tuplelist"_a, "callback"_a = nullptr);
}

// utility
template <typename SystemType>
inline std::string repr_impl(const utility::UpdaterParameter<SystemType> &);

template <>
inline std::string
repr_impl(const utility::UpdaterParameter<system::classical_system> &obj) {
  return "(beta: " + std::to_string(obj.beta) + ")";
}

template <>
inline std::string repr_impl(
    const utility::UpdaterParameter<system::classical_constraint_system> &obj) {
  return "(beta: " + std::to_string(obj.beta) +
         ", lambda: " + std::to_string(obj.lambda) + ")";
}

template <>
inline std::string repr_impl(
    const utility::UpdaterParameter<system::transverse_field_system> &obj) {
  return "(beta: " + std::to_string(obj.beta) +
         ", s: " + std::to_string(obj.s) + ")";
}

inline void declare_ClassicalUpdaterParameter(py::module &m) {
  py::class_<utility::ClassicalUpdaterParameter>(m, "ClassicalUpdaterParameter",
                                                 py::module_local())
      .def(py::init<>())
      .def(py::init<double>(), "beta"_a)
      .def_readwrite("beta", &utility::ClassicalUpdaterParameter::beta)
      .def("__repr__", [](const utility::ClassicalUpdaterParameter &self) {
        return repr_impl(self);
      });
}

inline void declare_ClassicalConstraintUpdaterParameter(py::module &m) {
  py::class_<utility::ClassicalConstraintUpdaterParameter>(
      m, "ClassicalConstraintUpdaterParameter", py::module_local())
      .def(py::init<>())
      .def(py::init<double, double>(), "beta"_a, "lambda"_a)
      .def(py::init<const std::pair<double, double> &>(), "obj"_a)
      .def_readwrite("beta",
                     &utility::ClassicalConstraintUpdaterParameter::beta)
      .def_readwrite("lambda",
                     &utility::ClassicalConstraintUpdaterParameter::lambda)
      .def("__repr__",
           [](const utility::ClassicalConstraintUpdaterParameter &self) {
             return repr_impl(self);
           });
}

inline void declare_TransverseFieldUpdaterParameter(py::module &m) {
  py::class_<utility::TransverseFieldUpdaterParameter>(
      m, "TransverseFieldUpdaterParameter", py::module_local())
      .def(py::init<>())
      .def(py::init<double, double>(), "beta"_a, "s"_a)
      .def(py::init<const std::pair<double, double> &>(), "obj"_a)
      .def_readwrite("beta", &utility::TransverseFieldUpdaterParameter::beta)
      .def_readwrite("s", &utility::TransverseFieldUpdaterParameter::s)
      .def("__repr__",
           [](const utility::TransverseFieldUpdaterParameter &self) {
             return repr_impl(self);
           });
}

template <typename SystemType>
inline void declare_Schedule(py::module &m, const std::string &systemtype_str) {
  auto str = systemtype_str + "Schedule";
  py::class_<utility::Schedule<SystemType>>(m, str.c_str(), py::module_local())
      .def(py::init<>())
      .def(py::init<const std::pair<
               const utility::UpdaterParameter<SystemType> &, std::size_t> &>(),
           "obj"_a)
      .def_readwrite("one_mc_step", &utility::Schedule<SystemType>::one_mc_step)
      .def_readwrite("updater_parameter",
                     &utility::Schedule<SystemType>::updater_parameter)
      .def("__repr__", [](const utility::Schedule<SystemType> &self) {
        return "(" + repr_impl(self.updater_parameter) +
               " mcs: " + std::to_string(self.one_mc_step) + ")";
      });

  // define make_schedule_list
  m.def("make_schedule_list", &utility::make_schedule_list<SystemType>,
        "tuplelist"_a);
}

// result
// get_solution
template <typename System> inline void declare_get_solution(py::module &m) {
  m.def(
      "get_solution",
      [](const System &system) { return result::get_solution(system); },
      "system"_a);
}


template<typename FloatType>
void declare_BinaryPolynomialModel(py::module &m) {
   using BPM = graph::BinaryPolynomialModel<FloatType>;

   std::string name = std::string("BinaryPolynomialModel");
   auto py_class = py::class_<BPM>(m, name.c_str(), py::module_local());

   py_class.def(py::init<const std::vector<std::vector<typename BPM::IndexType>>&,
                         const std::vector<FloatType>&>(),
                         "key_list"_a, "value_list"_a);
  
   py_class.def("get_degree", &BPM::GetDegree);
   py_class.def("get_system_size", &BPM::GetSystemSize);
   py_class.def("get_index_list", &BPM::GetIndexList);
   py_class.def("get_index_map", &BPM::GetIndexMap);
   py_class.def("get_key_value_list", &BPM::GetKeyValueList);
   py_class.def("get_adjacency_list", &BPM::GetAdjacencyList);
   py_class.def("get_estimated_min_energy_difference", &BPM::GetEstimatedMinEnergyDifference);
   py_class.def("get_estimated_max_energy_difference", &BPM::GetEstimatedMaxEnergyDifference);
   py_class.def("calculate_energy", &BPM::CalculateEnergy);
}

template<typename FloatType>
void declare_IsingPolynomialModel(py::module &m) {
   using IPM = graph::IsingPolynomialModel<FloatType>;

   std::string name = std::string("IsingPolynomialModel");
   auto py_class = py::class_<IPM>(m, name.c_str(), py::module_local());

   py_class.def(py::init<std::vector<std::vector<typename IPM::IndexType>>&,
                         std::vector<FloatType>&>(),
                         "key_list"_a, "value_list"_a);
  
   py_class.def("get_degree", &IPM::GetDegree);
   py_class.def("get_system_size", &IPM::GetSystemSize);
   py_class.def("get_index_list", &IPM::GetIndexList);
   py_class.def("get_index_map", &IPM::GetIndexMap);
   py_class.def("get_key_value_list", &IPM::GetKeyValueList);
   py_class.def("get_adjacency_list", &IPM::GetAdjacencyList);
   py_class.def("get_estimated_min_energy_difference", &IPM::GetEstimatedMinEnergyDifference);
   py_class.def("get_estimated_max_energy_difference", &IPM::GetEstimatedMaxEnergyDifference);
   py_class.def("calculate_energy", &IPM::CalculateEnergy);
}


template<class ModelType>
void declare_SASampler(py::module &m, const std::string &post_name = "") {
   using SAS = sampler::SASampler<ModelType>;
   
   std::string name = std::string("SASampler") + post_name;

   auto py_class = py::class_<SAS>(m, name.c_str(), py::module_local());

   py_class.def(py::init<const ModelType&>(), "model"_a);

   py_class.def("set_num_sweeps", &SAS::SetNumSweeps, "num_sweeps"_a);
   py_class.def("set_num_reads", &SAS::SetNumReads, "num_reads"_a);
   py_class.def("set_num_threads", &SAS::SetNumThreads, "num_threads"_a);
   py_class.def("set_beta_min", &SAS::SetBetaMin, "beta_min"_a);
   py_class.def("set_beta_max", &SAS::SetBetaMax, "beta_max"_a);
   py_class.def("set_beta_min_auto", &SAS::SetBetaMinAuto);
   py_class.def("set_beta_max_auto", &SAS::SetBetaMaxAuto);
   py_class.def("set_update_method", &SAS::SetUpdateMethod, "update_method"_a);
   py_class.def("set_random_number_engine", &SAS::SetRandomNumberEngine, "random_number_engine"_a);
   py_class.def("set_temperature_schedule", &SAS::SetTemperatureSchedule, "temperature_schedule"_a);
   py_class.def("get_model", &SAS::GetModel);
   py_class.def("get_num_sweeps", &SAS::GetNumSweeps);
   py_class.def("get_num_reads", &SAS::GetNumReads);
   py_class.def("get_num_threads", &SAS::GetNumThreads);
   py_class.def("get_beta_min", &SAS::GetBetaMin);
   py_class.def("get_beta_max", &SAS::GetBetaMax);
   py_class.def("get_update_method", &SAS::GetUpdateMethod);
   py_class.def("get_random_number_engine", &SAS::GetRandomNumberEngine);
   py_class.def("get_temperature_schedule", &SAS::GetTemperatureSchedule);
   py_class.def("get_seed", &SAS::GetSeed);
   py_class.def("get_index_list", &SAS::GetIndexList);
   py_class.def("get_samples", &SAS::GetSamples);
   py_class.def("calculate_energies", &SAS::CalculateEnergies);
   py_class.def("sample", py::overload_cast<>(&SAS::Sample));
   py_class.def("sample", py::overload_cast<const std::uint64_t>(&SAS::Sample), "seed"_a);

   m.def("make_sa_sampler", [](const ModelType &model) {
      return sampler::make_sa_sampler(model);
   }, "model"_a);

}

void declare_UpdateMethod(py::module &m) {
   py::enum_<algorithm::UpdateMethod>(m, "UpdateMethod")
      .value("METROPOLIS", algorithm::UpdateMethod::METROPOLIS)
      .value("HEAT_BATH", algorithm::UpdateMethod::HEAT_BATH);
}

void declare_RandomNumberEngine(py::module &m) {
   py::enum_<algorithm::RandomNumberEngine>(m, "RandomNumberEngine")
      .value("MT", algorithm::RandomNumberEngine::MT)
      .value("MT_64", algorithm::RandomNumberEngine::MT_64)
      .value("XORSHIFT", algorithm::RandomNumberEngine::XORSHIFT);
}

void declare_TemperatureSchedule(py::module &m) {
   py::enum_<utility::TemperatureSchedule>(m, "TemperatureSchedule")
      .value("LINEAR", utility::TemperatureSchedule::LINEAR)
      .value("GEOMETRIC", utility::TemperatureSchedule::GEOMETRIC);
}


} // namespace openjij
