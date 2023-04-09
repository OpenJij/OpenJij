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

#include <exception>
#include <tuple>
#include <vector>
#include <numeric>

#include <nlohmann/json.hpp>

#include <cimod/binary_polynomial_model.hpp>
#include <cimod/binary_quadratic_model.hpp>
#include <cimod/binary_quadratic_model_dict.hpp>

#include "openjij/graph/graph.hpp"

namespace openjij {
namespace graph {

using json = nlohmann::json;

/**
 * @brief parse json object from bqm.to_serializable
 *
 * @tparam FloatType
 * @tparam CimodDataType DataType of cimod (cimod::Dense, cimod::Sparse, or
 * cimod::Dict)
 * @param obj JSON object
 * @param relabel re-label variable_labels. Disable the option if the model has
 * specified topology (such as square lattice or chimera model). if the option
 * is disabled, IndexType of JSON must be an integer.
 *
 * @return BinaryQuadraticModel with IndexType=size_t
 *
 */
template <typename FloatType, typename CimodDataType>
inline auto json_parse(const json &obj, bool relabel = true) {

  using namespace cimod;
  // convert variable_labels
  json temp = obj;
  temp["type"] = "BinaryQuadraticModel";
  if (relabel) {
    // re-labeling
    std::size_t num_variables = temp["num_variables"];
    std::vector<size_t> variables(num_variables);
    // generate sequence numbers
    std::iota(variables.begin(), variables.end(), 0);
    temp["variable_labels"] = variables;
  }
  // make cimod object and apply to_serializable function
  auto bqm =
      BinaryQuadraticModel<size_t, FloatType, CimodDataType>::from_serializable(
          temp);
  return bqm.change_vartype(Vartype::SPIN, false);
}

template <typename FloatType>
inline auto json_parse_polynomial(const nlohmann::json &obj,
                                  const bool relabel = true) {

  if (obj.at("type") != "BinaryPolynomialModel") {
    throw std::runtime_error("Type must be \"BinaryPolynomialModel\".\n");
  }

  if (obj.at("poly_key_distance_list").size() !=
      obj.at("poly_value_list").size()) {
    throw std::runtime_error(
        "The sizes of key_list and value_list must match each other");
  }

  const std::size_t num_variables = obj["variables"].size();
  const int64_t num_interactions =
      static_cast<int64_t>(obj["poly_value_list"].size());

  if (num_interactions == 0) {
    throw std::runtime_error("The interaction is empty.");
  }

  const cimod::PolynomialKeyList<std::size_t> &poly_key_distance_list =
      obj["poly_key_distance_list"];
  const cimod::PolynomialValueList<FloatType> &poly_value_list =
      obj["poly_value_list"];
  cimod::PolynomialKeyList<Index> poly_key_list(num_interactions);

  if (relabel) {
    std::vector<std::size_t> sorted_variables;
    sorted_variables.resize(num_variables);
    std::iota(sorted_variables.begin(), sorted_variables.end(), 0);
#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions; ++i) {
      std::vector<Index> temp;
      for (const auto &it : poly_key_distance_list[i]) {
        temp.push_back(sorted_variables[it]);
      }
      std::sort(temp.begin(), temp.end());
      poly_key_list[i] = temp;
    }
  } else {
    const std::vector<Index> &variables = obj["variables"];
#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions; ++i) {
      std::vector<Index> temp;
      for (const auto &it : poly_key_distance_list[i]) {
        temp.push_back(variables[it]);
      }
      std::sort(temp.begin(), temp.end());
      poly_key_list[i] = temp;
    }
  }
  return std::tuple<cimod::PolynomialKeyList<Index>,
                    cimod::PolynomialValueList<FloatType>>(poly_key_list,
                                                           poly_value_list);
}

} // namespace graph
} // namespace openjij
