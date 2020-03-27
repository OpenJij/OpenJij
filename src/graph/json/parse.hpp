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

#ifndef OPENJIJ_GRAPH_JSON_PARSE_HPP__
#define OPENJIJ_GRAPH_JSON_PARSE_HPP__

#include <vector>
#include <tuple>
#include <nlohmann/json.hpp>
#include <exception>
#include <graph/cimod/src/binary_quadratic_model.hpp>
#include <numeric>

namespace openjij {
    namespace graph {

        using json = nlohmann::json;

        /**
         * @brief parse json object from bqm.to_serializable
         *
         * @tparam FloatType
         * @param obj
         *
         * @return BinaryQuadraticModel with IndexType=size_t
         * 
         */
        template<typename FloatType>
        inline auto json_parse(const json& obj){

            using namespace cimod;
            //convert variable_labels
            json temp = obj;
            std::size_t num_variables = temp["num_variables"];
            std::vector<size_t> variables(num_variables);
            //generate sequence numbers
            std::iota(variables.begin(), variables.end(), 0);
            temp["variable_labels"] = variables;
            //make cimod object and apply to_serializable function
            auto bqm = BinaryQuadraticModel<size_t, FloatType>::from_serializable(temp);
            bqm.change_vartype(Vartype::SPIN);
            bqm.print();
            return bqm;
        }
    } // namespace graph
} // namespace openjij

#endif
