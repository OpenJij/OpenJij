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

#ifndef OPENJIJ_GRAPH_BQM_VARTYPES_HPP__
#define OPENJIJ_GRAPH_BQM_VARTYPES_HPP__

#include <iostream>

namespace openjij {
    namespace graph {
        namespace bqm{
            /**
             * @brief Enum class for representing problem type
             * 
             */
            enum class Vartype
            {
                SPIN = 0,
                BINARY,
                NONE = -1,
            };

            /**
             * @brief Check that the variable has appropriate value
             * 
             * @param var 
             * @param vartype 
             * @return true or false
             */
            bool check_vartype
            (
             int32_t var,
             Vartype vartype
            )
            {
                if(vartype == Vartype::SPIN)
                {
                    if(var == 1 || var == -1)
                    {
                        return true;
                    }
                    else
                    {
                        std::cerr << "Spin variable must be +1 or -1." << std::endl;
                        return false;
                    }
                }
                else if(vartype == Vartype::BINARY)
                {
                    if(var == 1 || var == 0)
                    {
                        return true;
                    }
                    else
                    {
                        std::cerr << "Binary variable must be 1 or 0." << std::endl;
                        return false;
                    }
                }
                else
                {
                    std::cerr << "Unknown variable type." << std::endl;
                }
            };
        } // namespace bqm 
    } // namespace graph 
} // namespace openjij

#endif
