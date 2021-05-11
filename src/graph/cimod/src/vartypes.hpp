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

/**
 * @file vartypes.hpp
 * @author Fumiya Watanabe
 * @brief Definition of variable type
 * @version 1.0.0
 * @date 2020-03-17
 * 
 * @copyright Copyright (c) Jij Inc. 2020
 * 
 */

#ifndef VARTYPES_HPP__
#define VARTYPES_HPP__

#include <iostream>


namespace cimod
{
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
inline bool check_vartype
(
    const int32_t &var,
    const Vartype &vartype
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
        return false;
    }
}
}
#endif
