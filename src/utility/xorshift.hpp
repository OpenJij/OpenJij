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

#ifndef OPENJIJ_UTILITY_XORSHIFT_HPP__
#define OPENJIJ_UTILITY_XORSHIFT_HPP__

#include <random>
#include <climits>

namespace openjij {
    namespace utility {

        /**
         * @brief xorshift random generator for c++11 random
         */
        class Xorshift{
            public:

                /**
                 * @brief returns minimum value 
                 *
                 * @return minimum value
                 */
                inline static constexpr unsigned min(){
                    return 0u;
                }

                /**
                 * @brief returns maximum value
                 *
                 * @return maximum value
                 */
                inline static constexpr unsigned max(){
                    return UINT_MAX;
                }

                /**
                 * @brief generate random number
                 *
                 * @return random number
                 */
                inline unsigned operator()(){ 
                    unsigned t = x ^ (x << 11);
                    x = y;
                    y = z;
                    z = w;
                    return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
                }

                /**
                 * @brief Xorshift constructor
                 */
                Xorshift(){
                    std::random_device rd;
                    w=rd();
                }

                /**
                 * @brief Xorshift constructor with seed
                 *
                 * @param s seed
                 */
                Xorshift(unsigned s){
                    w=s;
                } 
            private:
                unsigned x=123456789u,y=362436069u,z=521288629u,w;
        };
    } // namespace utility
} // namespace openjij
#endif
