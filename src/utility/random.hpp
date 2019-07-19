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

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include <utility>
#include <utility/gpu/handle_error.hpp>
#include <utility/gpu/memory.hpp>
#include <cassert>
#endif

namespace openjij {
    namespace utility {

        /**
         * @brief xorshift random generator for c++11 random
         */
        class Xorshift{
            public:
                using result_type = uint_fast32_t;

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

#ifdef USE_CUDA
        namespace cuda {
            template<typename FloatType>
                inline curandStatus_t curand_generate_uniform_impl(curandGenerator_t generator, FloatType *outputPtr, size_t num){
                    static_assert(std::is_floating_point<FloatType>::value, "FloatType must be float or double.");
                    static_assert(!std::is_same<FloatType, long double>::value, "long double is not supported");
                    return CURAND_STATUS_SUCCESS;
                }

            template<>
                inline curandStatus_t curand_generate_uniform_impl<float>(curandGenerator_t generator, float *outputPtr, size_t num){
                    return curandGenerateUniform(generator, outputPtr, num);
                }

            template<>
                inline curandStatus_t curand_generate_uniform_impl<double>(curandGenerator_t generator, double *outputPtr, size_t num){
                    return curandGenerateUniformDouble(generator, outputPtr, num);
                }

            template<typename FloatType, curandRngType_t rng_type>
                class CurandWrapper{
                    public:
                       CurandWrapper(std::size_t n, std::uint64_t seed)
                       : _dev_random(cuda::make_dev_unique<FloatType[]>(n)), _alloc_size(n){
                           //generate psudo_random_number generator
                           HANDLE_ERROR_CURAND(curandCreateGenerator(&_rng, rng_type));
                           //set seed
                           HANDLE_ERROR_CURAND(curandSetPseudoRandomGeneratorSeed(_rng, seed));
                       }

                       CurandWrapper(std::size_t n) : CurandWrapper(n, std::random_device{}()){}

                       CurandWrapper(CurandWrapper&& obj) noexcept
                       : _dev_random(std::move(obj._dev_random)),_alloc_size(std::move(obj._alloc_size)){
                           //move curand handler
                           this->_rng = obj._rng;
                           obj._rng = NULL;
                       }

                       ~CurandWrapper(){
                           //destroy generator
                           if(_rng != NULL)
                               HANDLE_ERROR_CURAND(curandDestroyGenerator(_rng));
                       }

                       inline const FloatType* get() const{
                           return _dev_random.get();
                       }

                       inline void generate_uniform(std::size_t n){
                           assert(n <= _alloc_size);
                           HANDLE_ERROR_CURAND(curand_generate_uniform_impl(_rng, _dev_random.get(), n));
                       }

                    private:
                        curandGenerator_t _rng;
                        cuda::unique_dev_ptr<FloatType[]> _dev_random;
                        std::size_t _alloc_size;
                };
        } // namespace cuda
#endif
    } // namespace utility
} // namespace openjij
#endif
