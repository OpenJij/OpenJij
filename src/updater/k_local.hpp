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

#ifndef k_local_hpp
#define k_local_hpp

#include <system/k_local_polynomial.hpp>
#include <utility/schedule_list.hpp>

#include <random>


namespace openjij {
namespace updater {

template<typename System>
struct KLocal;


template<typename GraphType>
struct KLocal<system::KLocalPolynomial<GraphType>> {
  
   using CPIsing = system::KLocalPolynomial<GraphType>;
   
   //! @brief floating point type
   using FloatType = typename GraphType::value_type;
   
   template<typename RandomNumberEngine>
   inline static void update(CPIsing &system,
                             RandomNumberEngine &random_number_engine,
                             const utility::ClassicalUpdaterParameter &parameter
                             ) {
      
      auto urd = std::uniform_real_distribution<>(0, 1.0);
      
      for (std::size_t index_binary = 0; index_binary < system.num_spins; ++index_binary) {
         const auto dE = system.dE_single(index_binary);
         if (dE <= 0 || std::exp(-parameter.beta*dE) > urd(random_number_engine)) {
            system.update_system_single(index_binary);
            
            //system.spin[index_binary] = 1 - system.spin[index_binary];
            //system.reset_spins(system.spin);
            
         }
      }
      

   
   }
   
   
};



} // namespace updater
} // namespace openjij


#endif /* k_local_hpp */
