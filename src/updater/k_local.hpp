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
      
      /*
      //Here, we check if the binary variables associated with specified interactions should be all 1.
      for (std::size_t index = 0; index < system.GetNumInteractions(); ++index) {
        const auto dE = system.dE_k_local(index);
         if (system.GetPolyKey(index).size() > 1 && (dE <= 0 || std::exp(-parameter.beta*dE) > urd(random_number_engine))) {
            for (const auto &index: system.GetPolyKey(index)) {
               if (system.spin[index] == 0) {
                  system.update_system(index);
               }
            }
         }
      }
      */
      
      
      //Here, we implement traditional single spin flip
      //TO DO: Move to the single spin flip updater after performance check
      for (std::size_t index = 0; index < system.num_spins; ++index) {
         const auto dE = system.dE_single(index);
         if (dE <= 0 || std::exp(-parameter.beta*dE) > urd(random_number_engine)) {
            system.update_system(index);
         }
      }
       
   
   }
   
   
};



} // namespace updater
} // namespace openjij


#endif /* k_local_hpp */
