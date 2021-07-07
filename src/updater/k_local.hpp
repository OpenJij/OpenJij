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

//! @brief K-local update for higher order unconstrained binary optimization (HUBO) problems with vartype being "BINARY".
//! @tparam GraphType graph type for Polynomial graph class
template<typename GraphType>
struct KLocal<system::KLocalPolynomial<GraphType>> {
   
      //! @brief floating point type
   using FloatType = typename GraphType::value_type;
   
   //! @brief Operate k-local update for HUBO.
   //! @param system ClPIsing&. Object of a KLocalPolynomial system.
   //! @param random_number_engine RandomNumberEngine&. Eandom number engine.
   //! @param parameter const utility::ClassicalUpdaterParameter&. Parameter object including inverse temperature \f\beta:=(k_B T)^{-1}\f.
   template<typename RandomNumberEngine>
   inline static void update(system::KLocalPolynomial<GraphType> &system,
                             RandomNumberEngine &random_number_engine,
                             const utility::ClassicalUpdaterParameter &parameter
                             ) {
      auto urd = std::uniform_real_distribution<>(0, 1.0);
     
     
      auto urd_int = std::uniform_int_distribution<std::size_t>(0, system.get_active_binaries().size() - 1);
      int count = 0;
      int count2 = 0;
      for (std::size_t i = 0; i < system.get_active_binaries().size(); ++i) {
         const auto index_binary = system.get_active_binaries_index(urd_int(random_number_engine));
         const FloatType dE_s = system.dE_single(index_binary);
         if (dE_s >= 0.0) {
            count++;
         }
         //if (dE_s == 0.0 && system.binaries[index_binary] == 1) {
         //   system.update_system_single(index_binary);
         //}
         //else if (dE_s == 0.0 && system.binaries[index_binary] == 0) {count2++;}
         if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
            system.update_system_single(index_binary);
         }
      }
      
      if (count == system.get_active_binaries().size()) {
         for (const auto &index_binary: system.get_active_binaries()) {
            const FloatType dE_s = system.dE_single(index_binary);
            if (dE_s == 0.0 && system.binaries[index_binary] == 1) {
               system.update_system_single(index_binary);
            }
         }
         int count_a = 0;
         int count_b = 0;
         for (std::size_t index_key = 0; index_key < system.get_keys().size(); ++index_key) {
            //if (system.GetPolyValue(index_key) < 0.0) {
               count_b++;
               const FloatType dE_i = system.dE_k_local(index_key);
               if (dE_i >= 0.0) {
                  count_a++;
               }
               if (dE_i < 0.0) {
                  system.update_system_k_local();
               }
               else {
                  system.reset_virtual_system();
               }
            //}
         }
         
         //if (count_a == count_b) {
         //   for (std::size_t index_key = 0; index_key < system.get_keys().size(); ++index_key) {
         //      if (system.GetPolyValue(index_key) > 0.0) {
         //         const FloatType dE_i = system.dE_k_local_rev(index_key);
         //         if (dE_i < 0.0) {
         //            system.update_system_k_local();
         //         }
         //         else {
         //            system.reset_virtual_system();
         //         }
         //      }
         //   }
         //}
         
      }
      
      //printf("%d/%d(%d)\n",count2, system.get_active_binaries().size(), count);
      
      
      
      /*
      auto urd_int = std::uniform_int_distribution<std::size_t>(0, system.get_active_binaries().size() - 1);
      for (std::size_t i = 0; i < system.get_active_binaries().size(); ++i) {
         const auto index_binary = system.get_active_binaries_index(urd_int(random_number_engine));
         const FloatType dE_s = system.dE_single(index_binary);
         if (dE_s == 0.0) {
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetZeroCount(index_key) == 0) {
                  system.update_system_single(index_binary);
                  break;
               }
               const FloatType dE_i = system.dE_k_local_p3(index_key);
               if (dE_i <= 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                  system.update_system_k_local();
               }
               else {
                  system.reset_virtual_system();
               }
            }
            if (system.dE_single(index_binary) == 0.0) {
               system.update_system_single(index_binary);
            }
         }
         else if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
            system.update_system_single(index_binary);
         }
      }
      */
      //for (const auto &index_binary: system.get_active_binaries()) {
      //   const FloatType dE_s = system.dE_single(index_binary);
      //   if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
      //      system.update_system_single(index_binary);
      //   }
      //}
      
      
      
      /*
      for (std::size_t index_key = 0; index_key < system.get_keys().size(); ++index_key) {
         if (system.GetZeroCount(index_key) != 0) {
            const FloatType dE_i = system.dE_k_local(index_key);
            if (dE_i < 0.0) {
               system.update_system_k_local();
            }
            else {
               system.reset_virtual_system();
               for (const auto index_binary: system.get_key(index_key)) {
                  const FloatType dE_s = system.dE_single(index_binary);
                  if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
                     system.update_system_single(index_binary);
                  }
               }
            }
         }
         else {
            for (const auto index_binary: system.get_key(index_key)) {
               const FloatType dE_s = system.dE_single(index_binary);
               if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
                  system.update_system_single(index_binary);
               }
            }
         }
      }
      */
      
      
      /*
      for (std::size_t index_key = 0; index_key < system.get_keys().size(); ++index_key) {
         const FloatType dE_i_p2 = system.template dE_k_local_p2<RandomNumberEngine>(index_key, random_number_engine);
         if (dE_i_p2 <= 0.0 || std::exp(-parameter.beta*dE_i_p2) > urd(random_number_engine)) {
            system.update_system_k_local();
         }
         else {
            system.reset_virtual_system();
         }
      }
      */

      /*
      auto urd_int = std::uniform_int_distribution<std::size_t>(0, system.get_active_binaries().size() - 1);
      int count = 0;
      for (std::size_t i = 0; i < system.get_active_binaries().size(); ++i) {
         const auto index_binary = system.get_active_binaries_index(urd_int(random_number_engine));
         const FloatType dE_s = system.dE_single(index_binary);
         if (dE_s == 0.0) {
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetZeroCount(index_key) == 0) {
                  system.update_system_single(index_binary);
                  //printf("SSF-Accept\n");
                  //system.print_active_binaries();
                  break;
               }
               if (system.GetPolyValue(index_key) >= 0.0) {
                  break;
               }
               const FloatType dE_i = system.dE_k_local(index_key);
               if (dE_i == 0.0 && system.binaries[index_binary] == 1) {
                  system.reset_virtual_system();
                  system.update_system_single(index_binary);
                  break;
               }
               if (dE_i < 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                  system.update_system_k_local();
                  count++;
                  //count++;
                  //printf("KLF-Accept: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                  //system.print_active_binaries();
               }
               else {
                  system.reset_virtual_system();
                  //printf("KLF-Reject: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                  //system.print_active_binaries();
               }
            }
         }
         else {
            if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
               system.update_system_single(index_binary);
            }
         }
         //if (count > 1) {
         //   return;
         //}
      }
      */
      
      

      
      /*
      for (const auto &index_binary: system.get_active_binaries()) {
         const FloatType dE_s = system.dE_single(index_binary);
         
         if (dE_s != 0.0) {
            //printf("dEs=%lf\n", dE_s);
            if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
               system.update_system_single(index_binary);
            }
         }
         else {
            //int count = 0;
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetZeroCount(index_key) == 0) {
                  system.update_system_single(index_binary);
                  //printf("SSF-Accept\n");
                  //system.print_active_binaries();
                  break;
               }
               if (system.GetPolyValue(index_key) >= 0.0) {
                  break;
               }
               const FloatType dE_i = system.dE_k_local(index_key);
               if (dE_i == 0.0 && system.binaries[index_binary] == 1) {
                  system.reset_virtual_system();
                  system.update_system_single(index_binary);
                  break;
               }
               else if (dE_i < 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                  system.update_system_k_local();
                  //count++;
                  //printf("KLF-Accept: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                  //system.print_active_binaries();
               }
               else {
                  system.reset_virtual_system();
                  //printf("KLF-Reject: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                  //system.print_active_binaries();
               }
            }
            //if (count > 1) {
            //   return;
            //}
         }
      }
      */
      
      /*
      for (const auto &index_binary: system.get_active_binaries()) {
         const FloatType dE_s = system.dE_single(index_binary);
         if (dE_s == 0.0) {
            if (system.binaries[index_binary] == 1) {
               for (const auto &index_key: system.get_adj(index_binary)) {
                  if (system.GetZeroCount(index_key) == 0) {
                     system.update_system_single(index_binary);
                     //printf("SSF-Accept\n");
                     //system.print_active_binaries();
                     break;
                  }
                  if (system.GetPolyValue(index_key) >= 0.0) {
                     break;
                  }
                  const FloatType dE_i = system.dE_k_local(index_key);
                  if (dE_i == 0.0) {
                     system.reset_virtual_system();
                     system.update_system_single(index_binary);
                     break;
                  }
                  else if (dE_i < 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                     system.update_system_k_local();
                     break;
                     //printf("KLF-Accept: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                     //system.print_active_binaries();
                  }
                  else {
                     system.reset_virtual_system();
                     //printf("KLF-Reject: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                     //system.print_active_binaries();
                  }
               }
            }
            else {
               if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
                  system.update_system_single(index_binary);
                  //printf("SSF-Accept: dEs=%lf\n", dE_s);
                  //system.print_active_binaries();
               }
            }
         }
         else {
            if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
               system.update_system_single(index_binary);
               //printf("SSF-Accept: dEs=%lf\n", dE_s);
               //system.print_active_binaries();
            }
         }
         
      }
      */
      
      
      /*
      static int ze = 0;
      static int tot = 0;
      for (const auto &index_binary: system.get_active_binaries()) {
         const FloatType dE_s = system.dE_single(index_binary);
         if (dE_s == 0.0) {
            ze++;
            tot++;
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetZeroCount(index_key) == 0) {
                  system.update_system_single(index_binary);
                  //printf("SSF-Accept\n");
                  //system.print_active_binaries();
                  break;
               }
               if (system.GetPolyValue(index_key) >= 0.0) {
                  break;
               }
               const FloatType dE_i = system.dE_k_local(index_key);
               if (dE_i == 0.0 && system.binaries[index_binary] == 1) {
                  system.reset_virtual_system();
                  system.update_system_single(index_binary);
                  break;
                  //printf("KLF-Reject,SSF-Accept: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
               }
               else if (dE_i < 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                  system.update_system_k_local();
                  break;
                  //printf("KLF-Accept: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                  //system.print_active_binaries();
               }
               else {
                  system.reset_virtual_system();
                  //printf("KLF-Reject: dEs=%lf, dE_i=%lf\n", dE_s, dE_i);
                  //system.print_active_binaries();
               }
            }
         }
         else {
            tot++;
            if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
               system.update_system_single(index_binary);
               //printf("SSF-Accept: dEs=%lf\n", dE_s);
               //system.print_active_binaries();
            }
            else {
               //printf("Skip\n");
            }
         }
      }
      printf("p=%lf\n", (double)ze/tot);
      */
      
      /*
      for (const auto &index_binary: system.get_active_binaries()) {
         const FloatType dE_s = system.dE_single(index_binary);
         
         if (dE_s != 0.0) {
            //printf("dEs=%lf\n", dE_s);
            if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
               system.update_system_single(index_binary);
            }
         }
         else {
            int count = 0;
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetPolyValue(index_key) >= 0.0) {
                  break;
               }
               const FloatType dE_i = system.dE_k_local(index_key);
               //printf("dEi=%lf\n", dE_i);
               if (dE_i == 0.0) {
                  count = 0;
                  system.reset_virtual_system();
                  system.update_system_single(index_binary);
                  break;
               }
               else {
                  if (dE_i <= 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                     system.update_system_k_local();
                     count++;
                  }
                  else {
                     system.reset_virtual_system();
                  }
               }
            }
            if (count > 1) {
               return;
            }
         }
      }
       */
      
      
      /*
      for (const auto &index_binary: system.get_active_binaries()) {
         if (system.binaries[index_binary] == 0) {
            const FloatType dE_s = system.dE_single(index_binary);
            if (dE_s != 0.0) {
               if (dE_s < 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
                  system.update_system_single(index_binary);
               }
            }
            else {
               for (const auto &index_key: system.get_adj(index_binary)) {
                  if (system.GetPolyValue(index_key) >= 0.0) {
                     break;
                  }
                  const FloatType dE_i = system.dE_k_local(index_key);
                  if (dE_i <= 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                     system.update_system_k_local();
                     return;
                  }
                  else {
                     system.reset_virtual_system();
                  }
               }
            }
         }
         else {
            const FloatType dE_s = system.dE_single(index_binary);
            if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
               system.update_system_single(index_binary);
            }
         }
      }
      */
     
      
      /*
      for (const auto &index_binary: system.get_active_binaries()) {
         const FloatType dE_s = system.dE_single(index_binary);
         if (system.count_call_updater%system.rate_call_k_local == 0 && dE_s == 0.0) {
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetPolyValue(index_key) >= 0.0) {
                  break;
               }
               if (system.GetZeroCount(index_key) > 0) {
                  const FloatType dE_i = system.dE_k_local(index_key);
                  //printf("count=%ld, key=%ld, 9=%d, 23=%d,dE_i=%lf[%lld<%lld], beta=%lf, E=%lf\n", index_key,system.count_call_updater, system.binaries[9], system.binaries[23], dE_i, system.GetZeroCount(index_key), system.GetKeySize(index_key), parameter.beta, system.energy());
                  if (dE_i <= 0.0 || std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                     system.update_system_k_local();
                  }
                  else {
                     system.reset_virtual_system();
                  }
               }
            }
         }
         else if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
            system.update_system_single(index_binary);
         }
         //printf("#######count=%ld, 9=%d, 23=%d,dE_s=%lf, beta=%lf, E=%lf\n",system.count_call_updater, system.binaries[9], system.binaries[23], dE_s,  parameter.beta, system.energy());
      }
      system.count_call_updater++;
      */
      
      /*
      //The very first
      for (const auto &index_binary: system.get_active_binaries()) {
         const FloatType dE_s = system.dE_single(index_binary);
         if (system.count_call_updater%system.rate_call_k_local == 0 && dE_s == 0.0) {
            for (const auto &index_key: system.get_adj(index_binary)) {
               if (system.GetPolyValue(index_key) >= 0.0) {
                  break;
               }
               const FloatType dE_i = system.dE_k_local(index_key);
               if (dE_i <= 0.0|| std::exp(-parameter.beta*dE_i) > urd(random_number_engine)) {
                  system.update_system_k_local();
               }
               else {
                  system.reset_virtual_system();
               }
            }
         }
         else if (dE_s <= 0.0 || std::exp(-parameter.beta*dE_s) > urd(random_number_engine)) {
            system.update_system_single(index_binary);
         }
      }
      system.count_call_updater++;
   */
      
      
      
   }
};

} // namespace updater
} // namespace openjij


#endif /* k_local_hpp */
