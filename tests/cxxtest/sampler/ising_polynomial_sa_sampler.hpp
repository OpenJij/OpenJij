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


namespace openjij {
namespace test {

TEST(Sampler, SASamplerOperationIsingPolynomial) {
   
   using FloatType = double;
   using Tup = utility::AnyTupleType;
   using IPM = graph::IsingPolynomialModel<FloatType>;
   
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {1, 1},
      {"a", "a"},
      {Tup{2, "b"}, Tup{2, "b"}},
      {1, 2},
      {"a", 1},
      {Tup{2, "b"}, Tup{2, "a"}}
   };
      
   std::vector<FloatType> value_list = {
      +4.0,
      +2.0,
      +3.0,
      -1.0,
      -1.5,
      -2.5
   };
   
   const auto ipm = IPM{key_list, value_list};
   
   auto sa_sampler = sampler::SASampler{ipm};
   sa_sampler.SetBetaMaxAuto();
   sa_sampler.SetBetaMinAuto();
   
   std::vector<algorithm::RandomNumberEngine> engine_list = {
      algorithm::RandomNumberEngine::XORSHIFT,
      algorithm::RandomNumberEngine::MT,
      algorithm::RandomNumberEngine::MT_64
   };
   
   std::vector<algorithm::UpdateMethod> updater_list = {
      algorithm::UpdateMethod::METROPOLIS,
      algorithm::UpdateMethod::HEAT_BATH
   };
   
   std::vector<utility::TemperatureSchedule> schedule_list = {
      utility::TemperatureSchedule::LINEAR,
      utility::TemperatureSchedule::GEOMETRIC
   };
   
   for (const auto &engine: engine_list) {
      sa_sampler.SetRandomNumberEngine(engine);
      for (const auto &algorithm: updater_list) {
         sa_sampler.SetUpdateMethod(algorithm);
         for (const auto &schedule: schedule_list) {
            sa_sampler.SetTemperatureSchedule(schedule);
            EXPECT_NO_THROW(sa_sampler.Sample());
            EXPECT_NO_THROW(sa_sampler.CalculateEnergies());
         }
      }
   }

}


}
}
