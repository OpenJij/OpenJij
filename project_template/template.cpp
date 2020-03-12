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
//
#include <graph/all.hpp>
#include <system/all.hpp>
#include <updater/all.hpp>
#include <algorithm/all.hpp>
#include <result/all.hpp>
#include <utility/schedule_list.hpp>
#include <random>

#include <iostream>

using namespace openjij;

int main(void){

    //generate dense graph with size N=5
    constexpr std::size_t N = 5;
    auto dense = graph::Dense<double>(N);

    //set interactions
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=0; j<N; j++){
            dense.J(i, j) = (i == j) ? 0 : -1;
        }
    }

    //set local fields
    for(std::size_t i=0; i<N; i++){
        dense.h(i) = -1;
    }

    //generate random engine (mersenne twister)
    auto rand_engine = std::mt19937(0x1234);

    //create classical Ising system
    auto system = system::make_classical_ising<system::USE_EIGEN>(dense.gen_spin(rand_engine), dense);

    //generate schedule list
    //from beta=0.1 to beta=100, 10 samples, 10 Monte Carlo step for each tempearature
    auto schedule_list = utility::make_classical_schedule_list(0.1, 100, 10, 10);

    //do annealing (updater: SingleSpinFlip)
    algorithm::Algorithm<updater::SingleSpinFlip>::run(system, rand_engine, schedule_list);

    //show spins
    std::cout << "The result spins are [";
    for(auto&& elem : result::get_solution(system)){
        std::cout << elem << " ";
    }

    std::cout << "]" << std::endl;
}
