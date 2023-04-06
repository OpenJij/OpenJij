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

#ifdef USE_CUDA

TEST(GPU, glIdxConsistencyCheck_Chimera) {
    using namespace openjij;

    system::ChimeraInfo info{134,175,231};

    size_t a = 0;

    for(size_t t=0; t<info.trotters; t++){
        for(size_t r=0; r<info.rows; r++){
            for(size_t c=0; c<info.cols; c++){
                for(size_t i=0; i<info.chimera_unitsize; i++){
                    EXPECT_EQ(a, system::chimera_cuda::glIdx(info,r,c,i,t));
                    a++;
                }
            }
        }
    }
}

TEST(GPU, FindTrueGroundState_ChimeraTransverseGPU) {
    using namespace openjij;

    //generate classical chimera system
    const auto interaction = generate_chimera_interaction<float>();
    auto engine_for_spin = std::mt19937(1253);
    std::size_t num_trotter_slices = 1;
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto chimera_quantum_gpu = system::make_chimera_transverse_gpu<1,1,1>(init_trotter_spins, interaction, 1.0);

    auto random_number_engine = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(12356);

    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::GPU>::run(chimera_quantum_gpu, random_number_engine, schedule_list);

    graph::Spins res = result::get_solution(chimera_quantum_gpu);
    
    EXPECT_EQ(get_true_chimera_groundstate(interaction), result::get_solution(chimera_quantum_gpu));
}

TEST(GPU, FindTrueGroundState_ChimeraClassicalGPU) {
    using namespace openjij;

    //generate classical chimera system
    const auto interaction = generate_chimera_interaction<float>();
    auto engine_for_spin = std::mt19937(1264);
    const auto spin = interaction.gen_spin(engine_for_spin);

    auto chimera_classical_gpu = system::make_chimera_classical_gpu<1,1>(spin, interaction);

    auto random_number_engine = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(12356);

    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::GPU>::run(chimera_classical_gpu, random_number_engine, schedule_list);

    graph::Spins res = result::get_solution(chimera_classical_gpu);
    
    EXPECT_EQ(get_true_chimera_groundstate(interaction), result::get_solution(chimera_classical_gpu));
}

#endif


}
}
