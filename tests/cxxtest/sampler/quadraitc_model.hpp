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

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Dense) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Sparse) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_CSRSparse) {
    using namespace openjij;

    //generate classical dense system
    const auto dense_interaction = generate_interaction<graph::Dense<double>>();
    //output sparse interaction
    Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat = dense_interaction.get_interactions().sparseView();
    const auto interaction = graph::CSRSparse<double>(sp_mat.template triangularView<Eigen::Upper>());
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Dense) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    std::size_t num_trotter_slices = 10;

    //generate random trotter spins
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto transverse_ising = system::make_transverse_ising(init_trotter_spins, interaction, 1.0);

    auto transverse_ising2 = system::make_transverse_ising(interaction.gen_spin(engine_for_spin), interaction, 1.0, 10);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Sparse) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    std::size_t num_trotter_slices = 10;

    //generate random trotter spins
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto transverse_ising = system::make_transverse_ising(init_trotter_spins, interaction, 1.0); //gamma = 1.0
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_CSRSparse) {
    using namespace openjij;

    //generate classical dense system
    const auto dense_interaction = generate_interaction<graph::Dense<double>>();
    //output sparse interaction
    Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat = dense_interaction.get_interactions().sparseView();
    const auto interaction = graph::CSRSparse<double>(sp_mat.template triangularView<Eigen::Upper>());
    auto engine_for_spin = std::mt19937(1);
    std::size_t num_trotter_slices = 10;

    //generate random trotter spins
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto transverse_ising = system::make_transverse_ising(init_trotter_spins, interaction, 1.0); //gamma = 1.0
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}


}
}
