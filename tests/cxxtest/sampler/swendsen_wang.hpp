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

TEST(SwendsenWang, FindTrueGroundState_ClassicalIsing_Sparse_OneDimensionalIsing) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = [](){
        auto interaction = graph::Sparse<double>(num_system_size);
        interaction.J(0,1) = -1;
        interaction.J(1,2) = -1;
        interaction.J(2,3) = -1;
        interaction.J(3,4) = -1;
        interaction.J(4,5) = +1;
        interaction.J(5,6) = +1;
        interaction.J(6,7) = +1;
        interaction.h(0) = +1;
        return interaction;
    }();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_number_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_number_engine, schedule_list);

    EXPECT_EQ(openjij::graph::Spins({-1, -1, -1, -1, -1, +1, -1, +1}), result::get_solution(classical_ising));
}

TEST(SwendsenWang, FindTrueGroundState_ClassicalIsing_Sparse) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //with eigen implementation

    auto random_numder_engine = std::mt19937(1);

    //in general swendsen wang is not efficient in simulating frustrated systems. We need more annealing time.
    const auto schedule_list = openjij::utility::make_classical_schedule_list(0.01, 100.0, 100, 3000);

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}


// Continuous time Swendsen-Wang test
TEST(ContinuousTimeSwendsenWang, Place_Cuts) {
    using namespace openjij;
    using TimeType = typename system::ContinuousTimeIsing<graph::Sparse<double>>::TimeType;
    using CutPoint = typename system::ContinuousTimeIsing<graph::Sparse<double>>::CutPoint;

    std::vector<CutPoint> timeline;
    timeline.emplace_back(1.0, 1);
    timeline.emplace_back(2.0, 2);
    timeline.emplace_back(3.0, 2);
    timeline.emplace_back(4.0, 3);
    timeline.emplace_back(5.0, 3);
    timeline.emplace_back(6.0, 4);
    timeline.emplace_back(7.0, 4);

    std::vector<TimeType> cuts { 0.5, 1.5, 3.5, 4.5, 5.5, 7.5, 8.5 };
    timeline = updater::ContinuousTimeSwendsenWang<system::ContinuousTimeIsing<graph::Sparse<double>>>::create_timeline(timeline, cuts);

    std::vector<CutPoint> correct_timeline;
    correct_timeline.emplace_back(0.5, 4);
    correct_timeline.emplace_back(1.0, 1);
    correct_timeline.emplace_back(1.5, 1);
    correct_timeline.emplace_back(2.0, 2);
    correct_timeline.emplace_back(3.5, 2);
    correct_timeline.emplace_back(4.0, 3);
    correct_timeline.emplace_back(4.5, 3);
    correct_timeline.emplace_back(5.5, 3);
    correct_timeline.emplace_back(6.0, 4);
    correct_timeline.emplace_back(7.5, 4);
    correct_timeline.emplace_back(8.5, 4);

    EXPECT_EQ(timeline, correct_timeline);
}

TEST(ContinuousTimeSwendsenWang, Place_Cuts_Special_Case) {
    using namespace openjij;
    using TimeType = typename system::ContinuousTimeIsing<graph::Sparse<double>>::TimeType;
    using CutPoint = typename system::ContinuousTimeIsing<graph::Sparse<double>>::CutPoint;

    std::vector<CutPoint> timeline { {1.0, 1}, {2.0, 1} };

    std::vector<TimeType> cuts { };
    timeline = updater::ContinuousTimeSwendsenWang<system::ContinuousTimeIsing<graph::Sparse<double>>>::create_timeline(timeline, cuts);
    std::vector<CutPoint> correct_timeline { {1.0, 1} };

    EXPECT_EQ(timeline, correct_timeline);
}

/*************** currently disabled *************

TEST(ContinuousTimeSwendsenWang, FindTrueGroundState_ContinuousTimeIsing_Sparse_OneDimensionalIsing) {
    using namespace openjij;

    const auto interaction = [](){
        auto interaction = graph::Sparse<double>(num_system_size);
        interaction.J(0,1) = -1;
        interaction.J(1,2) = -1;
        interaction.J(2,3) = -1;
        interaction.J(3,4) = -1;
        interaction.J(4,5) = +1;
        interaction.J(5,6) = +1;
        interaction.J(6,7) = +1;
        interaction.h(0) = +1;
        return interaction;
    }();

    auto engine_for_spin = std::mt19937(1);
    const auto spins = interaction.gen_spin(engine_for_spin);

    auto ising = system::make_continuous_time_ising(spins, interaction, 1.0);

    //auto random_numder_engine = std::mt19937(1);
    //const auto schedule_list = utility::make_transverse_field_schedule_list(10, 100, 100);

    //algorithm::Algorithm<updater::ContinuousTimeSwendsenWang>::run(ising, random_numder_engine, schedule_list);

    //EXPECT_EQ(openjij::graph::Spins({-1, -1, -1, -1, -1, +1, -1, +1}), result::get_solution(ising));
}

TEST(ContinuousTimeSwendsenWang, FindTrueGroundState_ContinuousTimeIsing_Sparse) {
    using namespace openjij;

    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);

    const auto spins = interaction.gen_spin(engine_for_spin);

    auto ising = system::make_continuous_time_ising(spins, interaction, 1.0);

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = utility::make_transverse_field_schedule_list(10, 100, 3000);

    algorithm::Algorithm<updater::ContinuousTimeSwendsenWang>::run(ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(ising));
}

**********************************/


}
}
