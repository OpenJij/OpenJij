// #include "../src/model.h"
// #include "../src/updater/single_spin_flip.h"
// #include "../src/sampler/sampler.h"
// #include <cxxjij/updater/single_spin_flip.h>
// #include <cxxjij/sampler/sampler.h>

#include "../src/graph/dense.h"
#if 0
#include "../src/graph/sparse.h"
#include "../src/graph/square.h"
#include "../src/graph/chimera.h"
#include "../src/system/classical_ising.h"
#include "../src/system/quantum_ising.h"
#include "../src/updater/classical_updater.h"
#include "../src/updater/quantum_updater.h"
#include "../src/algorithm/sa.h"
#include "../src/algorithm/sqa.h"
#endif

#include "../src/system/system.hpp"
#include "../src/system/schedule_list.hpp"
#include "../src/algorithm/single_spin_flip.hpp"
#include "../src/utility/create_geometric_progression.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>

#include <utility>
#include <numeric>
#include <tuple>

using ::testing::ElementsAre;
using ::testing::_;

TEST(Call_SimulatedAnnealing, SingleSpinFlip) {
    auto spins = openjij::graph::Spins(10);
    auto interactions = openjij::graph::Dense<double>(10);

    auto system = openjij::system::ClassicalIsing(spins, interactions);
    auto schedule_list = openjij::system::ClassicalScheduleList();
    schedule_list.push_back(std::make_pair(1, 3.45));

    auto single_spin_flip = openjij::algorithm::Algorithm<openjij::algorithm::SingleSpinFlip>{};

    single_spin_flip.run(system, schedule_list);
}

TEST(Call_QuantumSimulatedAnnealing, SingleSpinFlip) {
    auto spins = openjij::graph::TrotterSpins(10);
    auto interactions = openjij::graph::Dense<double>(10);

    auto system = openjij::system::QuantumIsing(spins, interactions);
    auto schedule_list = openjij::system::QuantumScheduleList();
    schedule_list.push_back(std::make_pair(2, std::make_pair(3.45, 67.89)));

    auto single_spin_flip = openjij::algorithm::Algorithm<openjij::algorithm::SingleSpinFlip>{};

    single_spin_flip.run(system, schedule_list);
}

TEST(Check_ScheduleListSize, create_sa_schedule_list) {
    const auto list_size = 10;

    const auto schedule_list = openjij::system::create_sa_schedule_list(list_size, 1, 1, 1);

    EXPECT_EQ(schedule_list.size(), list_size);
}

TEST(Set_InverseTemperatureGeometricProgressionToScheduleList, create_sa_schedule_list) {
    const auto list_size = 8;
    const double beta_min = 2.0;
    const double beta_max = 256.0;

    const auto schedule_list = openjij::system::create_sa_schedule_list(list_size, 1, beta_min, beta_max);

    const auto expected_list = [&list_size, &beta_min](){
        auto list = std::vector<double>(list_size);
        const auto initial_value = beta_min;
        const auto ratio = beta_min;

        openjij::utility::make_geometric_progression(list.begin(), list.end(), initial_value, ratio);
        return list;
    }();

    for (auto i = 0; i < list_size; ++i) {
        EXPECT_EQ(schedule_list[i].second, expected_list[i]);
    }
}

TEST(Set_LoopStepPerOneMonteCarloStep, create_sa_schedule_list) {
    const auto list_size = 8;
    const auto one_mc_step = 12;

    const auto schedule_list = openjij::system::create_sa_schedule_list(list_size, one_mc_step, 1, 1);

    for (auto&& schedule : schedule_list) {
        EXPECT_EQ(schedule.first, one_mc_step);
    }
}

#if 0
template<typename num> void show_matrix(std::vector<std::vector<num>>& mat){
    for(std::vector<num> vec: mat){
        for(num v: vec)
            std::cout << v << " ";
        std::cout << std::endl;
    }
}

// --------- openjij basics test --------------
TEST(OpenJijTest, spin_generator){
    int N = 4;
    openjij::graph::Dense<double> dense(N);
    openjij::graph::Spins spin = dense.gen_spin(1);
    openjij::graph::Spins new_spin = dense.gen_spin(1);
    EXPECT_EQ(spin, new_spin);

    new_spin = dense.gen_spin(10);
    EXPECT_NE(spin, new_spin);

}

TEST(OpenJijTest, classicalIsing_initialize){
    size_t N=10;
    openjij::graph::Dense<double> dense(N);
    openjij::system::ClassicalIsing cising(dense);
    openjij::graph::Spins spins = cising.get_spins();

    cising.initialize_spins();

    openjij::graph::Spins new_spins = cising.get_spins();

    EXPECT_NE(spins, new_spins);

    // input initial state
    openjij::graph::Spins init_spins(N, 1);
    openjij::system::ClassicalIsing input_cising(dense, init_spins);
    spins = input_cising.get_spins();
    EXPECT_EQ(init_spins, spins); 
}

TEST(OpenJijTest, quantumIsing_initialize){
    size_t N=10;
    size_t trotter = 5;
    openjij::graph::Dense<double> dense(N);
    openjij::system::QuantumIsing qising(dense, trotter);
    openjij::system::TrotterSpins spins = qising.get_spins();

    qising.initialize_spins();

    openjij::system::TrotterSpins new_spins = qising.get_spins();

    for(int i=0; i < spins.size(); i++){
        EXPECT_NE(spins[i], new_spins[i]);
    }

    // input initial state
    openjij::graph::Spins init_classical_spins(N, 1);
    openjij::system::QuantumIsing input_qising(dense, trotter, init_classical_spins);
    spins = input_qising.get_spins();
    for(openjij::graph::Spins c_spin : spins){
        EXPECT_EQ(init_classical_spins, c_spin);
    }

    openjij::graph::Spins c_spin(N, 1);
    qising.set_spins(c_spin);
    for(openjij::graph::Spins cc_spins : qising.get_spins()){
        EXPECT_EQ(cc_spins, c_spin);
    }

    openjij::graph::Spins small_state(5, 1);
    ASSERT_ANY_THROW(qising.set_spins(small_state));
}

TEST(OpenJijTest, sa_temperature_schedule){
    class MockClassicalSystem : public openjij::updater::ClassicalUpdater {
        public:
            MOCK_METHOD2(update, double(double beta, const std::string& algo));
    } mock_classical_system;

    // Make schedule
    openjij::algorithm::Schedule schedule;

    schedule.emplace_back(std::make_pair(1.5, 1));
    schedule.emplace_back(std::make_pair(2.5, 2));
    schedule.emplace_back(std::make_pair(3.5, 3));

    // calculate the summation of second elements of std::vector<std::pair<double, size_t>>
    auto add_second = [](const int n, const std::pair<double, size_t>& p){ return n + p.second; };
    const int total_call_times = std::accumulate(schedule.begin(), schedule.end(), 0, add_second);

    EXPECT_CALL(mock_classical_system, update(_, "")).Times(total_call_times);

    openjij::algorithm::SA sa(schedule);
    sa.run(mock_classical_system);
}

TEST(OpenJijTest, qsa_transverse_schedule){
    class MockQuantumSystem : public openjij::updater::QuantumUpdater {
        public:
            MOCK_METHOD4(update, double(double beta, double gamma, double s, const std::string& algo));
    } mock_quantum_system;

    // Make schedule
    openjij::algorithm::Schedule schedule;

    schedule.emplace_back(std::make_pair(1.5, 1));
    schedule.emplace_back(std::make_pair(2.5, 2));
    schedule.emplace_back(std::make_pair(3.5, 3));

    // calculate the summation of second elements of std::vector<std::pair<double, size_t>>
    auto add_second = [](const int n, const std::pair<double, size_t>& p){ return n + p.second; };
    const int total_call_times = std::accumulate(schedule.begin(), schedule.end(), 0, add_second);

    EXPECT_CALL(mock_quantum_system, update(_, _, _, "")).Times(total_call_times);

    const double beta = 0.1;
    const double gamma = 1.0;
    openjij::algorithm::SQA sqa(beta,gamma,schedule);
    sqa.run(mock_quantum_system);
}

TEST(OpenJijTest, times_sa_call_classical_updater){
    const double beta_min = 1.0;
    const double beta_max = 2.0;

    class MockClassicalSystem : public openjij::updater::ClassicalUpdater {
        public:
            MOCK_METHOD2(update, double(double beta, const std::string& algo));
    } mock_classical_system;

    // Case: step_length != step_num
    {
        const int step_length = 3, step_num = 5;
        EXPECT_CALL(mock_classical_system, update(_, "")).Times(step_length*step_num);

        openjij::algorithm::SA sa(beta_min, beta_max, step_length, step_num);
        sa.run(mock_classical_system);
    }

    // Case: step_length == step_num
    {
        const int step_length = 10, step_num = 10;
        EXPECT_CALL(mock_classical_system, update(_, "")).Times(step_length*step_num);

        openjij::algorithm::SA sa(beta_min, beta_max, step_length, step_num);
        sa.run(mock_classical_system);
    }
}

TEST(OpenJijTest, times_sqa_call_quantum_updater){
    const double beta = 1.0;
    const double gamma = 2.0;

    class MockQuantumSystem : public openjij::updater::QuantumUpdater {
        public:
            MOCK_METHOD4(update, double(double beta, double gamma, const double s, const std::string& algo));
    } mock_quantum_system;

    // Case: step_length != step_num
    {
        const int step_length = 3, step_num = 5;
        EXPECT_CALL(mock_quantum_system, update(_, _, _, "")).Times(step_length*step_num);

        openjij::algorithm::SQA sqa(beta, gamma, step_length, step_num);
        sqa.run(mock_quantum_system);
    }

    // Case: step_length == step_num
    {
        const int step_length = 10, step_num = 10;
        EXPECT_CALL(mock_quantum_system, update(_, _, _, "")).Times(step_length*step_num);

        openjij::algorithm::SQA sqa(beta, gamma, step_length, step_num);
        sqa.run(mock_quantum_system);
    }
}
#endif
