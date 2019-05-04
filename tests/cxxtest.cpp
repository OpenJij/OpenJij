// #include "../src/model.h"
// #include "../src/updater/single_spin_flip.h"
// #include "../src/sampler/sampler.h"
// #include <cxxjij/updater/single_spin_flip.h>
// #include <cxxjij/sampler/sampler.h>

#include "../src/graph/dense.h"
#include "../src/graph/sparse.h"
#include "../src/graph/square.h"
#include "../src/graph/chimera.h"
#include "../src/system/classical_ising.h"
#include "../src/system/quantum_ising.h"
#include "../src/updater/classical_updater.h"
#include "../src/updater/quantum_updater.h"
#include "../src/algorithm/sa.h"
#include "../src/algorithm/sqa.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>

#include <utility>
#include <numeric>

using ::testing::ElementsAre;
using ::testing::_;

template<typename num> void show_matrix(std::vector<std::vector<num>>& mat){
    for(std::vector<num> vec: mat){
        for(num v: vec)
            std::cout << v << " ";
        std::cout << std::endl;
    }
}

// --------- openjij basics test --------------
TEST(OpenJijTest, spin_matrix){
    int N = 4;
    // openjij::Spins(N, 1);
    // openjij::SquareMatrix<double> int_mat{N, 0.0};
    // int_mat(0, 1) = -1.0;
    // int_mat(0, 0) = 1.0;

    // ASSERT_EQ(int_mat(0, 1), -1.0);
    // ASSERT_EQ(int_mat(0, 0), 1.0);
    // EXPECT_ANY_THROW({int_mat(4, 0) = 1.0;});
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

// // ---------- Updater Test -------------------------
// class UpdaterTest: public ::testing::Test{
//     protected:
//         const int L=3;
//         int N;
//         openjij::SquareMatrix<double> int_mat{1, 0.0};
//         virtual void SetUp(){
//             N = L*L;
//             int_mat = openjij::SquareMatrix<double>(N, 0.0);
//             for(int x=0; x < L; x++){
//                 for(int y=0; y < L; y++){
//                     int pos = x + y*L;
//                     int n_pos = (x+1)%L + y*L;
//                     int d_pos = x + (y+1)%L * L;
//                     int_mat(pos, n_pos) = -1.0;
//                     int_mat(pos, d_pos) = -1.0;
//                     int_mat(n_pos, pos) = -1.0;
//                     int_mat(d_pos, pos) = -1.0;
//                     int_mat(pos, pos) = -1.0;
//                 }
//             }
//         }
// };

// TEST_F(UpdaterTest, members){
//     openjij::Spins spins(N, -1);
//     openjij::updater::SingleSpinFlip ssf(int_mat);
//     ASSERT_EQ(ssf.int_mat(0,0), -1.0);
//     ASSERT_EQ(ssf.adj_mat.size(), N);
//     for(int i=0; i < N; i++){
//         ASSERT_EQ(ssf.adj_mat[i].size(), 4);
//     }
// }

// TEST_F(UpdaterTest, single_spin_update){
//     openjij::Spins spins(N, 1);
//     spins[0] = -1;
//     spins[N-1] = -1;
//     openjij::updater::SingleSpinFlip ssf(int_mat);
//     double beta = 10.0;
//     EXPECT_NO_THROW({ssf.spins_update(spins, beta);});

//     EXPECT_THAT(spins, ElementsAre(1,1,1,1,1,1,1,1,1));
// }

// TEST_F(UpdaterTest, quantum_single_spin_update){
//     int trotter = 5;
//     std::vector<openjij::Spins> q_spins(trotter, openjij::Spins(N, 1));
//     q_spins[0][0] = -1;
//     q_spins[2][N-1] = -1;
//     q_spins[2][N] = -1;
//     openjij::updater::SingleSpinFlip ssf(int_mat);
//     double beta = 10.0;
//     double gamma = 0.1;
//     EXPECT_NO_THROW({ssf.quantum_spins_update(q_spins, beta, gamma);});

//     EXPECT_THAT(q_spins[0], ElementsAre(1,1,1,1,1,1,1,1,1));
//     EXPECT_THAT(q_spins[2], ElementsAre(1,1,1,1,1,1,1,1,1));
// }

// TEST(LargeSizeSSF, large_spins_update){
//     const int L=30;
//     int N = L * L;
//     openjij::SquareMatrix<double> int_mat{1, 0.0};
//     N = L*L;
//     int_mat = openjij::SquareMatrix<double>(N, 0.0);
//     for(int x=0; x < L; x++){
//         for(int y=0; y < L; y++){
//             int pos = x + y*L;
//             int n_pos = (x+1)%L + y*L;
//             int d_pos = x + (y+1)%L * L;
//             int_mat(pos, n_pos) = -1.0;
//             int_mat(pos, d_pos) = -1.0;
//             int_mat(n_pos, pos) = -1.0;
//             int_mat(d_pos, pos) = -1.0;
//             int_mat(pos, pos) = -1.0;
//         }
//     }

//     openjij::Spins spins(N, 1);
//     spins[0] = -1;
//     spins[3] = -1;
//     openjij::updater::SingleSpinFlip ssf(int_mat, 1);
//     double beta = 10.0;
//     EXPECT_NO_THROW({ssf.spins_update(spins, beta);});

//     for(int i=0; i < spins.size(); i++)
//         ASSERT_EQ(spins[i], 1);
// }

// // -------- SamplerTest -----------------
// class SamplerTest: public ::testing::Test{
//     protected:
//         const int L=3;
//         int N;
//         openjij::SquareMatrix<double> int_mat{1, 0.0};
//         virtual void SetUp(){
//             N = L*L;
//             int_mat = openjij::SquareMatrix<double>(N, 0.0);
//             for(int x=0; x < L; x++){
//                 for(int y=0; y < L; y++){
//                     int pos = x + y*L;
//                     int n_pos = (x+1)%L + y*L;
//                     int d_pos = x + (y+1)%L * L;
//                     int_mat(pos, n_pos) = -1.0;
//                     int_mat(pos, d_pos) = -1.0;
//                     int_mat(n_pos, pos) = -1.0;
//                     int_mat(d_pos, pos) = -1.0;
//                     int_mat(pos, pos) = -1.0;
//                 }
//             }
//         }
// };

// TEST_F(SamplerTest, insrance_test){
//     openjij::sampler::Results resu;
//     openjij::sampler::Sampler samp(int_mat);
//     double beta = 1.0;
//     int burn_out = 10;
//     int observe_num = 10;
//     EXPECT_NO_THROW({samp.sampling(beta, burn_out, observe_num, resu);});

//     ASSERT_EQ(resu.states.size(), observe_num);
// }

// TEST_F(SamplerTest, sa_test){
//     openjij::sampler::Results resu;
//     openjij::sampler::Sampler samp(int_mat);
//     double beta_min = 0.6;
//     double beta_max = 2.0;
//     int step_length = 10;
//     int step_num = 10;
//     int iter = 10;
//     samp.simulated_annealing(beta_min, beta_max, step_length, step_num, iter, resu);

//     ASSERT_EQ(resu.states.size(), iter);

//     EXPECT_THAT(resu.states[0], ElementsAre(1,1,1,1,1,1,1,1,1));
// }

// TEST_F(SamplerTest, sqa_test){
//     openjij::sampler::Results resu;
//     openjij::sampler::Sampler samp(int_mat);
//     double beta = 1.0;
//     int trotter = 3;
//     double gamma_min = 1.0;
//     double gamma_max = 5.0;
//     int step_length = 10;
//     int step_num = 10;
//     int iter = 10;
//     samp.simulated_quantum_annealing(beta, gamma_min, gamma_max, trotter, step_length, step_num, iter, resu);
//     ASSERT_EQ(resu.states.size(), iter);

//     EXPECT_THAT(resu.states[0], ElementsAre(1,1,1,1,1,1,1,1,1));
// }
