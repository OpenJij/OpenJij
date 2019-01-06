#include "../src/model.h"
#include "../src/updater/single_spin_flip.h"
#include "../src/sampler/sampler.h"
// #include <cxxjij/updater/single_spin_flip.h>
// #include <cxxjij/sampler/sampler.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>

using ::testing::ElementsAre;

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
    openjij::Spins(N, 1);
    openjij::SquareMatrix<double> int_mat{N, 0.0};
    int_mat(0, 1) = -1.0;
    int_mat(0, 0) = 1.0;

    ASSERT_EQ(int_mat(0, 1), -1.0);
    ASSERT_EQ(int_mat(0, 0), 1.0);
    EXPECT_ANY_THROW({int_mat(4, 0) = 1.0;});
}

TEST(OpenJijTest, energy){
	int N = 3;
	openjij::Spins spins(N, 1);
	openjij::SquareMatrix<double> int_mat{N, 0.0};
	// H = -s0 * s1 - s1 * s2 - s0 - s2
	int_mat(0, 0) = -1.0;
	int_mat(0, 1) = -1.0;
	int_mat(1, 0) = -1.0;
	int_mat(1, 2) = -1.0;
	int_mat(2, 1) = -1.0;
	int_mat(2, 2) = -1.0;

	openjij::sampler::Sampler samp(int_mat);
	double energy = samp.calc_energy(spins);
	ASSERT_EQ(energy, -4.0);

}
//---------------------------------------------

// ---------- Updater Test -------------------------
class UpdaterTest: public ::testing::Test{
    protected:
        const int L=3;
        int N;
        openjij::SquareMatrix<double> int_mat{1, 0.0};
        virtual void SetUp(){
            N = L*L;
            int_mat = openjij::SquareMatrix<double>(N, 0.0);
            for(int x=0; x < L; x++){
                for(int y=0; y < L; y++){
                    int pos = x + y*L;
                    int n_pos = (x+1)%L + y*L;
                    int d_pos = x + (y+1)%L * L;
                    int_mat(pos, n_pos) = -1.0;
                    int_mat(pos, d_pos) = -1.0;
                    int_mat(n_pos, pos) = -1.0;
                    int_mat(d_pos, pos) = -1.0;
                    int_mat(pos, pos) = -1.0;
                }
            }
        }
};

TEST_F(UpdaterTest, members){
    openjij::Spins spins(N, -1);
    openjij::updater::SingleSpinFlip ssf(int_mat);
    ASSERT_EQ(ssf.int_mat(0,0), -1.0);
    ASSERT_EQ(ssf.adj_mat.size(), N);
    for(int i=0; i < N; i++){
        ASSERT_EQ(ssf.adj_mat[i].size(), 4);
    }
}

TEST_F(UpdaterTest, single_spin_update){
    openjij::Spins spins(N, 1);
    spins[0] = -1;
    spins[N-1] = -1;
    openjij::updater::SingleSpinFlip ssf(int_mat);
    double beta = 10.0;
    EXPECT_NO_THROW({ssf.spins_update(spins, beta);});

    EXPECT_THAT(spins, ElementsAre(1,1,1,1,1,1,1,1,1));
}

TEST_F(UpdaterTest, quantum_single_spin_update){
    int trotter = 5;
    std::vector<openjij::Spins> q_spins(trotter, openjij::Spins(N, 1));
    q_spins[0][0] = -1;
    q_spins[2][N-1] = -1;
    q_spins[2][N] = -1;
    openjij::updater::SingleSpinFlip ssf(int_mat);
    double beta = 10.0;
    double gamma = 0.1;
    EXPECT_NO_THROW({ssf.quantum_spins_update(q_spins, beta, gamma);});

    EXPECT_THAT(q_spins[0], ElementsAre(1,1,1,1,1,1,1,1,1));
    EXPECT_THAT(q_spins[2], ElementsAre(1,1,1,1,1,1,1,1,1));
}

TEST(LargeSizeSSF, large_spins_update){
    const int L=30;
    int N = L * L;
    openjij::SquareMatrix<double> int_mat{1, 0.0};
    N = L*L;
    int_mat = openjij::SquareMatrix<double>(N, 0.0);
    for(int x=0; x < L; x++){
        for(int y=0; y < L; y++){
            int pos = x + y*L;
            int n_pos = (x+1)%L + y*L;
            int d_pos = x + (y+1)%L * L;
            int_mat(pos, n_pos) = -1.0;
            int_mat(pos, d_pos) = -1.0;
            int_mat(n_pos, pos) = -1.0;
            int_mat(d_pos, pos) = -1.0;
            int_mat(pos, pos) = -1.0;
        }
    }

    openjij::Spins spins(N, 1);
    spins[0] = -1;
    spins[3] = -1;
    openjij::updater::SingleSpinFlip ssf(int_mat, 1);
    double beta = 10.0;
    EXPECT_NO_THROW({ssf.spins_update(spins, beta);});

    for(int i=0; i < spins.size(); i++)
        ASSERT_EQ(spins[i], 1);
}

// -------- SamplerTest -----------------
class SamplerTest: public ::testing::Test{
    protected:
        const int L=3;
        int N;
        openjij::SquareMatrix<double> int_mat{1, 0.0};
        virtual void SetUp(){
            N = L*L;
            int_mat = openjij::SquareMatrix<double>(N, 0.0);
            for(int x=0; x < L; x++){
                for(int y=0; y < L; y++){
                    int pos = x + y*L;
                    int n_pos = (x+1)%L + y*L;
                    int d_pos = x + (y+1)%L * L;
                    int_mat(pos, n_pos) = -1.0;
                    int_mat(pos, d_pos) = -1.0;
                    int_mat(n_pos, pos) = -1.0;
                    int_mat(d_pos, pos) = -1.0;
                    int_mat(pos, pos) = -1.0;
                }
            }
        }
};

TEST_F(SamplerTest, insrance_test){
    openjij::sampler::Results resu;
    openjij::sampler::Sampler samp(int_mat);
    double beta = 1.0;
    int burn_out = 10;
    int observe_num = 10;
    EXPECT_NO_THROW({samp.sampling(beta, burn_out, observe_num, resu);});

    ASSERT_EQ(resu.states.size(), observe_num);
}

TEST_F(SamplerTest, sa_test){
    openjij::sampler::Results resu;
    openjij::sampler::Sampler samp(int_mat);
    double beta_min = 0.6;
    double beta_max = 2.0;
    int step_length = 10;
    int step_num = 10;
    int iter = 10;
    samp.simulated_annealing(beta_min, beta_max, step_length, step_num, iter, resu);

    ASSERT_EQ(resu.states.size(), iter);

    EXPECT_THAT(resu.states[0], ElementsAre(1,1,1,1,1,1,1,1,1));
}

TEST_F(SamplerTest, sqa_test){
    openjij::sampler::Results resu;
    openjij::sampler::Sampler samp(int_mat);
    double beta = 1.0;
    int trotter = 3;
    double gamma_min = 1.0;
    double gamma_max = 5.0;
    int step_length = 10;
    int step_num = 10;
    int iter = 10;
    samp.simulated_quantum_annealing(beta, gamma_min, gamma_max, trotter, step_length, step_num, iter, resu);
    ASSERT_EQ(resu.states.size(), iter);

    EXPECT_THAT(resu.states[0], ElementsAre(1,1,1,1,1,1,1,1,1));
}
