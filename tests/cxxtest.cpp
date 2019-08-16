// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h>
// include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>

// include OpenJij
#include <graph/all.hpp>
#include <system/all.hpp>
#include <updater/all.hpp>
#include <algorithm/algorithm.hpp>
#include <result/get_solution.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>
#include <utility/random.hpp>
#include <utility/gpu/memory.hpp>
#include <utility/gpu/cublas.hpp>

// #####################################
// helper functions
// #####################################
/**
 * @brief generate interaction
 *
 * @return classical interaction which represents specific optimization problem
 */

static constexpr std::size_t num_system_size = 8;

#define TEST_CASE_INDEX 1

#include "./testcase.hpp"


static openjij::utility::ClassicalScheduleList generate_schedule_list(){
    return openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 100);
}

static openjij::utility::TransverseFieldScheduleList generate_tfm_schedule_list(){
    return openjij::utility::make_transverse_field_schedule_list(10, 100, 100);
}

// #####################################
// tests
// #####################################

//graph tests

TEST(Graph, DenseGraphCheck){
    using namespace openjij::graph;
    using namespace openjij;

    std::size_t N = 500;
    Dense<double> a(N);
    auto r = utility::Xorshift(1234);
    auto urd = std::uniform_real_distribution<>{-10, 10};
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            a.J(i, j)  = urd(r);
        }
    }

    r = utility::Xorshift(1234);

    // check if graph holds correct variables
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(a.J(i, j) , urd(r));
        }
    }

    r = utility::Xorshift(1234);

    // check if graph index is reversible (Jij = Jji)
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(a.J(j, i) , urd(r));
        }
    }
}

TEST(Graph, SparseGraphCheck){
    using namespace openjij::graph;
    using namespace openjij;

    std::size_t N = 500;
    Sparse<double> b(N, N-1);
    auto r = utility::Xorshift(1234);
    auto urd = std::uniform_real_distribution<>{-10, 10};
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b.J(i, j) = urd(r);
        }
    }

    r = utility::Xorshift(1234);

    // check if graph holds correct variables
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(b.J(i, j) , urd(r));
        }
    }

    r = utility::Xorshift(1234);

    // check if graph index is reversible (Jij = Jji)
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(b.J(j, i) , urd(r));
        }
    }

    //check adj_nodes
    for(std::size_t i=0; i<N; i++){
        std::size_t tot = 0;
        for(auto&& elem : b.adj_nodes(i)){
            tot += elem;
        }
        EXPECT_EQ(tot, N*(N-1)/2 - i);
    }
    EXPECT_EQ(b.get_num_edges(), N-1);

    Sparse<double> c(N, N);
    r = utility::Xorshift(1234);
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c.J(j, i) = urd(r);
        }
    }

    r = utility::Xorshift(1234);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(c.J(i, j) , urd(r));
        }
    }

    r = utility::Xorshift(1234);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(c.J(j, i) , urd(r));
        }
    }
    for(std::size_t i=0; i<N; i++){
        std::size_t tot = 0;
        for(auto&& elem : c.adj_nodes(i)){
            tot += elem;
        }
        EXPECT_EQ(tot, N*(N-1)/2);
    }
    EXPECT_EQ(c.get_num_edges(), N);
}

TEST(Graph, EnergyCheck){
    using namespace openjij::graph;
    std::size_t N = 500;

    Dense<double> b_d(N);
    Sparse<double> b(N, N-1);

    Spins spins(N, 1);
    Spins spins_neg(N, -1);
    auto random_engine = std::mt19937(1);
    Spins spins_r = b_d.gen_spin(random_engine);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b_d.J(i, j) = 1;
        }
    }

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b.J(i, j) = 1;
        }
    }

    EXPECT_EQ(b_d.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b_d.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b_d.calc_energy(spins_r), b.calc_energy(spins_r));

    Dense<double> c_d(N);
    Sparse<double> c(N, N);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c_d.J(i, j) = 1;
        }
    }

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c.J(i, j) = 1;
        }
    }

    EXPECT_EQ(c_d.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c_d.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c_d.calc_energy(spins_r), c.calc_energy(spins_r));
}

//ClassicalIsing tests

TEST(ClassicalIsing, GenerateTheSameEigenObject){
    using namespace openjij;
    graph::Dense<double> d(4);
    graph::Sparse<double> s(4);
    d.J(2,3) = s.J(2,3) = 4;
    d.J(1,0) = s.J(1,0) = -2;
    d.J(1,1) = s.J(1,1) = 5;
    d.J(2,2) = s.J(2,2) = 10;

    auto engine_for_spin = std::mt19937(1);
    auto cl_dense = system::make_classical_ising<true>(d.gen_spin(engine_for_spin), d);
    auto cl_sparse = system::make_classical_ising<true>(s.gen_spin(engine_for_spin), s);
    Eigen::MatrixXd m1 = cl_dense.interaction;
    //convert from sparse to dense
    Eigen::MatrixXd m2 = cl_sparse.interaction;
    EXPECT_EQ(m1, m2);
}

//TODO: macro?
//SingleSpinFlip tests

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Dense_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Sparse_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Dense_WithEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising<true>(spin, interaction); //Eigen implementation enabled
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Sparse_WithEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising<true>(spin, interaction); //Eigen implementation enabled
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Dense_NoEigenImpl) {
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
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Sparse_NoEigenImpl) {
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

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Dense_WithEigenImpl) {
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

    auto transverse_ising = system::make_transverse_ising<true>(init_trotter_spins, interaction, 1.0);

    auto transverse_ising2 = system::make_transverse_ising<true>(interaction.gen_spin(engine_for_spin), interaction, 1.0, 10);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Sparse_WithEigenImpl) {
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

    auto transverse_ising = system::make_transverse_ising<true>(init_trotter_spins, interaction, 1.0); //gamma = 1.0
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

//swendsen-wang test
TEST(SwendsenWang, FindTrueGroundState_CLassicalIsing_Dense_OneDimensionalIsing) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = [](){
        auto interaction = graph::Dense<double>(num_system_size);
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

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(openjij::graph::Spins({-1, -1, -1, -1, -1, +1, -1, +1}), result::get_solution(classical_ising));
}

TEST(SwendsenWang, FindTrueGroundState_ClassicalIsing_Dense_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_numder_engine = std::mt19937(1);

    //in general swendsen wang is not efficient in simulating frustrated systems. We need more annealing time.
    const auto schedule_list = openjij::utility::make_classical_schedule_list(0.1, 100.0, 2000, 2000);

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

//gpu test

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

    //generate classical dense system
    const auto interaction = generate_chimera_interaction<float>();
    auto engine_for_spin = std::mt19937(1253);
    std::size_t num_trotter_slices = 10;
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto chimera_quantum_gpu = system::make_chimera_transverse_gpu<1,1,1>(init_trotter_spins, interaction, 1.0); 
    auto& info = chimera_quantum_gpu.info;

    auto random_number_engine = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(info.rows*info.cols*info.trotters*info.chimera_unitsize, 12356);

    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::GPU>::run(chimera_quantum_gpu, random_number_engine, schedule_list);

    graph::Spins res = result::get_solution(chimera_quantum_gpu);
    
    EXPECT_EQ(get_true_chimera_groundstate(interaction), result::get_solution(chimera_quantum_gpu));
}

#endif

//utility test

TEST(Eigen, CopyFromVectorToEigenMatrix) {
    using namespace openjij;
    
    std::size_t N = 500;
    auto spins = graph::Spins(N);
    graph::Dense<double> a(N);
    auto r = utility::Xorshift(1234);
    auto uid = std::uniform_int_distribution<>{0, 1};

    spins = a.gen_spin(r);

    Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
    vec = utility::gen_vector_from_std_vector<double>(spins);

    for(std::size_t i=0; i<N; i++){
        EXPECT_EQ(vec(i), spins[i]);
    }

    EXPECT_EQ(vec(N), 1);
}

TEST(Eigen, CopyFromTrotterSpinToEigenMatrix) {
    using namespace openjij;
    
    std::size_t N = 500;
    std::size_t num_trot = 10;

    auto trotter_spins = system::TrotterSpins(num_trot);
    graph::Dense<double> a(N);
    auto r = utility::Xorshift(1234);

    for(auto& spins : trotter_spins){
        spins = a.gen_spin(r);
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat = utility::gen_matrix_from_trotter_spins<double>(trotter_spins);

    //initialize spin
    for(size_t j=0; j<num_trot; j++){
        for(size_t i=0; i<N; i++){
            EXPECT_EQ(mat(i,j), trotter_spins[j][i]); //ith spin in jth trotter slice
        }
    }

    //dummy spins
    for(size_t j=0; j<num_trot; j++){
        EXPECT_EQ(mat(N,j), 1);
    }
}

TEST(Eigen, CopyFromGraphToEigenMatrix) {
    using namespace openjij;

    std::size_t N = 500;
    graph::Dense<double> a(N);
    
    //generate dense matrix
    auto r = utility::Xorshift(1234);
    auto urd = std::uniform_real_distribution<>{-10, 10};
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            a.J(i, j)  = urd(r);
        }
    }
    
    //copy to Eigen Matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat = utility::gen_matrix_from_graph(a);

    //interaction
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(mat(i,j), a.J(i,j));
            EXPECT_EQ(mat(j,i), a.J(j,i));
        }
    }

    //local field
    for(std::size_t i=0; i<N; i++){
        EXPECT_EQ(mat(i,N), a.h(i));
        EXPECT_EQ(mat(N,i), a.h(i));
    }

    EXPECT_EQ(mat(N,N), 1);

    graph::Sparse<double> b(N);
    
    //generate sparse matrix
    r = utility::Xorshift(1234);
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            b.J(i, j)  = urd(r);
        }
    }
    
    //copy to Eigen SparseMatrix
    Eigen::SparseMatrix<double> mat_s(N+1, N+1);
    mat_s = utility::gen_matrix_from_graph(b);

    //interaction
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(mat_s.coeff(i,j), a.J(i,j));
            EXPECT_EQ(mat_s.coeff(j,i), a.J(j,i));
        }
    }

    //local field
    for(std::size_t i=0; i<N; i++){
        EXPECT_EQ(mat_s.coeff(i,N), a.h(i));
        EXPECT_EQ(mat_s.coeff(N,i), a.h(i));
    }

    EXPECT_EQ(mat_s.coeff(N,N), 1);
}

TEST(UnionFind, UniteSevenNodesToMakeThreeSets) {
    auto union_find = openjij::utility::UnionFind(7);

    union_find.unite_sets(0,1);
    union_find.unite_sets(1,4);
    union_find.unite_sets(3,5);
    union_find.unite_sets(5,6);

    auto expect = std::vector<decltype(union_find)::Node>{1,1,2,5,1,5,5};
    for (std::size_t node = 0; node < 7; ++node) {
        EXPECT_EQ(union_find.find_set(node), expect[node]);
    }
}

TEST(UnionFind, EachNodeIsInEachClusterByDefault) {
    auto union_find = openjij::utility::UnionFind(7);

    auto expect = std::vector<decltype(union_find)::Node>{0,1,2,3,4,5,6};
    for (std::size_t node = 0; node < 7; ++node) {
        EXPECT_EQ(union_find.find_set(node), expect[node]);
    }
}

TEST(UnionFind, ConnectingEachNodeAndAllAdjacentNodesResultsInOneSet) {
    auto union_find = openjij::utility::UnionFind(7);

    for (std::size_t node = 0; node < 6; ++node) {
        union_find.unite_sets(node, node+1);
    }

    auto expect = std::vector<decltype(union_find)::Node>{1,1,1,1,1,1,1};
    for (std::size_t node = 0; node < 7; ++node) {
        EXPECT_EQ(union_find.find_set(node), expect[node]);
    }
}

#ifdef USE_CUDA

TEST(GPUUtil, UniqueDevPtrTest){
    constexpr std::size_t SIZE = 10000;
    using namespace openjij;
    auto urd = std::uniform_real_distribution<float>{-10, 10};
    auto r = utility::Xorshift(1234);
    std::vector<float> input(SIZE);
    std::vector<float> output(SIZE);
    for(std::size_t i=0; i<SIZE; i++){
        input[i] = urd(r);
    }
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    HANDLE_ERROR_CUDA(cudaMemcpy(gpu_mem.get(), input.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR_CUDA(cudaMemcpy(output.data(), gpu_mem.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(input, output);

}

TEST(GPUUtil, UniqueHostPtrTest){
    constexpr std::size_t SIZE = 10000;
    using namespace openjij;
    auto urd = std::uniform_real_distribution<float>{-10, 10};
    auto r = utility::Xorshift(1234);
    auto input = utility::cuda::make_host_unique<float[]>(SIZE);
    auto output = utility::cuda::make_host_unique<float[]>(SIZE);
    for(std::size_t i=0; i<SIZE; i++){
        input[i] = urd(r);
    }
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    HANDLE_ERROR_CUDA(cudaMemcpy(gpu_mem.get(), input.get(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR_CUDA(cudaMemcpy(output.get(), gpu_mem.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    for(std::size_t i=0; i<SIZE; i++){
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(GPUUtil, CurandWrapperTest){
    constexpr std::size_t SIZE = 10000;
    using namespace openjij;
    auto wrap = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(SIZE, 1234);
    auto output = utility::cuda::make_host_unique<float[]>(SIZE);
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    wrap.generate_uniform(SIZE);
    HANDLE_ERROR_CUDA(cudaMemcpy(output.get(), wrap.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(0 <= output[0] && output[0] <= 1);
    for(std::size_t i=1; i<SIZE; i++){
        EXPECT_NE(output[i-1], output[i]);
        EXPECT_TRUE(0 <= output[i] && output[i] <= 1);
    }
}
TEST(GPUUtil, CuBLASWrapperTest){
    using namespace openjij;
    constexpr std::size_t M = 1000;
    constexpr std::size_t K = 205;
    constexpr std::size_t N = 6;

    //Note: matrix in cuBLAS in column major
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_A(M, K);

    host_A = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(M, K);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_B(K, N);

    host_B = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(K, N);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_C_answer(M, N);
    host_C_answer = host_A * host_B;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_C(M, N);

    //copy to gpu
    auto A = utility::cuda::make_dev_unique<float[]>(M*K);
    auto B = utility::cuda::make_dev_unique<float[]>(K*N);
    auto C = utility::cuda::make_dev_unique<float[]>(M*N);

    HANDLE_ERROR_CUDA(cudaMemcpy(A.get(), host_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR_CUDA(cudaMemcpy(B.get(), host_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

    auto cublas = utility::cuda::CuBLASWrapper();

    //matrix product
    cublas.matmul(M, K, N, A, B, C);

    HANDLE_ERROR_CUDA(cudaMemcpy(host_C.data(), C.get(), M*N*sizeof(float), cudaMemcpyDeviceToHost));
    for(std::size_t i=0; i<M; i++){
        for(std::size_t j=0; j<N; j++){
            EXPECT_NEAR(host_C(i,j), host_C_answer(i,j), 1e-5);
        }
    }

    //Iamax test
    const size_t SIZE = 10000;
    auto r = utility::Xorshift(12345678);
    auto urd = std::uniform_real_distribution<float>{-10, 10};
    std::vector<float> host_vec(SIZE);
    auto device_vec = utility::cuda::make_dev_unique<float[]>(SIZE);
    for(auto&& elem : host_vec){
        elem = urd(r);
    }

    HANDLE_ERROR_CUDA(cudaMemcpy(device_vec.get(), host_vec.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));

    //index
    int host_idx;
    int dev_idx;
    auto device_idx = utility::cuda::make_dev_unique<int[]>(1);

    //calc maxind (host)
    host_idx = std::distance(host_vec.begin(), std::max_element(host_vec.begin(), host_vec.end()));
    //calc maxind (device)
    cublas.max_val_index(SIZE, device_vec, device_idx);
    HANDLE_ERROR_CUDA(cudaMemcpy(&dev_idx, device_idx.get(), 1*sizeof(int), cudaMemcpyDeviceToHost));
    //NOTE: max_val will return 1-indexed value!!
    EXPECT_EQ(host_idx, dev_idx-1);
}

#endif
