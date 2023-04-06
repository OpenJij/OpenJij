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

    // generate interaction matrix from dense graph
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat = b_d.get_interactions().sparseView();
    // and generate CSRSparse
    CSRSparse<double> b_csrs(mat.template triangularView<Eigen::Upper>());

    EXPECT_EQ(b_d.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b_d.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b_csrs.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b_csrs.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b_d.calc_energy(spins_r), b.calc_energy(spins_r));
    EXPECT_EQ(b_d.calc_energy(spins_r), b_csrs.calc_energy(spins_r));

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
    // generate interaction matrix from dense graph
    mat = c_d.get_interactions().sparseView();
    // and generate CSRSparse
    CSRSparse<double> c_csrs(mat.template triangularView<Eigen::Upper>());

    EXPECT_EQ(c_d.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c_d.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c_csrs.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c_csrs.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c_d.calc_energy(spins_r), c.calc_energy(spins_r));
    EXPECT_EQ(c_d.calc_energy(spins_r), c_csrs.calc_energy(spins_r));
}

//json tests
TEST(Graph, JSONTest){
    using namespace cimod;
    using namespace openjij;

    Linear<uint32_t, double> linear{ {0, 1.0}, {11, 3.0}};
    Quadratic<uint32_t, double> quadratic
    {
        {std::make_pair(0, 4), 12.0}, {std::make_pair(4, 12), 13.0}, {std::make_pair(6, 14), 14.0},
            {std::make_pair(3, 4), 23.0}, {std::make_pair(11, 12), 24.0},
            {std::make_pair(5, 13), 34.0}
    };
    double offset = 0.0;
    Vartype vartype = Vartype::SPIN;
    BinaryQuadraticModel<uint32_t, double, cimod::Sparse> bqm_k4(linear, quadratic, offset, vartype);
    auto s = graph::Chimera<double>(bqm_k4.to_serializable(), 1, 2);
    EXPECT_NEAR(s.J(0,1,3,graph::ChimeraDir::IN_0or4), 24, 1e-5);
    EXPECT_NEAR(s.J(0,0,4,graph::ChimeraDir::PLUS_C), 13, 1e-5);
    EXPECT_NEAR(s.J(0,1,6,graph::ChimeraDir::MINUS_C), 14, 1e-5);
    EXPECT_NEAR(s.J(0,0,4,graph::ChimeraDir::IN_3or7), 23, 1e-5);
    EXPECT_NEAR(s.J(0,0,3,graph::ChimeraDir::IN_0or4), 23, 1e-5);
    EXPECT_NEAR(s.J(0,1,5,graph::ChimeraDir::MINUS_C), 34, 1e-5);
    EXPECT_NEAR(s.h(0,1,3), 3, 1e-5);
}



} // namespace test
} // namespace openjij
