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

TEST(Eigen, CopyFromVectorToEigenMatrix) {
    using namespace openjij;
    
    std::size_t N = 500;
    auto spins = graph::Spins(N);
    graph::Dense<double> a(N);
    auto r = utility::Xorshift(1234);
    //auto uid = std::uniform_int_distribution<>{0, 1};

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



}
}
