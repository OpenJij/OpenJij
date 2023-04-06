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
    auto wrap = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(1234);
    auto output = utility::cuda::make_host_unique<float[]>(SIZE);
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    wrap.generate_uniform(SIZE, gpu_mem);
    HANDLE_ERROR_CUDA(cudaMemcpy(output.get(), gpu_mem.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

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
    auto r = utility::Xorshift(23456678);
    auto urd = std::uniform_real_distribution<float>{0, 10};
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
    cublas.absmax_val_index(SIZE, device_vec, device_idx);
    HANDLE_ERROR_CUDA(cudaMemcpy(&dev_idx, device_idx.get(), 1*sizeof(int), cudaMemcpyDeviceToHost));
    //NOTE: max_val will return 1-indexed value!!
    EXPECT_EQ(host_idx, dev_idx-1);

    //dot product test
    std::vector<float> host_vec2(SIZE);
    auto device_vec2 = utility::cuda::make_dev_unique<float[]>(SIZE);
    urd = std::uniform_real_distribution<float>{0, 1};
    for(auto&& elem : host_vec2){
        elem = urd(r);
    }
    HANDLE_ERROR_CUDA(cudaMemcpy(device_vec2.get(), host_vec2.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    float dotprod = 0;
    float dev_dotprod;
    auto device_dotprod = utility::cuda::make_dev_unique<float[]>(1);
    for(std::size_t i=0; i<SIZE; i++){
        dotprod += host_vec[i]*host_vec2[i];
    }
    cublas.dot(SIZE, device_vec, device_vec2, device_dotprod);
    HANDLE_ERROR_CUDA(cudaMemcpy(&dev_dotprod, device_dotprod.get(), 1*sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_NEAR(dev_dotprod/1000.0, dotprod/1000.0, 1e-4);
}

#endif



}
}
