#include "gradc/gradc.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>

using namespace gradc;
using namespace std::chrono;

void benchmark(bool use_cudnn, const char* name, int iterations, Tensor<float>& x, Tensor<float>& gamma, Tensor<float>& beta) {
    for (int i = 0; i < 3; i++) {
        Tensor<float> out = layernorm(x, gamma, beta, {2}, 1e-5f, use_cudnn);
        out.realize();
        Tensor<float> loss = out.sum({0, 1, 2}, false);
        loss.realize();
        loss.backward();
        
        x.zero_grad(); 
        gamma.zero_grad(); 
        beta.zero_grad();
    }
    
    // Catch asynchronous CUDA kernel errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error during warmup: ") + cudaGetErrorString(err));
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Tensor<float> out = layernorm(x, gamma, beta, {2}, 1e-5f, use_cudnn);
        out.realize();
        
        Tensor<float> loss = out.sum({0, 1, 2}, false);
        loss.realize();
        
        loss.backward(); 
        
        x.zero_grad();
        gamma.zero_grad(); 
        beta.zero_grad();
    }
    
    err = cudaDeviceSynchronize(); 
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error during benchmark: ") + cudaGetErrorString(err));
    }
    
    auto end = high_resolution_clock::now();
    float avg_ms = duration<float, std::milli>(end - start).count() / iterations;
    std::cout << name << " LayerNorm (Fwd + Bwd): " << avg_ms << " ms/iter\n";
}

int main() {
    try {
        Device cuda(DeviceType::CUDA, 0);
        
        int64_t B = 32, T_seq = 1024, C = 4096;
        std::cout << "Benchmarking LayerNorm on " << cuda << "\n";
        std::cout << "Shape: [" << B << ", " << T_seq << ", " << C << "]\n\n";

        Tensor<float> x = Tensor<float>::normal({B, T_seq, C}, 0.0f, 1.0f, cuda);
        x.set_requires_grad(true);

        Tensor<float> gamma = Tensor<float>::ones({C}, cuda);
        gamma.set_requires_grad(true);

        Tensor<float> beta = Tensor<float>::zeros({C}, cuda);
        beta.set_requires_grad(true);

        int iterations = 20;
        
        std::cout << "Warming up and running " << iterations << " iterations...\n\n";
        benchmark(true, "cuDNN", iterations, x, gamma, beta);
        benchmark(false, "Naive", iterations, x, gamma, beta);
        
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR]: " << e.what() << "\n";
        std::cin.get();
        return -1;
    } catch (...) {
        std::cerr << "\n[FATAL ERROR]: Unknown exception occurred!\n";
        std::cin.get();
        return -1;
    }
    std::cin.get();
    return 0;
}