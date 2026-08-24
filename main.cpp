#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "gradc/gradc.hpp"

using namespace gradc;

int main() {
    Device cuda_dev = Device(DeviceType::CUDA, 0);
    cudaSetDevice(cuda_dev.index);

    Tensor<float> A({8192, 8192}, 1.5f, cuda_dev);
    Tensor<float> B({8192, 8192}, 2.0f, cuda_dev);
    Tensor<float> C({8192, 8192}, 0.5f, cuda_dev);
    
    A.set_requires_grad(true);
    B.set_requires_grad(true);
    C.set_requires_grad(true);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    auto X1 = A + B;
    auto X2 = X1 * C;
    auto X3 = X2 + A;
    auto X4 = X3 * B;
    auto X5 = X4 + C;
    auto X6 = X5 * X1;
    auto X7 = X6 + X2;
    auto X8 = X7 * X3;

    auto loss = X8.sum({0, 1}, false);
    
    loss.realize();
    cudaDeviceSynchronize();
    
    auto mid = std::chrono::high_resolution_clock::now();

    loss.accumulate_grad(Tensor<float>(1.0f, cuda_dev), false);
    loss.backward();
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> fwd_time = mid - start;
    std::chrono::duration<double, std::milli> bwd_time = end - mid;
    std::chrono::duration<double, std::milli> total_time = end - start;

    std::cout << "FWD: " << fwd_time.count() << " ms\n";
    std::cout << "BWD: " << bwd_time.count() << " ms\n";
    std::cout << "TOT: " << total_time.count() << " ms\n";

    return 0;
}