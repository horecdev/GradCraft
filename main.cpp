#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "gradc/gradc.hpp"

using namespace gradc;

int main() {
    Device cuda_dev = Device(DeviceType::CUDA, 0);
    cudaSetDevice(cuda_dev.index);

    int64_t B = 4;
    int64_t T_seq = 2048;
    int64_t C = 1024;
    int64_t heads = 16;
    int warmup_iters = 5;
    int bench_iters = 20;

    MultiHeadAttention<float> mha(C, heads, true, T_seq, NormalInit<float>(0.0f, 0.02f), ZerosInit<float>());
    mha.to(cuda_dev);

    Tensor<float> x = Tensor<float>::normal({B, T_seq, C}, 0.0f, 0.02f, cuda_dev);
    x.set_requires_grad(true);

    Tensor<float> target = Tensor<float>::normal({B, T_seq, C}, 0.0f, 0.02f, cuda_dev);

    for (int i = 0; i < warmup_iters; ++i) {
        auto out = mha.forward(x);
        std::cout << "Warmup iter " << i << std::endl;
        auto loss = mse_loss(out, target);
        loss.realize();
        loss.accumulate_grad(Tensor<float>(1.0f, cuda_dev), false);
        loss.backward();
        
        mha.zero_grad();
        x.zero_grad();
        cudaDeviceSynchronize();
    }

    double total_fwd = 0.0;
    double total_bwd = 0.0;

    for (int i = 0; i < bench_iters; ++i) {
        std::cout << "Benchmark iter " << i << std::endl;
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        auto out = mha.forward(x);
        auto loss = mse_loss(out, target);
        loss.realize();

        cudaDeviceSynchronize();
        auto mid = std::chrono::high_resolution_clock::now();

        loss.accumulate_grad(Tensor<float>(1.0f, cuda_dev), false);
        loss.backward();

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_fwd += std::chrono::duration<double, std::milli>(mid - start).count();
        total_bwd += std::chrono::duration<double, std::milli>(end - mid).count();

        mha.zero_grad();
        x.zero_grad();
    }

    std::cout << "GradC MHA Avg FWD: " << (total_fwd / bench_iters) << " ms\n";
    std::cout << "GradC MHA Avg BWD: " << (total_bwd / bench_iters) << " ms\n";
    std::cout << "GradC MHA Avg TOT: " << ((total_fwd + total_bwd) / bench_iters) << " ms\n";

    return 0;
}