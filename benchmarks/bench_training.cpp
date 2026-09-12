#include "gradc/gradc.hpp" // IWYU pragma: keep
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace gradc;

void run_benchmark(int64_t B, int64_t T, int warmup, int iters) {
    Device dev(DeviceType::CUDA, 0);
    
    std::cout << "Config: B=" << B << ", T=" << T << ", Model=174M\n";

    int64_t num_layers = 20;

    float base_std = 0.02f;
    float residual_std = 0.02f / std::sqrt(2.0f * num_layers);

    NormalInit<float> base_init(0.0f, base_std);
    NormalInit<float> residual_init(0.0f, residual_std);
    GPT<float> model(32768, T, 768, 12, num_layers, base_init, residual_init, 1e-5f);
    model.to(dev);

    Tensor<int64_t> X = Tensor<int64_t>::zeros({B, T}, dev);
    Tensor<int64_t> Y = Tensor<int64_t>::zeros({B, T}, dev);
    X.realize();
    Y.realize();

    std::cout << "Number of params: " << model.num_params() << std::endl;;

    for (int i = 0; i < warmup; ++i) {
        Tensor<float> logits = model.forward(X);
        Tensor<float> loss = softmax_crossentropy_fast<float>(logits, Y, 1e-5f);
        loss.realize();
        loss.backward(false);
        model.zero_grad();
    }
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; ++i) {
        Tensor<float> logits = model.forward(X);
        Tensor<float> loss = softmax_crossentropy_fast<float>(logits, Y, 1e-5f);
        loss.realize();
        loss.backward(false);
        model.zero_grad();
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iters;
    double total_tokens = B * T * iters;
    double tok_sec = total_tokens / (total_ms / 1000.0);

    std::cout << "Avg Time per Step: " << std::fixed << std::setprecision(2) << avg_ms << " ms\n";
    std::cout << "Throughput:        " << static_cast<int64_t>(tok_sec) << " tok/s\n";
    CUDAMemPool::get().log_hwm();
}

int main() {
    try {
        std::cout << "Benchmarking training run\n";
        run_benchmark(1, 1024, 10, 50);
        //run_benchmark(2, 1024, 10, 50);
        //run_benchmark(4, 1024, 10, 50);
        //run_benchmark(5, 1024, 10, 50);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}