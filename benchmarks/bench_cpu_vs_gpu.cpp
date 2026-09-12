#include "gradc/gradc.hpp" // IWYU pragma: keep
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace gradc;

void run_device_benchmark(DeviceType type, const std::string& dev_name) {
    int64_t B = 4, T = 128;
    int64_t V = 32768, C = 768, H = 12, L = 20;
    
    Device dev(type, 0);
    std::cout << "Initializing Model on " << dev_name << " (B=" << B << ", T=" << T << ")" << std::endl;

    float base_std = 0.02f;
    float residual_std = 0.02f / std::sqrt(2.0f * L);

    NormalInit<float> base_init(0.0f, base_std);
    NormalInit<float> residual_init(0.0f, residual_std);

    GPT<float> model(V, T, C, H, L, base_init, residual_init, 1e-5f);
    model.to(dev);

    // dummy inputs
    Tensor<int64_t> X = Tensor<int64_t>::zeros({B, T}, dev);
    Tensor<int64_t> Y = Tensor<int64_t>::zeros({B, T}, dev);

    std::cout << "Starting benchmark.\n";

    for (int step = 0; step < 10; ++step) {
        auto start = std::chrono::high_resolution_clock::now();

        Tensor<float> logits = model.forward(X);

        Tensor<float> loss;
        if (type == DeviceType::CUDA) {
            loss = softmax_crossentropy_fast<float>(logits, Y, 1e-5f);
        } else {
            Tensor<float> flat_logits = logits.reshape({B * T, V});
            Tensor<int64_t> flat_targets_idx = Y.reshape({B * T});
            Tensor<float> flat_targets = one_hot_encode<float>(flat_targets_idx, V);
            
            loss = softmax_crossentropy_naive(flat_logits, flat_targets, 1, 1e-5f); 
        }
        loss.realize();
        float loss_val = loss.item();

        model.zero_grad();
        loss.backward(false);

        if (type == DeviceType::CUDA) {
            cudaDeviceSynchronize(); 
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double tok_sec = (B * T) / (ms / 1000.0);

        if (step < 2) {
            std::cout << "Step " << step << " (Warmup) | Loss: " << std::fixed << std::setprecision(4) << loss_val << std::endl;
        } else {
            std::cout << "Step " << step << " | Time: " << std::fixed << std::setprecision(2) << ms << " ms | Speed: " << static_cast<int64_t>(tok_sec) << " tok/s" << std::endl;
        }
    }
    std::cout << std::endl;
}

int main() {
    try {
        run_device_benchmark(DeviceType::CPU, "CPU");

        //run_device_benchmark(DeviceType::CUDA, "RTX 3070 Ti");

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}