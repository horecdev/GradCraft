#include "gradc/gradc.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>

using namespace gradc;

template <typename Func>
void run_benchmark(const std::string& name, int warmup, int iters, Func&& f) {
    for (int i = 0; i < warmup; ++i) { f(); }
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { f(); }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    std::cout << std::left << std::setw(40) << name << " | " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
}

int main() {
    try {
        Device gpu(DeviceType::CPU, 0);
        std::cout << "Initializing Benchmark Setup (B=4, T=1024, V=32768, C=768, H=12)...\n" << std::endl;

        int64_t B = 1;
        int64_t T_seq = 128;
        int64_t C = 768;
        int64_t V = 32768;
        int64_t H = 12;
        int64_t head_dim = C / H;

        int warmup = 10;
        int iters = 50;

        // 1. RMSNorm Benchmark
        Tensor<float> rms_parent = Tensor<float>::normal({B, T_seq, C}, 0.0f, 1.0f, gpu);
        rms_parent.set_requires_grad(true);
        rms_parent.realize();

        Tensor<float> rms_gamma = Tensor<float>::ones({C}, gpu);
        rms_gamma.set_requires_grad(true);
        rms_gamma.realize();

        std::cout << "--- RMSNorm ---" << std::endl;
        run_benchmark("Naive RMSNorm (Fwd + Bwd)", warmup, iters, [&]() {
            auto out = rmsnorm(rms_parent, rms_gamma, {2}, 1e-5f, false);
            out.realize();
            out.backward(false);
            rms_parent.zero_grad();
            rms_gamma.zero_grad();
        });

        run_benchmark("Fast RMSNorm (Fwd + Bwd)", warmup, iters, [&]() {
            auto out = rmsnorm(rms_parent, rms_gamma, {2}, 1e-5f, true);
            out.realize();
            out.backward(false);
            rms_parent.zero_grad();
            rms_gamma.zero_grad();
        });
        std::cout << std::endl;

        // 2. Scaled Dot-Product Attention (SDPA) Benchmark
        Tensor<float> q = Tensor<float>::normal({B, H, T_seq, head_dim}, 0.0f, 1.0f, gpu);
        Tensor<float> k = Tensor<float>::normal({B, H, T_seq, head_dim}, 0.0f, 1.0f, gpu);
        Tensor<float> v = Tensor<float>::normal({B, H, T_seq, head_dim}, 0.0f, 1.0f, gpu);
        q.set_requires_grad(true); k.set_requires_grad(true); v.set_requires_grad(true);
        q.realize(); k.realize(); v.realize();

        Tensor<float> custom_mask = Tensor<float>::upper_triangular(T_seq, -1e9f, gpu);
        custom_mask.realize();

        std::cout << "--- SDPA (Causal Attention) ---" << std::endl;
        run_benchmark("Naive SDPA (Fwd + Bwd)", warmup, iters, [&]() {
            auto out = sdpa(q, k, v, true, std::optional<Tensor<float>>(custom_mask), std::optional<float>(std::nullopt), false);
            out.realize();
            out.backward(false);
            q.zero_grad(); k.zero_grad(); v.zero_grad();
        });

        run_benchmark("Fast Causal SDPA (Fwd + Bwd)", warmup, iters, [&]() {
            auto out = sdpa(q, k, v, true, std::optional<Tensor<float>>(std::nullopt), std::optional<float>(std::nullopt), true);
            out.realize();
            out.backward(false);
            q.zero_grad(); k.zero_grad(); v.zero_grad();
        });
        std::cout << std::endl;

        // 3. Softmax Cross-Entropy Benchmark
        Tensor<float> logits = Tensor<float>::normal({B, T_seq, V}, 0.0f, 1.0f, gpu);
        logits.set_requires_grad(true);
        logits.realize();

        Tensor<int64_t> targets = Tensor<int64_t>::zeros({B, T_seq}, gpu);
        targets.realize();

        Tensor<float> flat_logits = logits.reshape({B * T_seq, V});
        flat_logits.realize();
        Tensor<int64_t> flat_targets_idx = targets.reshape({B * T_seq});
        flat_targets_idx.realize();
        Tensor<float> flat_targets = one_hot_encode<float>(flat_targets_idx, V);
        flat_targets.realize();

        std::cout << "--- Softmax Cross Entropy ---" << std::endl;
        run_benchmark("Naive SCE Loss (Fwd + Bwd)", warmup, iters, [&]() {
            auto loss = softmax_crossentropy_naive(flat_logits, flat_targets, 1, 1e-5f);
            loss.realize();
            loss.backward(false);
            logits.zero_grad(); 
        });

        run_benchmark("Fast Sparse SCE Loss (Fwd + Bwd)", warmup, iters, [&]() {
            auto loss = softmax_crossentropy_fast(logits, targets, 1e-5f);
            loss.realize();
            loss.backward(false);
            logits.zero_grad();
        });
        std::cout << std::endl;

        std::cout << "Benchmarking complete." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Benchmark Error: " << e.what() << std::endl;
        return 1;
    }
}