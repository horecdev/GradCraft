#include "gradc/gradc.hpp" // IWYU pragma: keep
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <functional>
#include <string>

using namespace gradc;

void run_benchmark(const std::string& name, const std::function<void()>& fn, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    std::cout << name << ": " << ms << " ms\n";
}

int main() {
    try {
        Device cuda(DeviceType::CUDA, 0);

        int64_t B = 8;
        int64_t T_seq = 1024;
        int64_t C = 4096;
        int64_t vocab = 32000;
        int64_t heads = 32;

        auto x_rms = Tensor<float>::zeros({B, T_seq, C}, cuda).set_requires_grad(true);
        auto g_rms = Tensor<float>::ones({C}, cuda).set_requires_grad(true);
        
        run_benchmark("RMSNorm Naive (Fwd+Bwd)", [&](){ 
            auto out = rmsnorm(x_rms, g_rms, {2}, 1e-5f, false);
            out.realize(); 
            out.backward(); 
        });
        run_benchmark("RMSNorm Fast (Fwd+Bwd)", [&](){ 
            auto out = rmsnorm(x_rms, g_rms, {2}, 1e-5f, true);
            out.realize(); 
            out.backward(); 
        });

        auto logits = Tensor<float>::zeros({B * T_seq, vocab}, cuda).set_requires_grad(true);
        auto targets_naive = Tensor<float>::zeros({B * T_seq, vocab}, cuda); // Targets never need gradients
        auto targets_fast = Tensor<int64_t>::zeros({B * T_seq}, cuda); 
        
        run_benchmark("SCEL Naive (Fwd+Bwd)", [&](){ 
            auto out = softmax_crossentropy_naive(logits, targets_naive, 1, 1e-5f);
            out.realize(); 
            out.backward(); 
        });
        run_benchmark("SCEL Fast (Fwd+Bwd)", [&](){ 
            auto out = softmax_crossentropy_fast(logits, targets_fast, 1e-5f);
            out.realize(); 
            out.backward(); 
        });

        Parameter<float> param(Tensor<float>::zeros({C, C}, cuda));
        param.tensor().accumulate_grad(Tensor<float>::zeros({C, C}, cuda));
        std::unordered_map<std::string, Parameter<float>*> params = {{"weight", &param}};
        AdamW<float> optim(params, 1e-3f);
        
        // Optimizers don't have a backward pass, leaving this as is
        run_benchmark("AdamW Step Naive", [&](){ 
            optim.set_cuda_fast(false); 
            optim.step(); 
        });
        run_benchmark("AdamW Step Fast", [&](){ 
            optim.set_cuda_fast(true); 
            optim.step(); 
        });

        auto w1_swi = Tensor<float>::zeros({B, T_seq, C}, cuda).set_requires_grad(true);
        auto w2_swi = Tensor<float>::zeros({B, T_seq, C}, cuda).set_requires_grad(true);
        
        run_benchmark("SwiGLU Naive (Fwd+Bwd)", [&](){ 
            auto out = (w1_swi.silu() * w2_swi);
            out.realize(); 
            out.backward(); 
        });
        run_benchmark("SwiGLU Fast (Fwd+Bwd)", [&](){ 
            auto out = swiglu_fast(w1_swi, w2_swi);
            out.realize(); 
            out.backward(); 
        });

        auto scores = Tensor<float>::zeros({B, heads, T_seq, T_seq}, cuda).set_requires_grad(true);
        auto mask = Tensor<float>::upper_triangular(T_seq, -1e9f, cuda);
        float scale = 1.0f;
        
        run_benchmark("Causal Softmax Naive (Fwd+Bwd)", [&](){ 
            auto out = ((scores * scale) + mask).softmax(3);
            out.realize(); 
            out.backward(); 
        });
        run_benchmark("Causal Softmax Fast (Fwd+Bwd)", [&](){ 
            auto out = causal_softmax(scores, scale);
            out.realize(); 
            out.backward(); 
        });

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}