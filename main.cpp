#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "gradc/gradc.hpp"

using namespace gradc;

// ============================================================================
// TYPE ALIAS: Change to 'float' or 'double' here to switch precision instantly
// ============================================================================
using Scalar = double;

// CUDA error-checking macro that forces synchronous execution to catch async errors
#define CHECK_CUDA() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "\n[CUDA ERROR]: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n\n"; \
            std::abort(); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "\n[CUDA SYNC ERROR]: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n\n"; \
            std::abort(); \
        } \
    } while(0)

int main() {
    try {
        Device dev(DeviceType::CUDA, 0);
        PrintOptions opts; 
        opts.show_metadata = false;
        
        std::cout << std::setprecision(15) << std::fixed;

        // --- SETUP PARAMETERS & LEAF TENSORS ---
        Tensor<Scalar> embed_w = (Tensor<Scalar>::arange(1.0, 41.0, 1.0, dev) / static_cast<Scalar>(100.0)).reshape({10, 4});
        embed_w.realize(); embed_w.make_leaf().set_requires_grad(true);

        Tensor<Scalar> X = (Tensor<Scalar>::arange(1.0, 25.0, 1.0, dev) / static_cast<Scalar>(10.0)).reshape({2, 4, 3});
        X.realize(); X.make_leaf().set_requires_grad(true);

        Tensor<Scalar> W = (Tensor<Scalar>::arange(1.0, 41.0, 1.0, dev) / static_cast<Scalar>(10.0)).reshape({2, 4, 5});
        W.realize(); W.make_leaf().set_requires_grad(true);

        Tensor<Scalar> gamma = Tensor<Scalar>::ones({10}, dev);
        gamma.realize(); gamma.make_leaf().set_requires_grad(true);

        Tensor<Scalar> beta = Tensor<Scalar>::zeros({10}, dev);
        beta.realize(); beta.make_leaf().set_requires_grad(true);

        Tensor<Scalar> W_lin = (Tensor<Scalar>::arange(1.0, 61.0, 1.0, dev) / static_cast<Scalar>(10.0)).reshape({10, 6});
        W_lin.realize(); W_lin.make_leaf().set_requires_grad(true);

        Tensor<int64_t> indices = Tensor<int64_t>::zeros({2, 3}, dev);
        indices.set_data({0, 5, 2, 9, 1, 8});
        indices.realize(); indices.make_leaf();

        Tensor<Scalar> targets_ce = (Tensor<Scalar>::arange(1.0, 73.0, 1.0, dev) / static_cast<Scalar>(73.0)).reshape({12, 6}).softmax(1);
        targets_ce.realize(); targets_ce.make_leaf();

        Tensor<Scalar> targets_mse = (Tensor<Scalar>::arange(72.0, 0.0, -1.0, dev) / static_cast<Scalar>(10.0)).reshape({12, 6});
        targets_mse.realize(); targets_mse.make_leaf();

        CHECK_CUDA(); // Check initialization kernels

        // --- FORWARD PASS ---
        Tensor<Scalar> emb = embed(indices, embed_w);
        Tensor<Scalar> Y = X.transpose(1, 2);
        Tensor<Scalar> Z = emb + Y;

        Tensor<Scalar> Z1 = Z.square();
        Tensor<Scalar> Z2 = Z1.sqrt();
        Tensor<Scalar> Z3 = Z2.cos();
        Tensor<Scalar> Z4 = Z3.sin();
        Tensor<Scalar> Z5 = Z4.exp();
        Tensor<Scalar> Z6 = Z5.log();
        Tensor<Scalar> Z7 = Z6.silu();
        Tensor<Scalar> Z8 = Z7.gelu();
        Tensor<Scalar> Z9 = Z8.tanh();
        Tensor<Scalar> Z10 = Z9.sigmoid();

        Tensor<Scalar> V = bmm(Z10, W);
        Tensor<Scalar> V_un = V.unsqueeze(1);
        Tensor<Scalar> attn = scaled_dot_product_attention_naive(V_un, V_un, V_un);

        Tensor<Scalar> attn_sq = attn.squeeze(1);
        Tensor<Scalar> attn_perm = attn_sq.permute({2, 0, 1});
        Tensor<Scalar> attn_trans = attn_perm.transpose(0, 2);

        Tensor<Scalar> stacked = stack<Scalar>({attn_trans, attn_trans}, 0);
        Tensor<Scalar> concat_res = concat<Scalar>({stacked, stacked}, 3);

        Tensor<Scalar> mean_val = concat_res.mean({0, 2}, true);
        Tensor<Scalar> max_val = concat_res.max({1}, true);

        Tensor<Scalar> b_add = concat_res + mean_val;
        Tensor<Scalar> b_sub = b_add - max_val;
        Tensor<Scalar> b_mul = b_sub * static_cast<Scalar>(0.5);
        Tensor<Scalar> b_div = b_mul / static_cast<Scalar>(2.0);

        Tensor<Scalar> ln = layernorm(b_div, gamma, beta, {3}, static_cast<Scalar>(1e-5));
        Tensor<Scalar> ln_flat = ln.reshape({12, 10});

        Tensor<Scalar> logits = matmul(ln_flat, W_lin);

        Tensor<Scalar> loss_ce = softmax_crossentropy(logits, targets_ce, 1, static_cast<Scalar>(1e-5));
        Tensor<Scalar> loss_mse = mse_loss(logits, targets_mse);

        Tensor<Scalar> total_loss = loss_ce + loss_mse;

        // --- REALIZE FORWARD PASS ---
        total_loss.realize();
        CHECK_CUDA(); // Catches any CUDA kernel crash in the Forward Pass

        // --- BACKWARD PASS ---
        total_loss.accumulate_grad(Tensor<Scalar>::ones(total_loss.shape(), dev));
        CHECK_CUDA(); // Catches memory allocation or copy errors during gradient init

        total_loss.backward(false);
        CHECK_CUDA(); // Catches any CUDA kernel crash in the Backward Pass

        // --- RESULTS ---
        std::cout << "Loss:\n" << total_loss.item() << "\n";
        std::cout << "\n=== DEEP GRAPH GRADIENTS ===\n";
        std::cout << "\nembed_w.grad:\n"; print_tensor(std::cout, embed_w.grad().value(), opts);
        std::cout << "\nX.grad:\n";       print_tensor(std::cout, X.grad().value(), opts);
        std::cout << "\nW.grad:\n";       print_tensor(std::cout, W.grad().value(), opts);

    } catch (const std::exception& e) {
        std::cerr << "\n[GRAD-C FATAL ERROR]: " << e.what() << "\n\n";
    }

    return 0;
}