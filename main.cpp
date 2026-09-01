#include "gradc/gradc.hpp"
#include <iostream>
#include <iomanip>

using namespace gradc;

void test_layernorm(Device cuda) {
    std::cout << "========================================\n";
    std::cout << "1. LAYERNORM CUDNN VERIFICATION\n";
    std::cout << "========================================\n";

    // Shape: [B=2, T=4, C=8]
    Tensor<float> x_raw = (Tensor<float>::arange(1.0f, 65.0f, 1.0f, cuda) * 0.1f).reshape({2, 4, 8});
    x_raw.realize();
    Tensor<float> x = x_raw.detach();
    x.set_requires_grad(true);

    Tensor<float> gamma = Tensor<float>::ones({8}, cuda);
    gamma.set_requires_grad(true);

    Tensor<float> beta = Tensor<float>::zeros({8}, cuda);
    beta.set_requires_grad(true);

    // Forward LayerNorm
    Tensor<float> out = layernorm(x, gamma, beta, {2}, 1e-5f, /*use_cudnn=*/true);
    out.realize();

    // Compute scalar loss and backward
    Tensor<float> loss = out.sum({0, 1, 2}, false);
    loss.realize();
    loss.backward();

    std::cout << "Forward Output Loss Sum: " << loss.item() << "\n";
    std::cout << "x.grad()[0, 0, 0..3]:\n";
    print_tensor(std::cout, x.grad().value()[0][0][Slice(0, 4)]);
    std::cout << "gamma.grad()[0..3]:\n";
    print_tensor(std::cout, gamma.grad().value()[Slice(0, 4)]);
    std::cout << "beta.grad()[0..3]:\n";
    print_tensor(std::cout, beta.grad().value()[Slice(0, 4)]);
    std::cout << "\n";
}

void test_sdpa(Device cuda) {
    std::cout << "========================================\n";
    std::cout << "2. SDPA CUDNN VERIFICATION (Causal=True)\n";
    std::cout << "========================================\n";

    // Shape: [B=2, H=2, T=4, D=8]
    Tensor<float> q_raw = (Tensor<float>::arange(1.0f, 65.0f, 1.0f, cuda) * 0.05f).reshape({2, 2, 4, 8});
    Tensor<float> k_raw = (Tensor<float>::arange(1.0f, 65.0f, 1.0f, cuda) * 0.03f).reshape({2, 2, 4, 8});
    Tensor<float> v_raw = (Tensor<float>::arange(1.0f, 65.0f, 1.0f, cuda) * 0.02f).reshape({2, 2, 4, 8});

    q_raw.realize(); k_raw.realize(); v_raw.realize();

    Tensor<float> q = q_raw.detach(); q.set_requires_grad(true);
    Tensor<float> k = k_raw.detach(); k.set_requires_grad(true);
    Tensor<float> v = v_raw.detach(); v.set_requires_grad(true);

    // Forward Causal SDPA
    Tensor<float> out = sdpa_cudnn(q, k, v, /*is_causal=*/true);
    out.realize();

    // Compute scalar loss and backward
    Tensor<float> loss = out.sum({0, 1, 2, 3}, false);
    loss.realize();
    loss.backward();

    std::cout << "Forward Output Loss Sum: " << loss.item() << "\n";
    std::cout << "q.grad()[0, 0, 0, 0..3]:\n";
    print_tensor(std::cout, q.grad().value()[0][0][0][Slice(0, 4)]);
    std::cout << "k.grad()[0, 0, 0, 0..3]:\n";
    print_tensor(std::cout, k.grad().value()[0][0][0][Slice(0, 4)]);
    std::cout << "v.grad()[0, 0, 0, 0..3]:\n";
    print_tensor(std::cout, v.grad().value()[0][0][0][Slice(0, 4)]);
    std::cout << "\n";
}

int main() {
    try {
        Device cuda(DeviceType::CUDA, 0);
        std::cout << std::fixed << std::setprecision(6);

        test_layernorm(cuda);
        test_sdpa(cuda);

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR]: " << e.what() << "\n";
        return -1;
    }
    return 0;
}