#include <iostream>
#include <vector>
#include "gradc/gradc.hpp"

using namespace gradc;

int main() {
    Device cpu_dev = Device(DeviceType::CUDA, 0);

    // ==========================================
    // TEST 1: STANDARD 2D MATMUL
    // ==========================================
    std::cout << "==========================================\n";
    std::cout << "       TEST 1: 2D MATMUL (M=2, K=3, N=2)  \n";
    std::cout << "==========================================\n";

    Tensor<float> A({2, 3}, 0.0f, cpu_dev);
    
    A.set_data({1.0f, 2.0f, 3.0f, 
                4.0f, 5.0f, 6.0f});
    std::cout << "before here" << std::endl;
    A.set_requires_grad(true);
    

    Tensor<float> B({3, 2}, 0.0f, cpu_dev);
    B.set_data({ 7.0f,  8.0f, 
                 9.0f, 10.0f, 
                11.0f, 12.0f});
    B.set_requires_grad(true);
    
    auto Y = matmul(A, B); 
    auto loss1 = Y.sum({0, 1}, false); // Sum all elements to get a scalar loss
    
    loss1.realize();

    std::cout << "\n--- Forward Pass (2D Matmul) ---\n";
    std::cout << "Expected Y:\n[[ 58,  64]\n [139, 154]]\n";
    std::cout << "Actual Y:\n";
    print_tensor(std::cout, Y);

    std::cout << "Expected Loss1: 415.0\n";
    std::cout << "Actual Loss1: ";
    print_tensor(std::cout, loss1);

    // Backward Pass
    loss1.accumulate_grad(Tensor<float>(1.0f, cpu_dev), false); 
    loss1.backward();

    std::cout << "\n--- Gradients (2D Matmul) ---\n";
    
    std::cout << "Grad A (dL/dY @ B^T):\n";
    std::cout << "Expected:\n[[ 15, 19, 23]\n [ 15, 19, 23]]\n";
    std::cout << "Actual:\n";
    print_tensor(std::cout, A.grad().value());

    std::cout << "Grad B (A^T @ dL/dY):\n";
    std::cout << "Expected:\n[[ 5, 5]\n [ 7, 7]\n [ 9, 9]]\n";
    std::cout << "Actual:\n";
    print_tensor(std::cout, B.grad().value());


    // ==========================================
    // TEST 2: BATCHED MATMUL
    // ==========================================
    std::cout << "\n==========================================\n";
    std::cout << "   TEST 2: BATCHED MATMUL (B=2, M=2, K=2, N=2)\n";
    std::cout << "==========================================\n";

    Tensor<float> C({2, 2, 2}, 0.0f, cpu_dev);
    C.set_data({1.0f, 2.0f, 
                3.0f, 4.0f,  // Batch 0
                
                5.0f, 6.0f, 
                7.0f, 8.0f});// Batch 1
    C.set_requires_grad(true);

    // D is broadcasted across the batch dimension
    Tensor<float> D({2, 2}, 0.0f, cpu_dev);
    D.set_data({1.0f, 0.0f, 
                0.0f, 2.0f});
    D.set_requires_grad(true);

    auto Z = matmul(C, D);
    auto loss2 = Z.sum({0, 1, 2}, false);

    loss2.realize();

    std::cout << "\n--- Forward Pass (Batched) ---\n";
    std::cout << "Expected Z:\n";
    std::cout << "Batch 0:\n[[ 1, 4]\n [ 3, 8]]\n";
    std::cout << "Batch 1:\n[[ 5, 12]\n [ 7, 16]]\n";
    std::cout << "Actual Z:\n";
    print_tensor(std::cout, Z);

    std::cout << "Expected Loss2: 56.0\n";
    std::cout << "Actual Loss2: ";
    print_tensor(std::cout, loss2);

    // Backward Pass
    loss2.accumulate_grad(Tensor<float>(1.0f, cpu_dev), false); 
    loss2.backward();

    std::cout << "\n--- Gradients (Batched) ---\n";

    std::cout << "Grad C (dL/dZ @ D^T for each batch):\n";
    std::cout << "Expected:\n";
    std::cout << "Batch 0:\n[[ 1, 2]\n [ 1, 2]]\n";
    std::cout << "Batch 1:\n[[ 1, 2]\n [ 1, 2]]\n";
    std::cout << "Actual:\n";
    print_tensor(std::cout, C.grad().value());

    std::cout << "Grad D (Sum over batch(C^T @ dL/dZ)):\n";
    std::cout << "Expected:\n[[ 16, 16]\n [ 20, 20]]\n";
    std::cout << "Actual:\n";
    print_tensor(std::cout, D.grad().value());

    return 0;
}