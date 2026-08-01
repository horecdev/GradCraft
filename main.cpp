#include <iostream>
#include <vector>
#include "gradc/gradc.hpp"

using namespace gradc;

int main() {
    Tensor<float> w({2, 2}, 2.0f, Device(DeviceType::CPU, 0));
    w.set_requires_grad(true);

    Tensor<float> x({1, 2}, 4.0f, Device(DeviceType::CPU, 0)); 
    x.set_requires_grad(true);

    Tensor<float> b({2}, 1.5f, Device(DeviceType::CPU, 0));
    b.set_requires_grad(true);

    std::cout << "Created Tensors on CPU" << std::endl;

    auto y = (w * x) + w; 
    auto y_max = y.max({1}, false); // <--- Integrated MaxNode here!
    
    std::vector<Tensor<float>> to_concat = {y_max, b};
    auto concat_res = lazy_concat(to_concat, 0); 
    
    auto loss = concat_res.sum({0}, false);

    loss.realize();

    std::cout << "\n--- Forward Pass Result ---\n";
    std::cout << "Expected Loss: 23.0\n";
    std::cout << "Actual Loss: ";
    print_tensor(std::cout, loss);

    y.realize();
    Tensor<int64_t> y_argmax = y.argmax(1, false);

    std::cout << "\n--- Eager ArgMax Result ---\n";
    std::cout << "Expected ArgMax indices along dim 1: [0, 0] (since both columns are equal, first hit wins)\n";
    std::cout << "Actual ArgMax: ";
    print_tensor(std::cout, y_argmax);


    loss.accumulate_grad(Tensor<float>(1.0f, Device(DeviceType::CPU, 0)), false); 
    loss.backward();
    
    std::cout << "\n--- Gradients ---\n";
    
    std::cout << "Grad w:\n";
    std::cout << "Expected Grad w: [[5.0, 0.0], [5.0, 0.0]]\n";
    std::cout << "Actual Grad w:\n";
    print_tensor(std::cout, w.grad().value());

    std::cout << "\nGrad x:\n";
    std::cout << "Expected Grad x: [[4.0, 0.0]]\n";
    std::cout << "Actual Grad x:\n";
    print_tensor(std::cout, x.grad().value());

    std::cout << "\nGrad b:\n";
    std::cout << "Expected Grad b: [1.0, 1.0]\n";
    std::cout << "Actual Grad b:\n";
    print_tensor(std::cout, b.grad().value());

    return 0;
}