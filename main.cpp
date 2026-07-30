#include <iostream>
#include <vector>
#include "gradc/gradc.hpp" // IWYU pragma: keep

using namespace gradc;

int main() {
    Tensor<float> w({3, 3}, 2.0f, Device(DeviceType::CUDA, 0));
    w.set_requires_grad(true);

    Tensor<float> x({1, 3}, 4.0f, Device(DeviceType::CUDA, 0)); 
    x.set_requires_grad(true);

    Tensor<float> b({6}, 1.5f, Device(DeviceType::CUDA, 0));
    b.set_requires_grad(true);
    std::cout << "Created tensors" << std::endl;

    auto y = (w * x) + w; 
    auto y_slice = y[Slice(0, 2), _]; 

    auto y_reshaped = y_slice.reshape({6});
    
    std::vector<Tensor<float>> to_concat = {y_reshaped, b};
    auto concat_res = lazy_concat(to_concat, 0); 
    
    auto loss = concat_res.sum({0}, false);
    loss.realize();

    std::cout << "--- Forward Pass Result ---\nLoss: ";
    print_tensor(std::cout, loss);

    // 3. Backward pass
    loss.accumulate_grad(Tensor<float>(1.0f)); 
    std::cout << "accumulate grad" << std::endl;
    loss.backward();
    std::cout << "backward grad" << std::endl;

    // 4. Print Gradients
    std::cout << "\n--- Gradients ---\n";
    
    std::cout << "Grad w (Row 2 sliced out, should be 0.0):\n";
    print_tensor(std::cout, w.grad().value());

    std::cout << "\nGrad x (Broadcasted, should be summed across rows 0 and 1):\n";
    print_tensor(std::cout, x.grad().value());

    std::cout << "\nGrad b (Passed through concat and sum, should be 1.0):\n";
    print_tensor(std::cout, b.grad().value());

    return 0;
    
}