#include "gradc/gradc.hpp"
#include <iostream>
#include <iomanip>

using namespace gradc;

// 1. Define the Neural Network Architecture
class SimpleNet : public Module<float> {
private:
    Linear<float> fc1;
    Linear<float> fc2;

public:
    SimpleNet() 
        : fc1(3, 16, KaimingNormalInit<float>(0.0f), ZerosInit<float>()),
          fc2(16, 1, KaimingNormalInit<float>(0.0f), ZerosInit<float>()) 
    {
        this->register_module("fc1", &fc1);
        this->register_module("fc2", &fc2);
    }

    Tensor<float> forward(Tensor<float> x) {
        Tensor<float> h = fc1.forward(x).relu();
        Tensor<float> out = fc2.forward(h);
        return out;
    }
};

int main() {
    try {
        std::cout << std::setprecision(6) << std::fixed;
        Device dev(DeviceType::CPU);
        
        SimpleNet model;
        model.to(dev);
        
        float lr = 0.005f;
        SGD<float> optimizer(model.named_parameters(), lr);
        Tensor<float> X = Tensor<float>::normal({8, 3}, 0.0f, 1.0f, dev);
        X.realize(); 
        X.make_leaf();
        
        Tensor<float> Y = Tensor<float>::ones({8, 1}, dev); 
        Y.realize(); 
        Y.make_leaf();
        
        std::cout << "Starting Training...\n";
        for (int epoch = 0; epoch <= 1000; ++epoch) {
            
            model.zero_grad();
            
            Tensor<float> preds = model.forward(X);
            Tensor<float> loss = mse_loss(preds, Y);
            
            loss.realize();
            loss.accumulate_grad(Tensor<float>::ones(loss.shape(), dev));
            loss.backward(false);
            
            optimizer.step();
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << " | MSE Loss: " << loss.item() << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[GRAD-C FATAL ERROR]: " << e.what() << "\n\n";
    }

    return 0;
}