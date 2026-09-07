#include "gradc/gradc.hpp"
#include <iostream>

using namespace gradc;
int main() {
    try {
        Device gpu(DeviceType::CUDA, 0);

        int64_t B = 4;
        int64_t T_seq = 1024;
        int64_t vocab_size = 32768;
        int64_t embed_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 12;

        std::cout << "Initializing model." << std::endl;
        NormalInit<float> init(0.0f, 0.02f); //
        
        GPT<float> model(vocab_size, T_seq, embed_dim, num_heads, num_layers, init, 1e-5f);

        std::cout << "Number of params: " << std::to_string(model.num_params());

        std::cout << "Moving weights to GPU" << std::endl;
        model.to(gpu);

        AdamW<float> optimizer(model.named_parameters(), 3e-4f);

        Tensor<int64_t> X = Tensor<int64_t>::ones({B, T_seq}, gpu);
        Tensor<float> targets = Tensor<float>::ones({B * T_seq, vocab_size}, gpu);

        for (int64_t i = 0; i < 5; ++i) {
            std::cout << "Running fwd." << std::endl;
            Tensor<float> logits = model.forward(X);

            Tensor<float> flat_logits = logits.reshape({B * T_seq, vocab_size});
            
            Tensor<float> loss = softmax_crossentropy(flat_logits, targets, 1, 1e-5f);

            std::cout << "Running Backward Pass (HWM)" << std::endl;
            model.zero_grad();
            loss.backward();
            std::cout << "Loss: " << std::to_string(loss.item());
            optimizer.step();
            
        }
        std::cout << "Success." << std::endl;
        std::cout << "HWM (GB): " << CUDAMemPool::get().get_hwm_gb() << std::endl;
        return 0;

        
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}