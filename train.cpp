#include "gradc/gradc.hpp"
#include <iostream>

using namespace gradc;
int main() {
    try {
        Device gpu(DeviceType::CUDA, 0);
        DataLoader loader = DataLoader("C:/Local Projects/autograd_cpp/data/datasets/cosmo_cpp.bin");

        // prep
        int64_t B = 4;
        int64_t seq_len = 1024;
        int64_t vocab_size = 32768;
        int64_t embed_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 12;

        NormalInit<float> init(0.0f, 0.02f);

        cudaStream_t copy_stream = create_stream();
        cudaEvent_t event = create_event();

        GPT<float> model(vocab_size, seq_len, embed_dim, num_heads, num_layers, init, 1e-5f);
        model.to(gpu);

        AdamW<float> optimizer(model.named_parameters(), 3e-4f); // karpathy constant!!!

        std::cout << "Number of params: " << std::to_string(model.num_params());

        for (int64_t i = 0; i < 5; ++i) {
            auto [X, Y] = loader.next_batch(B, seq_len, Device(DeviceType::CPU));
            X = X.to_async(gpu, copy_stream, event);
            Tensor<float> logits = model.forward(X);
            
            Tensor<float> loss = softmax_crossentropy_fast(logits, Y, 1e-5f);
            model.zero_grad();
            loss.backward();
            std::cout << "Loss: " << std::to_string(loss.item());
            optimizer.step();
            
        }
        std::cout << "Finished." << std::endl;
        std::cout << "HWM (GB): " << CUDAMemPool::get().get_hwm_gb() << std::endl;
        return 0;

        
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}