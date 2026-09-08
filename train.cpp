#include "gradc/gradc.hpp"
#include <iostream>
#include <chrono>

using namespace gradc;
int main() {
    try {
        Device gpu(DeviceType::CUDA, 0);
        DataLoader loader = DataLoader("C:/Local Projects/autograd_cpp/data/datasets/cosmo_cpp.bin");

        // prep
        int64_t B = 1;
        int64_t seq_len = 128;
        int64_t vocab_size = 32768;
        int64_t embed_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 16;

        NormalInit<float> init(0.0f, 0.003535f);

        cudaStream_t copy_stream = create_stream();
        cudaEvent_t event = create_event();

        GPT<float> model(vocab_size, seq_len, embed_dim, num_heads, num_layers, init, 1e-5f);
        model.to(gpu);

        AdamW<float> optimizer(model.named_parameters(), 3e-4f);
        CosineScheduler<float> scheduler(&optimizer, 3e-4f, 3e-5f, 5, 10);

        std::string num_params = std::format(std::locale("en_US.UTF-8"), "{:L}", model.num_params());
        std::cout << "Number of params: " << num_params << std::endl;;

        int64_t tokens_per_step = B * seq_len;

        for (int64_t i = 0; i < 10; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto [X, Y] = loader.next_batch(B, seq_len, Device(DeviceType::CPU));
            X = X.to_async(gpu, copy_stream, event);
            Y = Y.to_async(gpu, copy_stream, event);
            Tensor<float> logits = model.forward(X);

            Tensor<float> loss = softmax_crossentropy_fast<float>(logits, Y, 1e-5f);

            loss.realize();
            float loss_val = loss.item();
            model.zero_grad();
            loss.backward();
            optimizer.step();

            auto end_time = std::chrono::high_resolution_clock::now();
            double step_seconds = std::chrono::duration<double>(end_time - start_time).count();
            double tok_per_sec = tokens_per_step / step_seconds;
            
            std::cout << "Step: " << i << " | Loss: " << loss_val << " | Time: " << (step_seconds * 1000.0) << " ms" << " | Speed: " << static_cast<int64_t>(tok_per_sec) << " tok/s" << std::endl;
        }
        CUDAMemPool::get().log_hwm();
        return 0;

        
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}