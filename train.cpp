#include "gradc/gradc.hpp"
#include <iostream>

using namespace gradc;
int main() {
    try {
        Device gpu(DeviceType::CUDA, 0);
        DataLoader loader = DataLoader("C:/Local Projects/autograd_cpp/data/datasets/cosmo_cpp.bin");

        // prep
        int64_t B = 4;
        int64_t seq_len = 32;
        int64_t vocab_size = 32768;
        int64_t embed_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 16;

        NormalInit<float> init(0.0f, 0.02f);

        cudaStream_t copy_stream = create_stream();
        cudaEvent_t event = create_event();

        GPT<float> model(vocab_size, seq_len, embed_dim, num_heads, num_layers, init, 1e-5f);
        model.to(gpu);

        AdamW<float> optimizer(model.named_parameters(), 3e-4f);

        std::string num_params = std::format(std::locale("en_US.UTF-8"), "{:L}", model.num_params());
        std::cout << "Number of params: " << num_params << std::endl;;

        for (int64_t i = 0; i < 5; ++i) {
            auto [X, Y] = loader.next_batch(B, seq_len, Device(DeviceType::CPU));
            X = X.to_async(gpu, copy_stream, event);
            Y = Y.to_async(gpu, copy_stream, event);
            Tensor<float> logits = model.forward(X);

            std::cout << "Before reshape" << std::endl;
            Tensor<float> flat_logits = logits.reshape({-1, vocab_size});
            std::cout << "Before one-hot" << std::endl;
            Tensor<float> targets = one_hot_encode<float>(Y, vocab_size);

            std::cout << "Before reshape" << std::endl;
            Tensor<float> flat_targets = targets.reshape({-1, vocab_size});

            std::cout << "Before SCE" << std::endl;
            Tensor<float> loss = softmax_crossentropy_naive(flat_logits, flat_targets, 1, 1e-5f);
            std::cout << "Before realize" << std::endl;
            loss.realize();
            print_tensor(std::cout, loss);

            model.zero_grad();
            loss.backward();
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