#include "gradc/gradc.hpp"
#include "gradc/nn/generators/gpt_generator.hpp"
#include <iostream>
#include <string>

using namespace gradc;

int main() {
    try {
        Device cpu(DeviceType::CPU);
        Device gpu(DeviceType::CUDA, 0);

        std::cout << "Loading tokenizer vocab..." << std::endl;
        BytePairEncoding bpe;
        bpe.load_vocab("C:/Local Projects/GradCraft/data/vocab/vocab.bin"); 

        // must match training!!!!!!!!
        int64_t seq_len = 512;
        int64_t vocab_size = 8192;
        int64_t embed_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 11;
        float calc_eps = 1e-5f;

        std::cout << "Initializing MALLMOC-180 on GPU." << std::endl;
        NormalInit<float> dummy_init(0.0f, 0.02f); 
        GPT<float> model(vocab_size, seq_len, embed_dim, num_heads, num_layers, dummy_init, dummy_init, calc_eps);
        model.to(gpu);
        model.eval(); // put in eval (doesn't change shi but still)

        std::string model_path = "C:/Local Projects/GradCraft/models/mallmoc-180/trained_model.bin";
        std::cout << "Loading weights from: " << model_path << std::endl;
        
        auto model_state = load_tensor_checkpoint<float>(model_path);
        model.load_state_dict(model_state);
        std::cout << "Weights loaded successfully.\n" << std::endl;

        int64_t max_tokens = 80; 
        int64_t num_sequences = 1;
        std::optional<float> temperature = std::nullopt;
        //std::optional<float> temperature = 1.2f;

        std::string prompt;
        std::cout << "MALLMOC ready. Enter prompt (type 'quit' to exit): \n";
        
        while (true) {
            std::cout << "\n> ";
            std::getline(std::cin, prompt);
            size_t pos;
            while ((pos = prompt.find("\\n")) != std::string::npos) {
                prompt.replace(pos, 2, "\n");
            }
            
            if (prompt == "quit") break;
            if (prompt.empty()) continue;

            std::cout << "Generating...\n" << std::endl;
            
            std::vector<std::string> results = GPTGenerator<float>::run_inference(model, bpe, seq_len, prompt, max_tokens, num_sequences, cpu, gpu, temperature);

            std::cout << "--- Output ---\n";
            std::cout << results[0] << "\n----------------\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}