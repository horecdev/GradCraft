#include "gradc/gradc.hpp" // IWYU pragma: keep
#include "gradc/nn/generators/gpt_generator.hpp"
#include <iostream>
#include <string>
#include <filesystem>

using namespace gradc;

int main(int argc, char* argv[]) {
    try {
        // defaults
        std::string vocab_path = "./data/vocab/vocab.bin";
        std::string config_path = "./models/mallmoc/config.bin";
        std::string weights_path = "./models/mallmoc/trained_model.bin";
        int64_t max_tokens = 512;
        int64_t num_sequences = 1;
        std::optional<float> temperature = std::nullopt;
        int64_t top_k = 20;

        // argparse
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--vocab" && i + 1 < argc) {vocab_path = argv[++i];}
            else if (arg == "--config" && i + 1 < argc) {config_path = argv[++i];}
            else if (arg == "--weights" && i + 1 < argc) {weights_path = argv[++i];}
            else if (arg == "--max_tokens" && i + 1 < argc) {max_tokens = std::stoll(argv[++i]);}
            else if (arg == "--temp" && i + 1 < argc) {temperature = std::stof(argv[++i]);}
            else if (arg == "--top_k" && i + 1 < argc) {top_k = std::stoll(argv[++i]);}
        }

        // verify if its all there
        if (!std::filesystem::exists(vocab_path)) {
            std::cerr << "Error: Vocab file missing at: " << vocab_path << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(config_path)) {
            std::cerr << "Error: Config file missing at: " << config_path << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(weights_path)) {
            std::cerr << "Error: Weights file missing at: " << weights_path << std::endl;
            return 1;
        }

        // initialize the model
        Device cpu(DeviceType::CPU);
        Device gpu(DeviceType::CUDA, 0);

        std::cout << "Loading vocabulary from: " << vocab_path<< std::endl;
        BytePairEncoding bpe;
        bpe.load_vocab(vocab_path);

        std::cout << "Loading model configuration from: " << config_path << std::endl;
        GPTConfig cfg = load_gpt_config(config_path);

        std::cout << "Initializing GPT (" << cfg.num_layers << "L, " << cfg.embed_dim << "D, " << cfg.num_heads << "H)..." << std::endl;
        NormalInit<float> dummy_init(0.0f, 0.02f);
        GPT<float> model(cfg, dummy_init, dummy_init);
        model.to(gpu);
        model.eval();

        std::cout << "Loading weights from: " << weights_path << std::endl;
        auto model_state = load_tensor_checkpoint<float>(weights_path);
        model.load_state_dict(model_state);
        std::cout << "Model ready for generation.\n" << std::endl;

        std::string prompt;
        std::cout << "Enter prompt (type 'quit' to exit):\n";
        
        // inference session <3
        while (true) {
            std::cout << "\n> ";
            std::getline(std::cin, prompt);
            if (prompt == "quit") {break;}
            if (prompt.empty()) {continue;}

            std::cout << "Generating...\n";
            auto results = GPTGenerator<float>::run_inference(model, bpe, cfg.max_seq_len, prompt, max_tokens, num_sequences, cpu, gpu, temperature, top_k);

            std::cout << "--- Output ---\n";
            std::cout << results[0] << "\n----------------\n";
        }
        return 0;
    } 
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}