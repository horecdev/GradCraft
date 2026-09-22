#include "gradc/gradc.hpp" // IWYU pragma: keep
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

int main(int argc, char* argv[]) {
    try {
        // defaults and argparse
        std::string raw_data_dir = "./data/raw_data";
        std::string vocab_path = "./data/vocab/vocab.bin";
        std::string output_path = "./data/datasets/dataset.bin";
        int32_t vocab_size = 8192;
        int64_t sample_mb = 200;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--data_dir" && i + 1 < argc) {raw_data_dir = argv[++i];}
            else if (arg == "--vocab_path" && i + 1 < argc) {vocab_path = argv[++i];}
            else if (arg == "--output_path" && i + 1 < argc) {output_path = argv[++i];}
            else if (arg == "--vocab_size" && i + 1 < argc) {vocab_size = std::stoi(argv[++i]);}
            else if (arg == "--sample_mb" && i + 1 < argc) {sample_mb = std::stoll(argv[++i]);}
        }

        std::filesystem::create_directories(std::filesystem::path(vocab_path).parent_path());
        std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());

        // collect files from the directory user passed (everything that is .txt)
        std::vector<std::string> all_files;
        if (std::filesystem::exists(raw_data_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(raw_data_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    all_files.push_back(entry.path().string());
                }
            }
        }

        if (all_files.empty()) {
            std::cerr << "Error: No .txt files found in: " << raw_data_dir << std::endl;
            return 1;
        }

        if (vocab_size < 260) {
            throw std::runtime_error("Error: vocab_size must be >= 260.");
        }

        // create vocab !!! 
        if (!std::filesystem::exists(vocab_path)) {
            gradc::TokenManager::create_vocab_out_of_files(vocab_path, all_files, vocab_size, sample_mb);
        } else {
            std::cout << "Vocab exists at " << vocab_path << ". Skipping creation." << std::endl;
        }

        // encode the full file
        std::cout << "Encoding full dataset to: " << output_path << std::endl;
        gradc::TokenManager::encode_dataset(output_path, vocab_path, all_files);

        std::cout << "Tokenization complete." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}