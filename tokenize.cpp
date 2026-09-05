#include "gradc/gradc.hpp"

int main() {
    try {
        std::string vocab_dir = "C:/Local Projects/autograd_cpp/data/vocab";
        std::string dataset_dir = "C:/Local Projects/autograd_cpp/data/datasets";
        std::string vocab_path = vocab_dir + "/vocab.bin";
        std::string output_path = dataset_dir + "/cosmo_cpp.bin";

        std::filesystem::create_directories(vocab_dir);
        std::filesystem::create_directories(dataset_dir);

        std::vector<std::string> vocab_files = {
            "C:/Local Projects/autograd_cpp/data/raw_data/cosmopedia/cosmo_chunk_000.txt",
            "C:/Local Projects/autograd_cpp/data/raw_data/cpp/cpp_chunk_000.txt"
        };

        std::vector<std::string> all_files;
        for (int i = 0; i < 30; ++i) {
            all_files.push_back(std::format("C:/Local Projects/autograd_cpp/data/raw_data/cosmopedia/cosmo_chunk_{:03d}.txt", i));
            all_files.push_back(std::format("C:/Local Projects/autograd_cpp/data/raw_data/cpp/cpp_chunk_{:03d}.txt", i));
        }

        std::cout << "Creating vocab." << std::endl;
        gradc::TokenManager::create_vocab_out_of_files(vocab_path, vocab_files);

        std::cout << "Encoding dataset." << std::endl;
        gradc::TokenManager::encode_dataset(output_path, vocab_path, all_files);

        std::cout << "Tokenization complete.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}