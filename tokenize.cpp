#include "gradc/gradc.hpp" // IWYU pragma: keep

int main() {
    try {
        std::string vocab_dir = "C:/Local Projects/GradCraft/data/vocab";
        std::string dataset_dir = "C:/Local Projects/GradCraft/data/datasets";
        std::string vocab_path = vocab_dir + "/vocab.bin";
        std::string output_path = dataset_dir + "/python_edu.bin";

        std::filesystem::create_directories(vocab_dir);
        std::filesystem::create_directories(dataset_dir);

        std::vector<std::string> all_files;
        for (int i = 0; i < 122; ++i) {
            all_files.push_back(std::format("C:/Local Projects/GradCraft/data/raw_data/python_edu_{:04d}.txt", i));
        }

        if (!std::filesystem::exists(vocab_path)) {
            gradc::TokenManager::create_vocab_out_of_files(vocab_path, all_files);
        } 
        else {
            std::cout << "Vocab already exists. Skipping creating." << std::endl;
        }

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