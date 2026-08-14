#pragma once

#include "gradc/core/tensor.hpp"
#include <unordered_map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace gradc {
    template <typename T>
    void save_checkpoint(const std::unordered_map<std::string, Tensor<T>>& state_dict, const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {throw std::runtime_error("Failed to open file for saving: " + path);}

        int64_t num_tensors = std::ssize(state_dict);
        // literally write raw bytes of int64_t as chars (1 byte each)
        out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(int64_t));

        for (const auto& [name, tensor] : state_dict) {
            
        }

    }
}