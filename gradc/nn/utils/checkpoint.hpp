#pragma once

#include "gradc/core/tensor.hpp"
#include <unordered_map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace gradc {
    template <typename T>
    void save_tensor_checkpoint(const std::unordered_map<std::string, Tensor<T>>& state_dict, const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {throw std::runtime_error("Failed to open file for saving: " + path);}

        uint32_t type_tag = sizeof(T);
        out.write(reinterpret_cast<const char*>(&type_tag), sizeof(type_tag)); // write how many bytes T is

        int64_t num_tensors = std::ssize(state_dict);
        // literally write raw bytes of int64_t as chars (1 byte each)
        out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(int64_t));

        for (const auto& [name, tensor] : state_dict) {
            int64_t name_len = std::ssize(name);
            out.write(reinterpret_cast<const char*>(&name_len), sizeof(int64_t));
            out.write(name.c_str(), name_len);

            const std::vector<int64_t>& shape = tensor.shape();
            int64_t rank = std::ssize(tensor.shape());
            out.write(reinterpret_cast<const char*>(&rank), sizeof(int64_t));
            out.write(reinterpret_cast<const char*>(shape.data()), rank * sizeof(int64_t));

            int64_t total_bytes = tensor.volume() * sizeof(T);
            out.write(reinterpret_cast<const char*>(&total_bytes), sizeof(int64_t));
            out.write(reinterpret_cast<const char*>(tensor._get_storage()->data()), total_bytes);
        }
    }

    template <typename T>
    std::unordered_map<std::string, Tensor<T>> load_tensor_checkpoint(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {throw std::runtime_error("Failed to open file for loading: " + path);}

        uint32_t type_tag;
        in.read(reinterpret_cast<char*>(&type_tag), sizeof(type_tag)); // read in the same 
        if (type_tag != sizeof(T)) {
            throw std::runtime_error("Tried reading a file whose type is " + std::to_string(type_tag) + "-bytes with a type having " + std::to_string(sizeof(T)) + "-bytes.");
        }

        std::unordered_map<std::string, Tensor<T>> dict;

        int64_t num_tensors;
        // reads and inputs sizeof(int64_t) bytes into adress of num_tensors
        in.read(reinterpret_cast<char*>(&num_tensors), sizeof(int64_t));

        for (int64_t i = 0; i < num_tensors; ++i) {
            int64_t name_length;
            in.read(reinterpret_cast<char*>(&name_length), sizeof(int64_t));
            std::string name(name_length, '\0');
            in.read(&name[0], name_length);
            
            int64_t rank;
            in.read(reinterpret_cast<char*>(&rank), sizeof(int64_t));
            std::vector<int64_t> shape(rank);
            in.read(reinterpret_cast<char*>(shape.data()), rank * sizeof(int64_t));
            
            Tensor<T> loaded = Tensor<T>(shape, Device(DeviceType::CPU), uninitialized);

            int64_t total_bytes;
            in.read(reinterpret_cast<char*>(&total_bytes), sizeof(int64_t));
            in.read(reinterpret_cast<char*>(loaded._get_storage()->data()), total_bytes);
            
            dict[name] = std::move(loaded);
        }

        return dict;
    }

    template <typename T>
    void save_scalar_checkpoint(const std::unordered_map<std::string, T>& state_dict, const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {throw std::runtime_error("Failed to open file for saving: " + path);}

        uint32_t type_tag = sizeof(T);
        out.write(reinterpret_cast<const char*>(&type_tag), sizeof(type_tag));

        int64_t num_scalars = std::ssize(state_dict);
        // literally write raw bytes of int64_t as chars (1 byte each)
        out.write(reinterpret_cast<const char*>(&num_scalars), sizeof(int64_t));

        for (const auto& [name, scalar] : state_dict) {
            int64_t name_len = std::ssize(name);
            out.write(reinterpret_cast<const char*>(&name_len), sizeof(int64_t));
            out.write(name.c_str(), name_len);

            int64_t bytes = sizeof(T); // write our scalar as T
            out.write(reinterpret_cast<const char*>(&scalar),  bytes);
        }
    }

    template <typename T>
    std::unordered_map<std::string, T> load_scalar(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {throw std::runtime_error("Failed to open file for loading: " + path);}

        uint32_t type_tag;
        in.read(reinterpret_cast<char*>(&type_tag), sizeof(type_tag));
        if (type_tag != sizeof(T)) {
            throw std::runtime_error("Tried reading a file whose type is " + std::to_string(type_tag) + "-bytes with a type having " + std::to_string(sizeof(T)) + "-bytes.");
        }

        std::unordered_map<std::string, T> dict;

        int64_t num_scalars;
        in.read(reinterpret_cast<char*>(&num_scalars), sizeof(int64_t));

        for (int64_t i = 0; i < num_scalars; ++i) {
            int64_t name_length;
            in.read(reinterpret_cast<char*>(&name_length), sizeof(int64_t));
            std::string name(name_length, '\0');
            in.read(&name[0], name_length);

            int64_t bytes = sizeof(T);
            T scalar;
            in.read(reinterpret_cast<char*>(&scalar), bytes);
            
            dict[name] = scalar;
        }

        return dict;
    }
}