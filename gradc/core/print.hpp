#pragma once

#include "tensor.hpp"
#include "types.hpp"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace gradc {

    template <typename T>
    std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
        stream << "[";
        for (int64_t i = 0; i < std::ssize(vector); ++i) {
            if (i == std::ssize(vector) - 1) {
                stream << vector[i];
            }
            else {
                stream << vector[i] << ", ";
            }
            
        }
        stream << "]";
        return stream;
    }

    template <typename T>
    void print_dim(std::ostream& stream, const Tensor<T>& source, const PrintOptions& opts, int64_t current_dim, int64_t base_offset, bool is_last) {
        if (current_dim == std::ssize(source.m_shape) - 1) {
            stream << std::string(current_dim * opts.dim_indentation, ' ') << "[";
            // print values along the dim
            if (source.m_shape[current_dim] > 2 * opts.edge_items) {
                // print edge_items first, edge_items last
                for (int64_t i = 0; i < opts.edge_items; ++i) {
                    int64_t target_idx = base_offset + i * source.m_strides[current_dim];
                    stream << read_element(source, target_idx);
                    stream << ", ";
                }
                stream << "..., ";

                for (int64_t i = source.m_shape[current_dim] - opts.edge_items; i < source.m_shape[current_dim]; ++i) {
                    int64_t target_idx = base_offset + i * source.m_strides[current_dim];
                    stream << read_element(source, target_idx);
                    if (i != source.m_shape[current_dim] - 1) {
                        stream << ", ";
                    }
                }
            }

            else {
                // print everything
                for (int64_t i = 0; i < source.m_shape[current_dim]; ++i) {
                    int64_t target_idx = base_offset + i * source.m_strides[current_dim];
                    stream << read_element(source, target_idx);
                    if (i != source.m_shape[current_dim] - 1) {
                         stream << ", ";
                    }
                } 
            }
            stream << "]";
            if (!is_last) {
                stream << ",";
            }
            stream << std::endl;
        }

        else {
            stream << std::string(current_dim * opts.dim_indentation, ' ') << "[" << std::endl;
            if (source.m_shape[current_dim] > 2 * opts.edge_items) {
                // print edge_items first, edge_items last
                for (int64_t i = 0; i < opts.edge_items; ++i) {
                    print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                }
                stream << std::string((current_dim + 1) * opts.dim_indentation, ' ') << "..." << std::endl;

                for (int64_t i = source.m_shape[current_dim] - opts.edge_items; i < source.m_shape[current_dim]; ++i) {
                    if (i == source.m_shape[current_dim] - 1) {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], true);
                    }
                    else {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                    }
                }
            }

            else {
                // print everything
                for (int64_t i = 0; i < source.m_shape[current_dim]; ++i) {

                    if (i == source.m_shape[current_dim] - 1) {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], true);
                    }
                    else {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                    }
                }
            } 

            stream << std::string(current_dim * opts.dim_indentation, ' ');
            stream << "]";
            if (!is_last) {
                stream << ",";
            }
            stream << std::endl;
        }
    }

    template <typename T>
    std::ostream& print_tensor(std::ostream& stream, const Tensor<T>& source, const PrintOptions opts = {}) { // = {} means use initializer list if nothing is provided (but empty so default construct)
        if (source._get_storage()->data() == nullptr) {
            stream << "Attempted printing tensor without data: aborting (call .realize() first)" << std::endl;
            return stream;
        }
        if (std::ssize(source.m_shape) == 0) {
            if (opts.show_metadata) {
                stream << "requires_grad: " << source.m_requires_grad << std::endl;
            }
            stream << "Tensor(" << source.item() << ")" << std::endl;
            return stream;
        }
        
        else {
            if (opts.show_metadata) {
                std::cout << "Shape: " << source.m_shape << " | Strides: " << source.m_strides << " | Grad: " << source.m_requires_grad << std::endl;
            }
            if (source.device().is_cuda()) {cudaSetDevice(source.device().index);}

            print_dim(stream, source, opts, 0, source.m_offset, true);
            return stream;
        }
    }

    template <typename T>
    T read_element(const Tensor<T>& source, int64_t mem_idx) {
        const T* ptr = source._get_storage()->data() + mem_idx;
        if (source.device().is_cpu()) {
            return *ptr;
        }
        else if (source.device().is_cuda()) {
            T val;
            cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost);
            return val;
        }
        else {
            throw std::runtime_error("Invalid device");
        }
    }
}

