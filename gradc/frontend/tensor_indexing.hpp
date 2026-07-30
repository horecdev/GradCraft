#pragma once

#include "../core/detail/tensor_lob_view.hpp"
#include "../core/tensor.hpp"
#include "../graph/nodes/view_nodes.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gradc {
    template <typename T>
    template <typename... Args> // ... attached to typename means: create a bucket of types and name it Args.
    Tensor<T> Tensor<T>::operator[](Args... args) const { // ... next to the type bucket, before args: Look element-wise at both. Put type inside type bucket, argument inside args.
        // The compiler sees you passed x: short, y: int, z: int64_t, and automatically creates a template (short var_0, int var_1, int64_t var_2)
        // For class templates: pass <float> etc. For funcion/operator templates you dont have to.
        if (sizeof...(args) != std::ssize(m_shape)) { // sizeof... literally means "count elements in the bucket". Its a built-in token.
            throw std::out_of_range("Coordinate count does not match tensor dimensions.");
        }

        std::array<IndexDesc, sizeof...(args)> arr = {IndexDesc(args)...}; // when you do [expression(args)...] it means: apply expression to every arg, and separate it with comas.
        std::vector<IndexDesc> descriptors(arr.begin(), arr.end());

        Tensor<T> result = create_lobotomized_slice_view(*this, descriptors);

        result.m_requires_grad = m_requires_grad;
        result.m_state->m_creation_op = std::make_unique<SliceNode<T>>(*this, descriptors);

        return result;
    }

    template <typename T>
    T Tensor<T>::item() const {
        if (std::ssize(m_shape) == 0) {
            if (m_state->m_storage->data() == nullptr) {
                throw std::runtime_error("Called .item() on an empty tensor buffer. (call .realize first)");
            }
            
            if (this->device().is_cpu()) {
                return (m_state->m_storage->m_data)[m_offset];
            }
            else if (this->device().is_cuda()) {
                T host_val;
                cudaSetDevice(this->device().index);
                cudaError_t err = cudaMemcpy(&host_val, this->_get_storage().data(), sizeof(T), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA memcpy failed on .item()");
                }
                return host_val;
            }
        }
        else {
            throw std::runtime_error(".item() can be called only on 0-dimensional tensors.");
        }
    }
}