#pragma once

#include "../core/detail/shape_inference.hpp"
#include "../core/tensor.hpp"
#include "../graph/nodes/memory_nodes.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace gradc {

    template <typename T>
    Tensor<T> Tensor<T>::contiguous() const {
        Tensor result = Tensor(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<ContiguousNode<T>>(*this);
        return result;
    }


    template <typename T>
    Tensor<T> Tensor<T>::clone() const { 
        Tensor<T> tensor_copy = Tensor(m_shape, m_requires_grad, lazy, this->device());
        tensor_copy.m_state->m_creation_op = std::make_unique<CloneNode<T>>(*this);
        return tensor_copy;
    }



    template <typename T>
    template <typename TargetT>
    Tensor<TargetT> Tensor<T>::cast() const {
        if constexpr (std::is_same_v<T, TargetT>) {
            return *this;
        }
        else {
            Tensor<TargetT> new_tensor = Tensor<TargetT>(m_shape, m_requires_grad, lazy, this->device());
            new_tensor.m_state->m_creation_op = std::make_unique<CastNode<T, TargetT>>(*this);
            return new_tensor;
        }
    }

    template <typename T>
    Tensor<T> concat(std::vector<Tensor<T>> tensor_list, int64_t concat_dim) {
        Device target_device = infer_assert_device(tensor_list);

        const int64_t n_dim = std::ssize(tensor_list[0].m_shape);
        std::vector<int64_t> final_shape = tensor_list[0].m_shape;
        concat_dim = normalize_axis(concat_dim, n_dim);
        final_shape[concat_dim] = 0;
        bool requires_grad = false;

        for (const Tensor<T>& parent : tensor_list) {
            if (std::ssize(parent.m_shape) != n_dim) {
                throw std::runtime_error("Error during concat: Tensors must have the same number of dimensions.");
            }
            for (int64_t i = 0; i < n_dim; ++i) {
                if (i != concat_dim && parent.m_shape[i] != tensor_list[0].m_shape[i]) {
                    throw std::runtime_error("Error during concat: Tensors must match on non-concat dimensions.");
                }
            }
            final_shape[concat_dim] += parent.m_shape[concat_dim];
            requires_grad = requires_grad || parent.m_requires_grad;
        }

        Tensor<T> result = Tensor<T>(final_shape, requires_grad, lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<ConcatNode<T>>(std::move(tensor_list), concat_dim, std::move(final_shape));
        
        return result;
    }

    template <typename T>
    Tensor<T> stack(std::vector<Tensor<T>> tensor_list, int64_t stack_dim) {
        Device target_device = infer_assert_device(tensor_list);

        const int64_t n_dim = std::ssize(tensor_list[0].m_shape);
        std::vector<int64_t> final_shape = tensor_list[0].m_shape;
        concat_dim = normalize_axis(concat_dim, n_dim);
        final_shape[concat_dim] = 0;
        bool requires_grad = false;

        for (const Tensor<T>& parent : tensor_list) {
            if (std::ssize(parent.m_shape) != n_dim) {
                throw std::runtime_error("Error during concat: Tensors must have the same number of dimensions.");
            }
            for (int64_t i = 0; i < n_dim; ++i) {
                if (i != concat_dim && parent.m_shape[i] != tensor_list[0].m_shape[i]) {
                    throw std::runtime_error("Error during concat: Tensors must match on non-concat dimensions.");
                }
            }
            final_shape[concat_dim] += parent.m_shape[concat_dim];
            requires_grad = requires_grad || parent.m_requires_grad;
        }

        Tensor<T> result = Tensor<T>(final_shape, requires_grad, lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<ConcatNode<T>>(std::move(tensor_list), concat_dim, std::move(final_shape));
        
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::to(Device device) const {
        if (device == this->device()) {return *this;}

        if (!this->is_contiguous()) {
            Tensor<T> forced_contiguous = this->contiguous(); // compact data before sending over PCIE bus (dont copy whole underlying array)
            Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, device);
            result.m_state->m_creation_op = std::make_unique<ToNode<T>>(forced_contiguous, device);
            return result;
        }
        else {
            Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, device);
            result.m_state->m_creation_op = std::make_unique<ToNode<T>>(*this, device);
            return result;
        }
        
    }
}