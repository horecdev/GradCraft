#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/activation_nodes.hpp"

#include <memory>

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::relu() const {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<ReLUNode<T>>(*this);
        return result;
    }

    template <typename T>
    template <typename U>
    Tensor<T> Tensor<T>::sigmoid() const requires std::is_floating_point_v<U> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SigmoidNode<T>>(*this);
        return result;
    }

    template <typename T>
    template <typename U>
    Tensor<T> Tensor<T>::tanh() const requires std::is_floating_point_v<U> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<TanHNode<T>>(*this);
        return result;
    }

    template <typename T>
    template <typename U>
    Tensor<T> Tensor<T>::silu() const requires std::is_floating_point_v<U> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SiLUNode<T>>(*this);
        return result;
    }

    template <typename T>
    template <typename U>
    Tensor<T> Tensor<T>::gelu() const requires std::is_floating_point_v<U> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<GeLUNode<T>>(*this);
        return result;
    }
}
