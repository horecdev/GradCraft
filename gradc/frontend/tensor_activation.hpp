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
    Tensor<T> Tensor<T>::sigmoid() const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SigmoidNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::tanh() const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<TanHNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::silu() const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SiLUNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::gelu() const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<GeLUNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::softmax(int64_t dim) const requires std::is_floating_point_v<T> {
        RedMeta red_meta = infer_red_meta(m_shape, {dim}, true);
        Tensor result = Tensor(this->shape(), m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SoftmaxNode<T>>(*this, std::move(red_meta));

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::dropout(T p) const requires std::is_floating_point_v<T> {
        Tensor result = Tensor(this->shape(), m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<DropoutNode<T>>(*this, p);

        return result;
    }
}
