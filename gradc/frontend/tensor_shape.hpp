#pragma once

#include "../core/detail/tensor_lob_view.hpp"
#include "../core/tensor.hpp"
#include "../graph/nodes/view_nodes.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::transpose(int64_t dim0, int64_t dim1) const {
        Tensor<T> result = lobotomized_transpose_view(*this, dim0, dim1);
        result.m_state->m_creation_op = std::make_unique<TransposeNode<T>>(*this, dim0, dim1); 
        result.m_requires_grad = m_requires_grad;
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int64_t>& axes) const {
        Tensor<T> permuted = lobotomized_permute_view(*this, axes);
        permuted.m_state->m_creation_op = std::make_unique<PermuteNode<T>>(*this, axes);
        permuted.m_requires_grad = m_requires_grad;
        return permuted;
    }

    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& target_shape) const {
        if (this->is_contiguous()) {
            Tensor<T> reshaped = lobotomized_reshape_view(*this, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(*this);
            reshaped.m_requires_grad = m_requires_grad;
            return reshaped;
        }
        else {
            Tensor<T> new_contig = this->contiguous();
            Tensor<T> reshaped = lobotomized_reshape_view(new_contig, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(new_contig);
            reshaped.m_requires_grad = new_contig.m_requires_grad;
            return reshaped;
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::squeeze(std::optional<int64_t> dim) const {
        Tensor<T> squeezed = lobotomized_squeeze_view(*this, dim);
        squeezed.m_state->m_creation_op = std::make_unique<SqueezeNode<T>>(*this);
        squeezed.m_requires_grad = m_requires_grad;
        return squeezed;
    }

    template <typename T>
    Tensor<T> Tensor<T>::unsqueeze(int64_t dim) const {
        Tensor<T> unsqueezed = lobotomized_unsqueeze_view(*this, dim);
        unsqueezed.m_state->m_creation_op = std::make_unique<UnsqueezeNode<T>>(*this);
        unsqueezed.m_requires_grad = m_requires_grad;
        return unsqueezed;
    }
}