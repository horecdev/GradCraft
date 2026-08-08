#pragma once

#include "../core/detail/shape_inference.hpp"
#include "../core/tensor.hpp"
#include "../graph/nodes/reduce_nodes.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::sum(const std::vector<int64_t>& red_axes, bool keepdims) const {
        ReductionMetadata reduction_metadata = infer_reduction_metadata(m_shape, red_axes, keepdims);
        Tensor result = Tensor(reduction_metadata.result_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SumNode<T>>(*this, reduction_metadata);

        return result;
    }

    template <typename T>
    template <typename OutT>
    Tensor<OutT> Tensor<T>::mean(const std::vector<int64_t>& red_axes, bool keepdims) const {
        Tensor<OutT> promoted_self = this->template cast<OutT>(); // first: cast source into right type. Then just add MeanNode.
        ReductionMetadata reduction_metadata = infer_reduction_metadata(promoted_self.m_shape, red_axes, keepdims);
        Tensor<OutT> result = Tensor<OutT>(reduction_metadata.result_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<MeanNode<OutT>>(std::move(promoted_self), reduction_metadata);

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::max(const std::vector<int64_t>& red_axes, bool keepdims) const {
        ReductionMetadata reduction_metadata = infer_reduction_metadata(m_shape, red_axes, keepdims);
        Tensor<T> result = Tensor<T>(reduction_metadata.result_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<MaxNode<T>>(*this, reduction_metadata);

        return result;
    }

    template <typename T>
    Tensor<int64_t> Tensor<T>::argmax(int64_t dim, bool keepdims) {
        if (this->_get_storage()->data() == nullptr) {
            throw std::runtime_error("Tried invoking an eager function (argmax) on an unrealized tensor. Call .realize() first.");
        }
        std::vector<int64_t> red_axes = std::vector<int64_t>({dim});
        ReductionMetadata reduction_metadata = infer_reduction_metadata(m_shape, red_axes, keepdims);
        Tensor<int64_t> result = Tensor<int64_t>(reduction_metadata.result_shape, this->device(), uninitialized);
        dispatch(this->device(), ArgExtrOp::ArgMax, dim, result, *this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::softmax(int64_t dim, bool keepdims) const requires std::is_floating_point_v<T> {
        ReductionMetadata reduction_metadata = infer_reduction_metadata(m_shape, {dim}, keepdims);
        Tensor result = Tensor(reduction_metadata.result_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SoftmaxNode<T>>(*this, reduction_metadata);

        return result;
    }
}