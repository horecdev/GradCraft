#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/loss_nodes.hpp"

#include <memory>

namespace gradc {

    template <typename T>
    Tensor<T> softmax_crossentropy(Tensor<T> flat_logits, Tensor<T> flat_targets, int64_t distrib_dim, T eps) requires std::is_floating_point_v<T> {
        if (std::ssize(flat_logits.shape()) != std::ssize(flat_targets.shape()) || std::ssize(flat_logits.shape()) != 2) {
            throw std::runtime_error("softmax_crossentropy accepts only 2D tensors");
        }
        Device target_device = infer_assert_device(flat_logits, flat_targets);
        ReductionMetadata softmax_red_meta = infer_reduction_metadata(flat_targets.shape(), {distrib_dim}, true);
        ReductionMetadata loss_red_meta = infer_reduction_metadata(flat_targets.shape(), {0, 1}, false);
        int64_t batch_size = distrib_dim == 1 ? flat_logits.m_shape[0] : flat_logits.m_shape[1];
        Tensor<T> result = Tensor<T>(loss_red_meta.result_shape, flat_logits.requires_grad(), lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<SoftmaxCrossEntropyLossNode<T>>(std::move(flat_logits), std::move(flat_targets), std::move(softmax_red_meta), std::move(loss_red_meta), batch_size, eps);

        return result;
    }
}
