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

        RedMeta softmax_red_meta = infer_red_meta(flat_logits.shape(), {distrib_dim}, true);
        RedMeta loss_red_meta = infer_red_meta(flat_logits.shape(), {0, 1}, false);
        int64_t batch_size = distrib_dim == 1 ? flat_logits.m_shape[0] : flat_logits.m_shape[1];
        bool requires_grad = flat_logits.requires_grad() || flat_targets.requires_grad();
        Tensor<T> result = Tensor<T>(loss_red_meta.result_shape, requires_grad, lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<SoftmaxCrossEntropyLossNode<T>>(std::move(flat_logits), std::move(flat_targets), std::move(softmax_red_meta), std::move(loss_red_meta), batch_size, eps);

        return result;
    }

    template <typename T>
    Tensor<T> mse_loss(Tensor<T> preds, Tensor<T> targets) requires std::is_floating_point_v<T> {
        if (preds.shape() != targets.shape()) {
            throw std::runtime_error("preds and targets in MSELoss must be of the same shape.");
        }
        
        Device target_device = infer_assert_device(preds, targets);

        std::vector<int64_t> all_axes;
        all_axes.reserve(std::ssize(preds.shape()));
        for (int64_t i = 0; i < std::ssize(preds.shape()); ++i) {
            all_axes.push_back(i);
        }

        RedMeta mse_red_meta = infer_red_meta(preds.shape(), all_axes, false);
        bool requires_grad = preds.requires_grad() || targets.requires_grad();
        Tensor<T> result = Tensor<T>(mse_red_meta.result_shape, requires_grad, lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<MSELossNode<T>>(std::move(preds), std::move(targets), std::move(mse_red_meta));

        return result;
    }
}
