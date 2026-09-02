#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/normalization_nodes.hpp"

#include <memory>

namespace gradc {
    template <typename T>
    Tensor<T> layernorm(Tensor<T> parent, Tensor<T> gamma, Tensor<T> beta, const std::vector<int64_t>& axes, T eps) requires std::is_floating_point_v<T> {
        Device target_device = infer_assert_device(parent, gamma, beta);
        RedMeta red_meta = infer_red_meta(parent.shape(), axes, true);
        std::vector<int64_t> norm_shape = get_normalized_shape(parent.shape(), axes);
        Tensor<T> result = Tensor<T>(parent.shape(), parent.requires_grad(), lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<LayerNormNode<T>>(std::move(parent), std::move(gamma), std::move(beta), std::move(red_meta), std::move(norm_shape), eps);

        return result;
    }

    template <typename T>
    Tensor<T> rmsnorm_naive(Tensor<T> parent, Tensor<T> gamma, const std::vector<int64_t>& axes, T eps) requires std::is_floating_point_v<T> {
        Device target_device = infer_assert_device(parent, gamma);
        RedMeta red_meta = infer_red_meta(parent.shape(), axes, true);
        std::vector<int64_t> norm_shape = get_normalized_shape(parent.shape(), axes);
        Tensor<T> result = Tensor<T>(parent.shape(), parent.requires_grad(), lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<RMSNormNaiveNode<T>>(std::move(parent), std::move(gamma), std::move(red_meta), std::move(norm_shape), eps);

        return result;
    }

    template <typename T>
    Tensor<T> rmsnorm_fast(Tensor<T> parent, Tensor<T> gamma, const std::vector<int64_t>& axes, T eps) requires std::is_floating_point_v<T> {
        Device target_device = infer_assert_device(parent, gamma);
        if (!gamma.is_dense()) {
            gamma = gamma.contiguous();
        }
        RedMeta red_meta = infer_red_meta(parent.shape(), axes, true);
        std::vector<int64_t> norm_shape = get_normalized_shape(parent.shape(), axes);
        Tensor<T> result = Tensor<T>(parent.shape(), parent.requires_grad(), lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<RMSNormFastNode<T>>(std::move(parent), std::move(gamma), std::move(red_meta), std::move(norm_shape), eps);

        return result;
    }

    template <typename T>
    Tensor<T> rmsnorm(Tensor<T> parent, Tensor<T> gamma, const std::vector<int64_t>& axes, T eps, bool cuda_fast=true) requires std::is_floating_point_v<T> {
        Device target_device = infer_assert_device(parent, gamma);
        if (cuda_fast == true && target_device.is_cuda()) {
            return rmsnorm_fast(std::move(parent), std::move(gamma), axes, eps);
        }
        else {
            return rmsnorm_naive(std::move(parent), std::move(gamma), axes, eps);
        }
    }

    
}
