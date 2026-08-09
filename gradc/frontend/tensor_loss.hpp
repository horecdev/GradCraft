#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/loss_nodes.hpp"

#include <memory>

namespace gradc {

    template <typename T>
    Tensor<T> mseloss(Tensor<T> parent, Tensor<T> gamma, Tensor<T> beta, const std::vector<int64_t>& axes, T eps = static_cast<T>(1e-5)) requires std::is_floating_point_v<T> {
        Device target_device = infer_assert_device(parent, gamma, beta);
        ReductionMetadata reduction_metadata = infer_reduction_metadata(parent.shape(), axes, true);
        Tensor result = Tensor(parent.shape(), parent.requires_grad(), lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<MSELossNode<T>>(std::move(parent), std::move(gamma), std::move(beta), std::move(reduction_metadata), eps);

        return result;
    }
}
