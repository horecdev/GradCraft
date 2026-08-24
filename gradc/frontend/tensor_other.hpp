#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/other_nodes.hpp"

#include <memory>

namespace gradc {

    template <typename T>
    Tensor<T> Tensor<T>::dropout(T p) const requires std::is_floating_point_v<T> {
        Tensor result = Tensor(this->shape(), m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<DropoutNode<T>>(*this, p);

        return result;
    }

    template <typename T>
    Tensor<T> embed(Tensor<int64_t> indices, Tensor<T> embeds) {
        Device target_device = infer_assert_device(indices, embeds);
        if (!embeds.is_dense()) {
            throw std::runtime_error("Embeddings must be dense.");
        }

        auto [embed_shape, embed_vol] = infer_embed_shape(indices.shape(), embeds.shape());

        Tensor<T> result = Tensor<T>(embed_shape, target_device, lazy);
        result.m_state->m_creation_op = std::make_unique<EmbedNode<T>>(std::move(indices), std::move(embeds), std::move(embed_shape), embed_vol);
        // do not set requires_grad. Its false by default.

        return result;
    }
}
