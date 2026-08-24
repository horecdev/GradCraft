#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/normalization_nodes.hpp"
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
    Tensor<T> Tensor<T>::embed(Tensor<int64_t> indices, Tensor<T> embeddings) {
        Device target_device = infer_assert_device(indices, embeddings);
        if (!embeddings.is_dense()) {
            throw std::runtime_error("Embeddings must be dense.");
        }

        std::pair<std::vector<int64_t>, int64_t> embed_data = infer_embedding_shape(indices.shape(), embeddings.shape());

        Tensor<T> result = Tensor<T>(std::move(embed_data.first), target_device, lazy);
        result.m_state->m_creation_op = std::make_unique<EmbedNode<T>>(std::move(indices), std::move(embeddings), embed_data.second);

        return result;
    }
}
