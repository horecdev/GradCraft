#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/activation_nodes.hpp"

#include <memory>

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::embed(Tensor<int64_t> indices, Tensor<T> embeddings) {
        if (!embeddings.is_dense()) {
            throw std::runtime_error("Embeddings must be dense.");
        }
    }
}
