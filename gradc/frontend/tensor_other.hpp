#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/other_nodes.hpp"

#include <memory>

namespace gradc {

    template <typename T>
    Tensor<T> Tensor<T>::dropout(T p) const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(this->shape(), m_requires_grad, lazy, this->device());
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

        Tensor<T> result = Tensor<T>(embed_shape, embeds.requires_grad(), lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<EmbedNode<T>>(std::move(indices), std::move(embeds), std::move(embed_shape), embed_vol);

        return result;
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    Tensor<T> causal_softmax(Tensor<T> scores, T scale) {
        Device target_device = scores.device();

        if (!scores.is_dense()) {
            throw std::runtime_error("Causal Softmax scores must be dense.");
        }
        if (std::ssize(scores.shape()) != 4) {
            throw std::runtime_error("Causal Softmax scores must be 4D.");
        }

        Tensor<T> result = Tensor<T>(scores.shape(), scores.requires_grad(), lazy, target_device);
        result._get_state()->m_creation_op = std::make_unique<CausalSoftmaxNode<T>>(std::move(scores), scale);

        return result;
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    Tensor<T> sdpa(
        Tensor<T> q, Tensor<T> k, Tensor<T> v, 
        bool is_causal = false, 
        std::optional<Tensor<T>> custom_mask = std::nullopt, 
        std::optional<T> scale = std::nullopt, 
        bool cuda_fast = true
    ) {
        Device target_device = infer_assert_device(q, k, v);
        
        if (std::ssize(q.shape()) != 4 || std::ssize(k.shape()) != 4 || std::ssize(v.shape()) != 4) {
            throw std::runtime_error("SDPA expects 4D tensors.");
        }
        
        int64_t head_dim = q.shape()[3];
        T actual_scale = scale.value_or(static_cast<T>(1.0 / std::sqrt(head_dim)));
        
        Tensor<T> k_T = k.transpose(2, 3);
        Tensor<T> scores = bmm(q, k_T);
        
        Tensor<T> probs;
        
        if (cuda_fast && is_causal && target_device.is_cuda()) {
            probs = causal_softmax(scores, actual_scale);
        } 
        else {
            scores *= actual_scale;
            
            if (is_causal) {
                if (custom_mask.has_value()) {
                    scores += custom_mask.value();
                } 
                else {
                    throw std::runtime_error("Must pass causal_mask into SDPA when is_causal and not using CUDA causal softmax.");
                }
            } 
            
            probs = scores.softmax(3);
        }
        
        return bmm(probs, v);
    }
}
