#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

namespace gradc {
    template <typename T>
    class DropoutNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            T m_p;
            Tensor<T> m_mask;
        public:
            DropoutNode(Tensor<T> parent, T p) : m_parent(std::move(parent)), m_p(p) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                // y = x * M / (1-p)

                Tensor<T> mask = Tensor<T>::uniform(m_parent.shape(), static_cast<T>(0.0), static_cast<T>(1.0), target_device);
                dispatch(target_device, BinaryOpInPlace::GrThan, mask, Tensor<T>(m_p, target_device));
                dispatch(target_device, BinaryOpInPlace::Mul, mask, Tensor<T>(static_cast<T>(1.0) / (static_cast<T>(1.0) - m_p), target_device));

                if (m_parent.requires_grad()) {
                    m_mask = mask; // must be in the middle cuz we got early return
                }

                if (m_parent.is_exclusive()) {
                    // can mutate parent
                    dispatch(target_device, BinaryOpInPlace::Mul, m_parent, mask);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, BinaryOp::Mul, result, m_parent, mask);
                
                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();
                    // dL/dx = out_grad * M / (1-p)
                    Tensor<T> dropout_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, dropout_grad, out_grad, m_mask);

                    m_parent.accumulate_grad(dropout_grad);
                }

                if (!retain_graph) {
                    m_mask = Tensor<T>();
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class EmbedNode : public Node<T> {
        private:
            Tensor<int64_t> m_indices;
            Tensor<T> m_embeds;
            std::vector<int64_t> m_result_shape;
            int64_t m_embed_vol;
        public:
            EmbedNode(Tensor<int64_t> indices, Tensor<T> embeds, std::vector<int64_t> result_shape, int64_t embed_vol) : m_indices(std::move(indices)), m_embeds(std::move(embeds)), m_result_shape(std::move(result_shape)), m_embed_vol(embed_vol) {}

            Tensor<T> realize() override {
                m_indices.realize();
                m_embeds.realize();
                
                Device target_device = m_indices.device();
                Tensor<T> result = Tensor<T>(m_result_shape, target_device, uninitialized);

                dispatch_embed(target_device, result, m_indices, m_embeds, m_embed_vol);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) {
                if (m_embeds.requires_grad()) {
                    m_embeds.accumulate_grad_embeds(m_indices, out_grad, m_embed_vol);
                }
                if (m_indices.requires_grad()) {
                    throw std::runtime_error("Indices in EmbedNode cannot require grad.");
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_embeds._get_state_base(), m_indices._get_state_base()};
            }
    };

    template <typename T>
    class SDPACuDNNNode : public Node<T> {
        private:
            Tensor<T> m_q;
            Tensor<T> m_k;
            Tensor<T> m_v;
            T m_scale;
            bool m_is_causal;
            
            Tensor<T> m_saved_out;
            Tensor<T> m_saved_lse;
        
        public:
            SDPACuDNNNode(Tensor<T> q, Tensor<T> k, Tensor<T> v, T scale, bool is_causal) 
                : m_q(std::move(q)), m_k(std::move(k)), m_v(std::move(v)), m_scale(scale), m_is_causal(is_causal) {}

            Tensor<T> realize() override {
                m_q.realize();
                m_k.realize();
                m_v.realize();

                Device target_device = m_q.device();

                Tensor<T> result = Tensor<T>(m_q.shape(), target_device, uninitialized);

                bool save_intermediates = false;
                if (m_q.requires_grad() || m_k.requires_grad() || m_v.requires_grad()) {
                    std::vector<int64_t> lse_shape = m_q.shape();
                    lse_shape.back() = 1; // softmax over head dim

                    m_saved_out = Tensor<T>(m_q.shape(), target_device, uninitialized);
                    m_saved_lse = Tensor<T>(lse_shape, target_device, uninitialized);
                    save_intermediates = true;
                }

                dispatch_cudnn_sdpa_forward(target_device, result, m_saved_lse, m_q, m_k, m_v, m_scale, m_is_causal, save_intermediates);
                dispatch(target_device, UnaryOp::Identity, m_saved_out, result);

                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                if (m_q.requires_grad() || m_k.requires_grad() || m_v.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> dq = Tensor<T>(m_q.shape(), target_device, uninitialized);
                    Tensor<T> dk = Tensor<T>(m_k.shape(), target_device, uninitialized);
                    Tensor<T> dv = Tensor<T>(m_v.shape(), target_device, uninitialized);
                    
                    dispatch_cudnn_sdpa_backward(target_device, dq, dk, dv, out_grad, m_q, m_k, m_v, m_saved_out, m_saved_lse, m_scale, m_is_causal);

                    if (m_q.requires_grad()) {
                        m_q.accumulate_grad(dq);
                    }
                    if (m_k.requires_grad()) {
                        m_k.accumulate_grad(dk);
                    }
                    if (m_v.requires_grad()) {
                        m_v.accumulate_grad(dv);
                    }

                    if (!retain_graph) {
                        m_saved_out = Tensor<T>();
                        m_saved_lse = Tensor<T>();
                    }
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_q._get_state_base(), m_k._get_state_base(), m_v._get_state_base()};
            }
    };
}