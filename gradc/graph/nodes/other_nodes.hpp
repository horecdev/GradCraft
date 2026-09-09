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
    class CausalSoftmaxNode : public Node<T> {
        private:
            Tensor<T> m_scores; // [B, num_heads, T, T] dense
            Tensor<T> m_probs;
            T m_scale;
            int64_t m_seq_len;
        public:
            CausalSoftmaxNode(Tensor<T> scores, T scale) : m_scores(std::move(scores)), m_scale(scale), m_seq_len(m_scores.shape()[2]) {}

            Tensor<T> realize() override {
                m_scores.realize();

                Device target_device = m_scores.device();

                Tensor<T> result = Tensor<T>(m_scores.shape(), target_device, uninitialized);

                dispatch_causal_softmax_forward(target_device, result, m_scores, m_scale, m_seq_len);

                if (m_scores.requires_grad()) {
                    m_probs = Tensor<T>(m_scores.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Identity, m_probs, result);
                }

                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                if (m_scores.requires_grad()) {
                    Device target_device = out_grad.device();
                    
                    Tensor<T> dscores;
                    bool acc_dx = false;
                    
                    if (m_scores.grad().has_value()) {
                        dscores = m_scores.grad().value();
                        acc_dx = true;
                    } 
                    else {
                        dscores = Tensor<T>(m_scores.shape(), target_device, uninitialized);
                    }
                    
                    dispatch_causal_softmax_backward(target_device, dscores, out_grad, m_probs, m_scale, m_seq_len, acc_dx);
                    
                    if (!acc_dx) {
                        m_scores._get_state()->m_grad = std::move(dscores);
                    }
                }
                if (!retain_graph) {
                    m_probs = Tensor<T>();
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_scores._get_state_base()};
            }
    };

    template <typename T>
    class OneHotNode : public Node<T> {
        private:
            Tensor<int64_t> m_indices;
            std::vector<int64_t> m_result_shape;

        public:
            OneHotNode(Tensor<int64_t> indices, std::vector<int64_t> result_shape) : m_indices(std::move(indices)), m_result_shape(std::move(result_shape)) {}

            Tensor<T> realize() override {
                m_indices.realize();

                Device target_device = m_indices.device();

                Tensor<T> result = Tensor<T>(m_result_shape, T(0), target_device);

                dispatch_one_hot_encode(target_device, result, m_indices);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_indices.requires_grad()) {
                    throw std::runtime_error("Indices in OneHotNode backward cannot require grad.");
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_indices._get_state_base()};
            }
    };

    template <typename T>
    class SwiGLUFastNode : public Node<T> {
        private:
            Tensor<T> m_w1_out;
            Tensor<T> m_w2_out;

        public:
            SwiGLUFastNode(Tensor<T> w1_out, Tensor<T> w2_out) : m_w1_out(std::move(w1_out)), m_w2_out(std::move(w2_out)) {}

        Tensor<T> realize() override {
            m_w1_out.realize();
            m_w2_out.realize();

            Device target_device = m_w1_out.device();

            Tensor<T> result = Tensor<T>(m_w1_out.shape(), target_device, uninitialized);

            dispatch_swiglu_fast_forward(target_device, result, m_w1_out, m_w2_out);

            return result;
        }

        void backward(Tensor<T> out_grad, [[maybe_unused]] bool retain_graph) override {
            if (!m_w1_out.requires_grad() && !m_w2_out.requires_grad()){return;}

            // essentially: you check if grad exists. If it does, you accumulate and pass down the grad. If not, you create it, dont accumulate, and then move it in as the grad.
            Device target_device = out_grad.device();
            Tensor<T> da;
            bool acc_a = false;
            if (m_w1_out.requires_grad()) {
                if (m_w1_out.grad().has_value()) {
                    da = m_w1_out.grad().value();
                    acc_a = true;
                }
                else {
                    da = Tensor<T>(m_w1_out.shape(), target_device, uninitialized);
                }
            }

            Tensor<T> db;
            bool acc_b = false;
            if (m_w2_out.requires_grad()) {
                if (m_w2_out.grad().has_value()) {
                    db = m_w2_out.grad().value();
                    acc_b = true;
                } 
                else {
                    db = Tensor<T>(m_w2_out.shape(), target_device, uninitialized);
                }
            }

            dispatch_swiglu_fast_backward(target_device, da, db, out_grad, m_w1_out, m_w2_out, acc_a, acc_b);

            // check if the tensor was freshly created
            if (m_w1_out.requires_grad() && !acc_a) {
                m_w1_out._get_state()->m_grad = std::move(da);
            }
            if (m_w2_out.requires_grad() && !acc_b) {
                m_w2_out._get_state()->m_grad = std::move(db);
            }
        }

        std::vector<TensorStateBase*> get_input_states() override {
            return {m_w1_out._get_state_base(), m_w2_out._get_state_base()};
        }
    };
}