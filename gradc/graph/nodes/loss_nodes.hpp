#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

namespace gradc {
    
    template <typename T>
    class SoftmaxCrossEntropyLossNode : public Node<T> {
        private:
            Tensor<T> m_flat_logits; // logits are (B, C) flattened
            Tensor<T> m_flat_targets; // also (B, C)
            RedMeta m_softmax_red_meta;
            RedMeta m_loss_red_meta;
            int64_t m_batch_size;
            Tensor<T> m_probs;
            T m_eps;
            
        public:
            SoftmaxCrossEntropyLossNode(Tensor<T> flat_logits, Tensor<T> flat_targets, RedMeta softmax_red_meta, RedMeta loss_red_meta, int64_t batch_size, T eps)
             : m_flat_logits(std::move(flat_logits)), m_flat_targets(std::move(flat_targets)), m_softmax_red_meta(std::move(softmax_red_meta)), m_loss_red_meta(std::move(loss_red_meta)), m_batch_size(batch_size), m_eps(eps) {}

            Tensor<T> realize() override {
                m_flat_logits.realize();
                m_flat_targets.realize();

                Device target_device = m_flat_logits.device();

                Tensor<T> probs;
                if (m_flat_logits.is_exclusive()) {
                    probs = m_flat_logits;
                }
                else {
                    probs = Tensor<T>(m_flat_logits.shape(), target_device, uninitialized);
                }

                Tensor<T> max_logits = Tensor<T>(m_softmax_red_meta.temp_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Max, m_softmax_red_meta, max_logits, m_flat_logits);
                dispatch(target_device, BinaryOp::Sub, probs, m_flat_logits, max_logits);
                dispatch(target_device, UnaryOpInPlace::Exp, probs);
                Tensor<T>& logits_sum = max_logits;
                dispatch(target_device, ReduceOp::Sum, m_softmax_red_meta, logits_sum, probs);
                dispatch(target_device, BinaryOpInPlace::Div, probs, logits_sum);
                
                Tensor<T> log_probs = Tensor<T>(m_flat_logits.shape(), target_device, uninitialized);
                dispatch(target_device, BinaryOp::Add, log_probs, probs, Tensor<T>(m_eps, target_device));
                dispatch(target_device, UnaryOpInPlace::Log, log_probs);
                Tensor<T>& loss_elems = log_probs;

                dispatch(target_device, BinaryOpInPlace::Mul, log_probs, m_flat_targets);
                Tensor<T> loss = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Sum, m_loss_red_meta, loss, loss_elems);
                dispatch(target_device, BinaryOpInPlace::Div, loss, Tensor<T>(-static_cast<T>(m_batch_size), target_device));

                if (m_flat_logits.requires_grad() || m_flat_targets.requires_grad()) {
                    m_probs = probs; // probs (or m_flat_logits alias) is not ever edited here or anywhere else
                }

                return loss;

            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                Device target_device = out_grad.device();

                Tensor<T> scratchpad;
                if (m_flat_logits.requires_grad() || m_flat_targets.requires_grad()) {
                    scratchpad = Tensor<T>(m_flat_logits.shape(), target_device, uninitialized);
                }

                if (m_flat_logits.requires_grad()) {
                    Tensor<T>& dx = scratchpad;
                    dispatch(target_device, BinaryOp::Sub, dx, m_probs, m_flat_targets);
                    dispatch(target_device, BinaryOpInPlace::Mul, dx, out_grad);
                    // you averaged batch_size terms into the loss
                    dispatch(target_device, BinaryOpInPlace::Div, dx, Tensor<T>(static_cast<T>(m_batch_size), target_device));

                    m_flat_logits.accumulate_grad(dx);
                }

                if (m_flat_targets.requires_grad()) {

                    Tensor<T>& dtargets = scratchpad;
                    dispatch(target_device, BinaryOp::Add, dtargets, m_probs, Tensor<T>(m_eps, target_device));
                    dispatch(target_device, UnaryOpInPlace::Log, dtargets);

                    Tensor<T> scale = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, scale, out_grad, Tensor<T>(static_cast<T>(-1.0 / m_batch_size), target_device));

                    dispatch(target_device, BinaryOpInPlace::Mul, dtargets, scale);
    
                    m_flat_targets.accumulate_grad(dtargets); // -1/B * log(probs + eps) * out_grad
                }

                if (!retain_graph) {
                    m_probs = Tensor<T>();
                }
                // if probs is an alias of m_flat_logits: alias is deleted, but not wiped (cuz m_flat_logits is second alias)
                // also m_flat_logits has edited data if its an alias
                // otherwise its wiped
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_flat_logits._get_state_base(), m_flat_targets._get_state_base()};
            }
    };

    template <typename T>
    class MSELossNode : public Node<T> {
        private:
            Tensor<T> m_preds;
            Tensor<T> m_targets;
            RedMeta m_mse_red_meta;
        public:
            MSELossNode(Tensor<T> preds, Tensor<T> targets, RedMeta mse_red_meta) : m_preds(std::move(preds)), m_targets(std::move(targets)), m_mse_red_meta(std::move(mse_red_meta)) {}

            Tensor<T> realize() override {
                m_preds.realize();
                m_targets.realize();

                Device target_device = m_preds.device();

                Tensor<T> x_minus_y = Tensor<T>(m_preds.shape(), target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, x_minus_y, m_preds, m_targets);
                dispatch(target_device, UnaryOpInPlace::Square, x_minus_y);
                Tensor<T> loss = Tensor<T>(m_mse_red_meta.result_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Sum, m_mse_red_meta, loss, x_minus_y);
                dispatch(target_device, BinaryOpInPlace::Div, loss, Tensor<T>(static_cast<T>(m_mse_red_meta.reduced_vol), target_device));

                return loss;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) {
                Device target_device = out_grad.device();

                if (m_preds.requires_grad() || m_targets.requires_grad()) {
                    T factor = static_cast<T>(2.0) / static_cast<T>(m_mse_red_meta.reduced_vol);
                    Tensor<T> dx = Tensor<T>(m_preds.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Sub, dx, m_preds, m_targets);
                    dispatch(target_device, BinaryOpInPlace::Mul, dx, out_grad);
                    dispatch(target_device, BinaryOpInPlace::Mul, dx, Tensor<T>(factor, target_device));

                    if (m_preds.requires_grad()) {
                        m_preds.accumulate_grad(dx, false);
                    }
                    if (m_targets.requires_grad()) {
                        m_targets.accumulate_grad(dx, true);
                    }
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_preds._get_state_base(), m_targets._get_state_base()};
            }
    };
}