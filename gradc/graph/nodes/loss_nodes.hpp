#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

namespace gradc {

    template <typename T>
    class MSELossNode : public Node<T> {
        
    };
    
    template <typename T>
    class SoftmaxCrossEntropyLossNode : public Node<T> {
        private:
            Tensor<T> m_flat_logits; // logits are (B, C) flattened
            Tensor<T> m_flat_targets; // also (B, C)
            ReductionMetadata m_softmax_red_meta;
            ReductionMetadata m_loss_red_meta;
            int64_t m_batch_size;
            Tensor<T> m_probs;
            T m_eps;
            
        public:
            SoftmaxCrossEntropyLossNode(Tensor<T> flat_logits, Tensor<T> flat_targets, ReductionMetadata softmax_red_meta, ReductionMetadata loss_red_meta, int64_t batch_size, T eps)
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

                if (m_flat_logits.requires_grad()) {
                    m_probs = probs; // probs (or m_flat_logits alias) is not ever edited here or anywhere else
                }

                if (!m_flat_logits.requires_grad()) {m_flat_logits = Tensor<T>(); m_flat_targets = Tensor<T>();}

                return loss;

            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                if (m_flat_logits.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> dx = Tensor<T>(m_flat_logits.shape(), out_grad.device(), uninitialized);
                    dispatch(target_device, BinaryOp::Sub, dx, m_probs, m_flat_targets);
                    dispatch(target_device, BinaryOpInPlace::Mul, dx, out_grad);
                    // you averaged batch_size terms into the loss
                    dispatch(target_device, BinaryOpInPlace::Div, dx, Tensor<T>(static_cast<T>(m_batch_size), target_device));

                    m_flat_logits.accumulate_grad(dx);

                    if (!retain_graph) {m_probs = Tensor<T>(); m_flat_logits = Tensor<T>(); m_flat_targets = Tensor<T>();}
                    
                }
            }
    };
}