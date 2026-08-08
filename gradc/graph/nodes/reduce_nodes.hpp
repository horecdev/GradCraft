#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../../core/types.hpp"
#include "../node.hpp"

#include <vector>

namespace gradc {

    template <typename T>
    class SumNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
        public:
            SumNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Tensor<T> result = Tensor<T>(m_reduction_metadata.result_shape, m_parent.device(), uninitialized);
                dispatch(m_parent.device(), ReduceOp::Sum, m_reduction_metadata, result, m_parent);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    m_parent.accumulate_grad(Tensor<T>(m_parent.shape(), m_reduction_metadata.temp_strides, 0, out_grad._get_storage(), false));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class MeanNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
        public:
            MeanNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Tensor<T> summed = Tensor<T>(m_reduction_metadata.result_shape, m_parent.device(), uninitialized);
                dispatch(m_parent.device(), ReduceOp::Sum, m_reduction_metadata, summed, m_parent);
                dispatch(m_parent.device(), BinaryOpInPlace::Div, summed, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol), m_parent.device()));
                return summed;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> divided_grad = Tensor<T>(out_grad.shape(), out_grad.device(), uninitialized);
                    dispatch(out_grad.device(), BinaryOp::Div, divided_grad, out_grad, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol)));
                    Tensor<T> strided_mean_grad = Tensor<T>(m_parent.shape(), m_reduction_metadata.temp_strides, 0, divided_grad._get_storage(), false);
                    m_parent.accumulate_grad(strided_mean_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class MaxNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
            Tensor<T> m_result;
        public:
            MaxNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                Tensor<T> result = Tensor<T>(m_reduction_metadata.result_shape, m_parent.device(), uninitialized);
                dispatch(m_parent.device(), ReduceOp::Max, m_reduction_metadata, result, m_parent);

                if (m_parent.requires_grad()) {
                    m_result = Tensor<T>(m_reduction_metadata.result_shape, m_parent.device(), uninitialized); // create a deep copy of result
                    dispatch(m_parent.device(), UnaryOp::Identity, m_result, result);
                }
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> reshaped_result = lobotomized_reshape_view(m_result, m_reduction_metadata.temp_shape);
                    Tensor<T> broadcast_result = lobotomized_broadcast_view(reshaped_result, m_parent.shape());

                    Tensor<T> reshaped_grad = lobotomized_reshape_view(out_grad, m_reduction_metadata.temp_shape);
                    Tensor<T> broadcast_grad = lobotomized_broadcast_view(out_grad, m_parent.shape());
                    
                    Tensor<T> mask = Tensor<T>(m_parent.shape(), out_grad.device(), uninitialized);
                    dispatch(out_grad.device(), BinaryOp::EqMask, mask, m_parent, broadcast_result);

                    Tensor<T> grad_input = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::Mul, grad_input, mask, broadcast_grad);
                    m_parent.accumulate_grad(grad_input);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SoftmaxNode : public Node<T> {
        private:
            Tensor<T> m_logits;
            ReductionMetadata m_reduction_metadata;
            Tensor<T> m_result;
        public:
            SoftmaxNode(Tensor<T> logits, ReductionMetadata reduction_metadata) : m_logits(std::move(logits)), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() override {
                m_logits.realize();

                if (m_logits.is_exclusive()) {
                    // using temp_shape is keeping the collapsed dim as 1 so it can broadcast.
                    Tensor<T> max_logits = Tensor<T>(m_reduction_metadata.temp_shape, m_logits.device(), uninitialized);
                    dispatch(m_logits.device(), ReduceOp::Max, m_reduction_metadata, max_logits, m_logits);
                    
                    dispatch(m_logits.device(), BinaryOpInPlace::Sub, m_logits, max_logits);
                    // now m_logits are X - MAX
                    dispatch(m_logits.device(), UnaryOpInPlace::Exp, m_logits);
                    // exponentiated
                    
                    Tensor<T>& logits_sum = max_logits;
                    dispatch(m_logits.device(), ReduceOp::Sum, m_reduction_metadata, logits_sum, m_logits);
                    dispatch(m_logits.device(), BinaryOpInPlace::Div, m_logits, logits_sum);

                    if (m_logits.requires_grad()) {
                        m_result = Tensor<T>(m_logits.shape(), m_logits.device(), uninitialized);
                        dispatch(m_logits.device(), UnaryOp::Identity, m_result, m_logits);
                    }

                    return m_logits;
                }

                Tensor<T> max_logits = Tensor<T>(m_reduction_metadata.temp_shape, m_logits.device(), uninitialized);
                dispatch(m_logits.device(), ReduceOp::Max, m_reduction_metadata, max_logits, m_logits);
                Tensor<T> normalized_logits = Tensor<T>(m_logits.shape(), m_logits.device(), uninitialized);
                dispatch(m_logits.device(), BinaryOp::Sub, normalized_logits, m_logits, max_logits);

                dispatch(m_logits.device(), UnaryOpInPlace::Exp, normalized_logits);
                
                Tensor<T>& logits_sum = max_logits;
                dispatch(m_logits.device(), ReduceOp::Sum, m_reduction_metadata, logits_sum, normalized_logits);
                dispatch(m_logits.device(), BinaryOpInPlace::Div, normalized_logits, logits_sum);

                if (m_logits.requires_grad()) {
                    m_result = Tensor<T>(m_logits.shape(), m_logits.device(), uninitialized);
                    dispatch(m_logits.device(), UnaryOp::Identity, m_result, normalized_logits);
                }
                
                return normalized_logits;
            }
    };
}