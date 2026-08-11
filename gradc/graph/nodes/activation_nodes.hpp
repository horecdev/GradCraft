#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

namespace gradc {

    template <typename T>
    class ReLUNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            ReLUNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(target_device, UnaryOpInPlace::ReLU, m_parent);
                    return m_parent;
                }
                
                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, UnaryOp::ReLU, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> relu_grad = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::BReLU, relu_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(relu_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SigmoidNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            SigmoidNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(target_device, UnaryOpInPlace::Sigmoid, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, UnaryOp::Sigmoid, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> sig_grad = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::BSigmoid, sig_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(sig_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class TanHNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            TanHNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(target_device, UnaryOpInPlace::TanH, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, UnaryOp::TanH, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> tanh_grad = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::BTanH, tanh_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(tanh_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SiLUNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            SiLUNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(target_device, UnaryOpInPlace::SiLU, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, UnaryOp::SiLU, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> silu_grad = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::BSiLU, silu_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(silu_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class GeLUNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            GeLUNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(target_device, UnaryOpInPlace::GeLU, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, UnaryOp::GeLU, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();
                    
                    Tensor<T> gelu_grad = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::BGeLU, gelu_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(gelu_grad);
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
            RedMeta m_red_meta;
            Tensor<T> m_probs;
        public:
            SoftmaxNode(Tensor<T> logits, RedMeta red_meta) : m_logits(std::move(logits)), m_red_meta(std::move(red_meta)) {}

            Tensor<T> realize() override {
                m_logits.realize();

                Device target_device = m_logits.device();

                Tensor<T> probs;
                if (m_logits.is_exclusive()) {
                    probs = m_logits;
                }
                else {
                    probs = Tensor<T>(m_logits.shape(), target_device, uninitialized);
                }

                Tensor<T> max_logits = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Max, m_red_meta, max_logits, m_logits);
                dispatch(target_device, BinaryOp::Sub, probs, m_logits, max_logits);
                // probs is now X - max

                dispatch(target_device, UnaryOpInPlace::Exp, probs);
                // (X - max).exp()
                
                Tensor<T>& logits_sum = max_logits;
                dispatch(target_device, ReduceOp::Sum, m_red_meta, logits_sum, probs);
                dispatch(target_device, BinaryOpInPlace::Div, probs, logits_sum);
                // probs is now true probs

                if (m_logits.requires_grad()) {
                    m_probs = Tensor<T>(m_logits.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Identity, m_probs, probs);
                }
                
                return probs;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) {
                if (m_logits.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> y_mul_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, y_mul_grad, out_grad, m_probs);

                    Tensor<T> sum_y_mul_grad = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized);
                    dispatch(target_device, ReduceOp::Sum, m_red_meta, sum_y_mul_grad, y_mul_grad);
                    dispatch(target_device, BinaryOp::Sub, y_mul_grad, out_grad, sum_y_mul_grad);
                    dispatch(target_device, BinaryOpInPlace::Mul, y_mul_grad, m_probs);

                    m_logits.accumulate_grad(y_mul_grad);

                    if (!retain_graph) {m_probs = Tensor<T>();}
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_logits._get_state_base()};
            }
    };

    // REGULATIZATION (NOT ACTIVATION THO)

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
    };
    

}