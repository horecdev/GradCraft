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
            ReLUNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::ReLU, m_parent);
                    return m_parent;
                }
                
                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::ReLU, result, m_parent);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> relu_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BReLU, relu_grad, out_grad, m_parent);
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
            SigmoidNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Sigmoid, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Sigmoid, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> sig_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BSigmoid, sig_grad, out_grad, m_parent);
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
            TanHNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::TanH, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::TanH, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> tanh_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BTanH, tanh_grad, out_grad, m_parent);
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
            SiLUNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::SiLU, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::SiLU, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> silu_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BSiLU, silu_grad, out_grad, m_parent);
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
            GeLUNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::GeLU, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::GeLU, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> gelu_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BGeLU, gelu_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(gelu_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

}