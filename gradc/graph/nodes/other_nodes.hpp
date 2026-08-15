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
    };
}