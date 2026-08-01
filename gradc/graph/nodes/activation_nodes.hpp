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
                    dispatch(m_parent.device(), BinaryOp::ReLUBackward, relu_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(relu_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

}