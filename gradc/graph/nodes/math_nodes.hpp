#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/detail/tensor_detail.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T>
    class NegNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
        public:
            NegNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();

                if (m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Neg, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Neg, result, m_parent);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    m_parent.accumulate_grad(out_grad, true);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    }; 

    template <typename T>
    class AddNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<int64_t> m_target_shape;
        public:
            AddNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();

                Device target_device = m_left.device();

                if (m_left.is_exclusive() && m_left.shape() == m_target_shape) { // inside addnode m_left storage is used solely for producing a result. 
                    // If nobody else uses the m_left EVER, then instead of using new memory, can just edit it and return.
                    dispatch(target_device, BinaryOpInPlace::Add, m_left, m_right);
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape) {
                    dispatch(target_device, BinaryOpInPlace::Add, m_right, m_left);
                    return m_right;
                }

                Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Add, result, m_left, m_right);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override { // CPU child can only have CPU parents (enforced outside of graph nodes)
                if (m_left.requires_grad()) {
                    m_left.accumulate_grad(unbroadcast_grad(out_grad, m_left.shape()));
                }
                if (m_right.requires_grad()) {
                    m_right.accumulate_grad(unbroadcast_grad(out_grad, m_right.shape()));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

    template <typename T>
    class SubNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<int64_t> m_target_shape;
        public:
            SubNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();

                Device target_device = m_left.device();

                if (m_left.is_exclusive() && m_left.shape() == m_target_shape) {
                    dispatch(target_device, BinaryOpInPlace::Sub, m_left, m_right);
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape) {
                    dispatch(target_device, BinaryOpInPlace::ISub, m_right, m_left);
                    return m_right;
                }

                Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, result, m_left, m_right);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_left.requires_grad()) {
                    m_left.accumulate_grad(unbroadcast_grad(out_grad, m_left.shape()));
                }
                if (m_right.requires_grad()) {
                    m_right.accumulate_grad(unbroadcast_grad(out_grad, m_right.shape()), true);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

    template <typename T>
    class MulNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<int64_t> m_target_shape;
        public:
            MulNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();

                Device target_device = m_left.device();

                if (m_left.is_exclusive() &&  m_left.shape() == m_target_shape && m_right.requires_grad() == false) {
                    dispatch(target_device, BinaryOpInPlace::Mul, m_left, m_right);
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape && m_left.requires_grad() == false) {
                    dispatch(target_device, BinaryOpInPlace::Mul, m_right, m_left);
                    return m_right;
                }

                Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Mul, result, m_left, m_right);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                Device target_device = m_left.device();

                Tensor<T> scratchpad;
                if (m_left.requires_grad() || m_right.requires_grad()) {
                    scratchpad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                }

                if (m_left.requires_grad()) {
                    dispatch(target_device, BinaryOp::Mul, scratchpad, out_grad, m_right);
                    m_left.accumulate_grad(unbroadcast_grad(scratchpad, m_left.shape())); 
                }
                if (m_right.requires_grad()) {
                    dispatch(target_device, BinaryOp::Mul, scratchpad, out_grad, m_left);
                    m_right.accumulate_grad(unbroadcast_grad(scratchpad, m_right.shape())); // OOP dont gaf about what was there before
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

    template <typename T>
    class DivNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<int64_t> m_target_shape;
        public:
            DivNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();

                Device target_device = m_left.device();

                if (m_left.is_exclusive() && m_left.shape() == m_target_shape && m_right.requires_grad() == false) {
                    dispatch(target_device, BinaryOpInPlace::Div, m_left, m_right);
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape && m_left.requires_grad() == false && m_right.requires_grad() == false) {
                    dispatch(target_device, BinaryOpInPlace::IDiv, m_right, m_left);
                    return m_right;
                }

                Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Div, result, m_left, m_right);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                Device target_device = m_left.device();

                Tensor<T> scratchpad;
                if (m_left.requires_grad() || m_right.requires_grad()) {
                    scratchpad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                }

                if (m_left.requires_grad()) {
                    Tensor<T> raw_left_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, raw_left_grad, out_grad, m_right);
                    m_left.accumulate_grad(unbroadcast_grad(raw_left_grad, m_left.shape()));
                }
                if (m_right.requires_grad()) {
                    Tensor<T> raw_right_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, raw_right_grad, out_grad, m_left);
                    m_right.accumulate_grad(unbroadcast_grad(raw_right_grad, m_right.shape()));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };
} 