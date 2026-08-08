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
                    dispatch(target_device, BinaryOp::Mul, scratchpad, out_grad, m_left); // OOP dont gaf about what was there before. Right result.
                    m_right.accumulate_grad(unbroadcast_grad(scratchpad, m_right.shape())); 
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
                    dispatch(target_device, BinaryOp::Div, scratchpad, out_grad, m_right);
                    m_left.accumulate_grad(unbroadcast_grad(scratchpad, m_left.shape()));
                }

                if (m_right.requires_grad()) {
                    if (m_left.requires_grad()) { // means that the scratchpad was just populated ----> do * (-a/b)
                        dispatch(target_device, BinaryOpInPlace::Mul, scratchpad, m_left);
                        dispatch(target_device, BinaryOpInPlace::Div, scratchpad, m_right);
                        m_right.accumulate_grad(unbroadcast_grad(scratchpad, m_right.shape()), true);
                    }
                    else { // scratchpad was allocated but is uninitialized
                        dispatch(target_device, BinaryOp::Mul, scratchpad, out_grad, m_left);
                        dispatch(target_device, BinaryOpInPlace::Div, scratchpad, m_right);
                        dispatch(target_device, BinaryOpInPlace::Div, scratchpad, m_right);
                        m_right.accumulate_grad(unbroadcast_grad(scratchpad, m_right.shape()), true);
                    }
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

    template <typename T>
    class ExpNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            ExpNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Exp, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Exp, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> exp_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BExp, exp_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(exp_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class LogNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            LogNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Log, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Log, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> log_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BLog, log_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(log_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SinNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            SinNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Sin, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Sin, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> sin_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BSin, sin_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(sin_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class CosNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            CosNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Cos, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Cos, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> cos_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BCos, cos_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(cos_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SquareNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            SquareNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                if (!m_parent.requires_grad() && m_parent.is_exclusive()) {
                    dispatch(m_parent.device(), UnaryOpInPlace::Square, m_parent);
                    return m_parent;
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Square, result, m_parent);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> square_grad = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                    dispatch(m_parent.device(), BinaryOp::BSquare, square_grad, out_grad, m_parent);
                    m_parent.accumulate_grad(square_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class MatMulNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            BLASGEMMMeta m_blas_meta;
            // left, right are already 2D and good dims for BLAS. No contig nodes will be attached.
            std::vector<int64_t> m_left_original_shape;

        public:
            MatMulNode<T>(Tensor<T> left, Tensor<T> right, BLASGEMMMeta blas_meta, std::vector<int64_t> left_original_shape) : m_left(std::move(left)), m_right(std::move(right)), m_blas_meta(std::move(blas_meta)), m_left_original_shape(std::move(left_original_shape)) {}
            
            Tensor<T> realize() override {
                Device target_device = m_left.device(); 

                m_left.realize();
                m_right.realize();

                Tensor<T> result = Tensor<T>(m_blas_meta.result_shape, target_device, uninitialized);
                dispatch_batched_gemm(target_device, result, m_left, m_right, m_blas_meta);

                return result;
            }

            void backward(const Tensor<T>& out_grad) override {

                // dL/dx = out_grad @ W.T
                // dL/dW = X.T @ out_grad
                Tensor<T> grad_flat = lobotomized_reshape_view(out_grad, {m_blas_meta.M, m_blas_meta.N});
                Tensor<T> W_T = lobotomized_transpose_view(m_right, 0, 1);
                Tensor<T> X_T = lobotomized_transpose_view(m_left, 0, 1);
                // left, right are already perfectly viable for BLAS. They will NOT be attached contiguous nodes etc. what would break everything.
                // One of their dimensions is exactly 1.
                // They are NOT guaranteed to be perfectly contiguous. Can be transposed.

                // Youre doing initially: X @ W which is (M, K) @ (K, N) where M is (B, T, ...) 
                // Now its out_grad is (M, N), You do (M, N) @ (N, K) = (M, K). Thats dL/dx
                // p_out is m_left.grad. It is contiguous, of shape (B, T, K). infer function will infer ldc to be K

                // GRAD FLAT BELOW IS NOT MODIFIED. IT GETS BASICALLY REPLICATED INSIDE ANOTHER TENSOR AND RETURNED. ITS BECAUSE ITS CONTIGUOUS AND 2D ALREADY
                std::pair<std::pair<Tensor<T>, Tensor<T>>, BLASGEMMMeta> dleft_gemm_prep = infer_blas_meta(std::move(grad_flat), std::move(W_T), true);
                BLASGEMMMeta dleft_blas_meta = dleft_gemm_prep.second;
                grad_flat = std::move(dleft_gemm_prep.first.first);
                W_T = std::move(dleft_gemm_prep.first.second);

                // For dL/dW you do (K, M) @ (M, N) = (K, N)

                // HERE GRAD_FLAT IS ALSO NOT MODIFIED. 
                std::pair<std::pair<Tensor<T>, Tensor<T>>, BLASGEMMMeta> dright_gemm_prep = infer_blas_meta(std::move(X_T), std::move(grad_flat), true);
                BLASGEMMMeta dright_blas_meta = dright_gemm_prep.second;
                X_T = std::move(dright_gemm_prep.first.first);
                grad_flat = std::move(dright_gemm_prep.first.second);

                m_left.accumulate_grad_matmul(grad_flat, W_T, dleft_blas_meta, m_left_original_shape);
                m_right.accumulate_grad_matmul(X_T, grad_flat, dright_blas_meta, m_right.shape());
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };
} 