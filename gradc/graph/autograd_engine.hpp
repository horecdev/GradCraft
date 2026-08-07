#pragma once

#include "../backend/dispatcher.hpp"
#include "../core/tensor.hpp"
#include "../core/tensor_state.hpp"
#include "node.hpp"

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gradc {
    // If Y1, Y2 depend on X:
    // Case 1: Reaches Y1 before it has ever seen X - first adds X, then Y1
    // Case 2: Reaches Y2 after it has seen X - Y2 is added AFTER X (cuz X was added before). Still holds. Order: X, Y1, Y2
    class AutogradEngine {
        public:
            static void visit(TensorStateBase* current, std::unordered_set<TensorStateBase*>& visited, std::vector<TensorStateBase*>& topo_order) {
                if (visited.contains(current)) {
                    return;
                }

                visited.insert(current);

                for (TensorStateBase* new_root : current->get_dependencies()) {
                    visit(new_root, visited, topo_order);
                }

                topo_order.push_back(current);
            } 

            static std::vector<TensorStateBase*> build_topo(TensorStateBase* root) {
                std::unordered_set<TensorStateBase*> visited;
                std::vector<TensorStateBase*> topo_order;
                
                visit(root, visited, topo_order);

                std::reverse(topo_order.begin(), topo_order.end());
                return topo_order;
            }
    };

    
    template <typename T>
    void Tensor<T>::accumulate_grad(const Tensor<T>& incoming_grad, bool is_sub) {
        Device target_device = infer_assert_device(*this, incoming_grad); // guard if a shitshow happened (but probably redundant)

        if (!m_requires_grad) {return;}

        if (!m_state->m_grad.has_value()) {
            UnaryOp op = !is_sub ? UnaryOp::Identity : UnaryOp::Neg;
            Tensor<T> local_grad = Tensor<T>(m_shape, target_device, uninitialized);
            dispatch(target_device, op, local_grad, incoming_grad);
            m_state->m_grad = std::move(local_grad);
        }
        else {
            BinaryOpInPlace op = !is_sub ? BinaryOpInPlace::Add : BinaryOpInPlace::Sub;
            dispatch(target_device, op, m_state->m_grad.value(), incoming_grad);
        }
    }

    template <typename T>
    void Tensor<T>::accumulate_grad_matmul(const Tensor<T>& left, const Tensor<T>& right, BLASGEMMMeta& blas_meta, const std::vector<int64_t>& orig_shape) requires std::is_floating_point_v<T> {
        // why pass orig_shape?
        // what goes into matmul is flat. m_left and m_right. They have wrong shapes, but they share the same tensor state as before flattening.
        // They share states because the swaps were done via mutating member variables m_shape and m_strides, not .reshape() etc.
        // (exception: can be forced to be contiguous. Then it is sharing tensor state with A.contiguous() and not A. But still, issue persists)
        // Since they share TensorState, initializing grad of flat tensor means initializing grad of A or A.contiguous() what makes it the wrong shape (flat) 
        Device target_device = infer_assert_device(*this, left, right);

        if (!m_requires_grad) {return;}

        if (!m_state->m_grad.has_value()) {
            Tensor<T> local_grad = Tensor<T>(orig_shape, target_device, uninitialized);
            blas_meta.beta = static_cast<T>(0.0);
            dispatch_batched_gemm(target_device, local_grad, left, right, blas_meta);
            m_state->m_grad = std::move(local_grad);
        }
        else {
            dispatch_batched_gemm(target_device, m_state->m_grad.value(), left, right, blas_meta);
        }
    }

    template <typename T>
    void Tensor<T>::backward() {
        std::vector<TensorStateBase*> topo_order = AutogradEngine::build_topo(m_state.get());

        for (TensorStateBase* current : topo_order) {
            current->backward();
            current->clear_grad_if_non_leaf();
        }
    }

    template <typename T>
    void Tensor<T>::zero_grad() {
        m_state->m_grad = std::nullopt;
    }
}