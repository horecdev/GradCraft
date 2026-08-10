#pragma once

#include "../../core/detail/shape_inference.hpp"
#include "../../core/detail/tensor_lob_view.hpp"
#include "../../backend/op_types.hpp"
#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../../core/types.hpp"
#include "../node.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T>
    class TransposeNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
            int64_t m_dim0;
            int64_t m_dim1;

        public:
            TransposeNode(Tensor<T> parent, int64_t dim0, int64_t dim1) : m_parent(std::move(parent)), m_dim0(dim0), m_dim1(dim1) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> transposed_grad = lobotomized_transpose_view(out_grad, m_dim0,  m_dim1);
                    m_parent.accumulate_grad(transposed_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class ReshapeNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            ReshapeNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override { // parent AND out_grad are contiguous, (aligns 1D memory perfectly) so can just copy shape, strides.
                if (m_parent.requires_grad()) {
                    Tensor<T> reshaped_grad = Tensor<T>(m_parent.shape(), m_parent.strides(), 0, out_grad._get_storage(), false);
                    m_parent.accumulate_grad(reshaped_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class PermuteNode: public Node<T> {
        private:
            Tensor<T> m_parent;
            std::vector<int64_t> m_axes;
        public:
            PermuteNode(Tensor<T> parent, std::vector<int64_t> axes) : m_parent(std::move(parent)), m_axes(std::move(axes)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    const int64_t n_dim = std::ssize(m_parent.shape());
                    std::vector<int64_t> normalized_axes = normalize_axes_vector(m_axes, n_dim);
                    std::vector<int64_t> backward_axes(n_dim, -1);
                    for (int64_t i = 0; i < n_dim; ++i) {
                        backward_axes[normalized_axes[i]] = i; // if 0 went to 2, then 2 goes to 0
                    }

                    Tensor<T> permuted_grad = lobotomized_permute_view(out_grad, backward_axes);
                    m_parent.accumulate_grad(permuted_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SliceNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
            std::vector<IndexDesc> m_descriptors;
        public:
            SliceNode(Tensor<T> parent, std::vector<IndexDesc> descriptors) : m_parent(std::move(parent)), m_descriptors(std::move(descriptors)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> full_grad = Tensor<T>(m_parent.shape(), T(0), target_device); // Incoming grad is [5, 32] but the parent is [5, 10, 32]. 
                    // You add dimension back for the temporary tensor (filled with 0s).
                    Tensor<T> grad_view = create_lobotomized_slice_view(full_grad, m_descriptors);
                    dispatch(target_device, UnaryOp::Identity, grad_view, out_grad);
                    // grad_view and out_grad have the same dimensions.
                    // grad_view has buffer of full_grad that matches shape of m_parent, so in-place assignment actually edits full_grad.
                    m_parent.accumulate_grad(full_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class SqueezeNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            SqueezeNode(Tensor<T> parent) : m_parent(std::move(parent)) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> unsqueezed_grad = lobotomized_reshape_view(out_grad, m_parent.shape());
                    m_parent.accumulate_grad(unsqueezed_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class UnsqueezeNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            UnsqueezeNode(Tensor<T> parent) : m_parent(std::move(parent)) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> squeezed_grad = lobotomized_reshape_view(out_grad, m_parent.shape());
                    m_parent.accumulate_grad(squeezed_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

}