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
            RedMeta m_red_meta;
        public:
            SumNode(Tensor<T> parent, RedMeta red_meta) : m_parent(std::move(parent)), m_red_meta(std::move(red_meta)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();
                
                Tensor<T> result = Tensor<T>(m_red_meta.result_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Sum, m_red_meta, result, m_parent);
                return result;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    m_parent.accumulate_grad(Tensor<T>(m_parent.shape(), m_red_meta.temp_strides, 0, out_grad._get_storage(), false));
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
            RedMeta m_red_meta;
        public:
            MeanNode(Tensor<T> parent, RedMeta red_meta) : m_parent(std::move(parent)), m_red_meta(std::move(red_meta)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                Tensor<T> summed = Tensor<T>(m_red_meta.result_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Sum, m_red_meta, summed, m_parent);
                dispatch(target_device, BinaryOpInPlace::Div, summed, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));
                return summed;
            }

            void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();
                    Tensor<T> divided_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Div, divided_grad, out_grad, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));
                    Tensor<T> strided_mean_grad = Tensor<T>(m_parent.shape(), m_red_meta.temp_strides, 0, divided_grad._get_storage(), false);
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
            RedMeta m_red_meta;
            Tensor<T> m_result;
        public:
            MaxNode(Tensor<T> parent, RedMeta red_meta) : m_parent(std::move(parent)), m_red_meta(std::move(red_meta)) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                Tensor<T> result = Tensor<T>(m_red_meta.result_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Max, m_red_meta, result, m_parent);

                if (m_parent.requires_grad()) {
                    m_result = Tensor<T>(m_red_meta.result_shape, target_device, uninitialized); // create a deep copy of result
                    dispatch(target_device, UnaryOp::Identity, m_result, result);
                }
                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> reshaped_result = lobotomized_reshape_view(m_result, m_red_meta.temp_shape);
                    Tensor<T> broadcast_result = lobotomized_broadcast_view(reshaped_result, m_parent.shape());

                    Tensor<T> reshaped_grad = lobotomized_reshape_view(out_grad, m_red_meta.temp_shape);
                    Tensor<T> broadcast_grad = lobotomized_broadcast_view(out_grad, m_parent.shape());
                    
                    Tensor<T> mask = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::EqMask, mask, m_parent, broadcast_result);

                    Tensor<T> grad_input = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, grad_input, mask, broadcast_grad);
                    m_parent.accumulate_grad(grad_input);

                    if (!retain_graph) {
                        m_result = Tensor<T>();
                    }
                    
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class MinNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            RedMeta m_red_meta;
            Tensor<T> m_result;
        public:
            MinNode(Tensor<T> parent, RedMeta red_meta) : m_parent(std::move(parent)), m_red_meta(std::move(red_meta)) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                Tensor<T> result = Tensor<T>(m_red_meta.result_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Min, m_red_meta, result, m_parent);

                if (m_parent.requires_grad()) {
                    m_result = Tensor<T>(m_red_meta.result_shape, target_device, uninitialized); // create a deep copy of result
                    dispatch(target_device, UnaryOp::Identity, m_result, result);
                }
                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    Device target_device = out_grad.device();

                    Tensor<T> reshaped_result = lobotomized_reshape_view(m_result, m_red_meta.temp_shape);
                    Tensor<T> broadcast_result = lobotomized_broadcast_view(reshaped_result, m_parent.shape());

                    Tensor<T> reshaped_grad = lobotomized_reshape_view(out_grad, m_red_meta.temp_shape);
                    Tensor<T> broadcast_grad = lobotomized_broadcast_view(out_grad, m_parent.shape());
                    
                    Tensor<T> mask = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::EqMask, mask, m_parent, broadcast_result);

                    Tensor<T> grad_input = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, grad_input, mask, broadcast_grad);
                    m_parent.accumulate_grad(grad_input);

                    if (!retain_graph) {
                        m_result = Tensor<T>(); 
                    }
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class ArgMaxNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            std::vector<int64_t> m_result_shape;
            int64_t m_dim;
        public:
            ArgMaxNode(Tensor<T> parent, std::vector<int64_t> result_shape, int64_t dim) : m_parent(std::move(parent)), m_result_shape(std::move(result_shape)), m_dim(dim) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                Tensor<T> result = Tensor<T>(m_result_shape,target_device, uninitialized);
                dispatch(target_device, ArgExtrOp::ArgMax, m_dim, result, m_parent);

                return result;
            }

            void backward([[maybe_unused]] const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    throw std::runtime_error("ArgMax parent cannot require grad");
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class ArgMinNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            std::vector<int64_t> m_result_shape;
            int64_t m_dim;
        public:
            ArgMinNode(Tensor<T> parent, std::vector<int64_t> result_shape, int64_t dim) : m_parent(std::move(parent)), m_result_shape(std::move(result_shape)), m_dim(dim) {}
            
            Tensor<T> realize() override {
                m_parent.realize();
                Device target_device = m_parent.device();

                Tensor<T> result = Tensor<T>(m_result_shape, target_device, uninitialized);
                dispatch(target_device, ArgExtrOp::ArgMin, m_dim, result, m_parent);

                return result;
            }

            void backward([[maybe_unused]] const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
                if (m_parent.requires_grad()) {
                    throw std::runtime_error("ArgMin parent cannot require grad");
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };
}