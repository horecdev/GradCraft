#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

namespace gradc {

    template <typename T>
    class LayerNormNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            Tensor<T> m_gamma;
            Tensor<T> m_beta;
            ReductionMetadata m_reduction_metadata;
            std::vector<int64_t> m_normalized_shape;
            T m_eps;
            Tensor<T> m_inv_std;
            Tensor<T> m_normalized_z_scores;
        public:
            LayerNormNode(Tensor<T> parent, Tensor<T> gamma, Tensor<T> beta, ReductionMetadata reduction_metadata, std::vector<int64_t> normalized_shape, T eps)
             : m_parent(std::move(parent)), m_gamma(std::move(gamma)), m_beta(std::move(beta)), m_reduction_metadata(std::move(reduction_metadata)), m_normalized_shape(std::move(normalized_shape)), m_eps(eps) {}

            Tensor<T> realize() override {
                Device target_device = m_parent.device();
                if (m_parent.is_exclusive()) {
                    // X is (B, T, C), reducing over axis=1
                    // normalized shape is (1, T, 1) (what I am reducing over)

                    // first calcualate the mean, shape is (B, 1, C)
                    Tensor<T> mean = Tensor<T>(m_reduction_metadata.temp_shape, target_device, uninitialized);
                    dispatch(target_device, ReduceOp::Sum, m_reduction_metadata, mean, m_parent);
                    dispatch(target_device, BinaryOpInPlace::Div, m_reduction_metadata, mean, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol), target_device));
                    
                    dispatch(target_device, BinaryOpInPlace::Sub, m_parent, mean); // X - mean (needed later)

                    Tensor<T> squared_x_minus_mean = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Square, squared_x_minus_mean, m_parent);
                    Tensor<T>& inv_std = mean;
                    dispatch(target_device, ReduceOp::Sum, m_reduction_metadata, mean, squared_x_minus_mean);
                    dispatch(target_device, BinaryOpInPlace::IDiv, inv_std, Tensor<T>(T(static_cast<T>(1.0)), target_device));

                    Tensor<T>& normalized_z_scores = m_parent;
                    dispatch(target_device, BinaryOpInPlace::Mul, normalized_z_scores, inv_std);

                    // gamma must be same shape as normalized_shape. Same with beta. Both must be contiguous
                    Tensor<T> gamma_reshaped = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                    Tensor<T> beta_reshaped = lobotomized_reshape_view(m_beta, m_normalized_shape);
                    Tensor<T> shifted_scaled = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, shifted_scaled, normalized_z_scores, gamma_reshaped);
                    dispatch(target_device, BinaryOpInPlace::Add, shifted_scaled, beta_reshaped);

                    if (m_parent.requires_grad()) {
                        m_inv_std = inv_std;
                        m_normalized_z_scores = normalized_z_scores;
                    }

                    return shifted_scaled;
                }

                else {
                    Tensor<T> mean = Tensor<T>(m_reduction_metadata.temp_shape, target_device, uninitialized);
                    dispatch(target_device, ReduceOp::Sum, m_reduction_metadata, mean, m_parent);
                    dispatch(target_device, BinaryOpInPlace::Div, m_reduction_metadata, mean, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol), target_device));
                    
                    Tensor<T> x_minus_mean = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Sub, x_minus_mean, m_parent, mean);

                    Tensor<T> squared_x_minus_mean = Tensor<T>(x_minus_mean.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Square, squared_x_minus_mean, x_minus_mean);
                    Tensor<T>& inv_std = mean;
                    dispatch(target_device, ReduceOp::Sum, m_reduction_metadata, mean, squared_x_minus_mean);
                    dispatch(target_device, BinaryOpInPlace::IDiv, inv_std, Tensor<T>(T(static_cast<T>(1.0)), target_device));

                    Tensor<T>& normalized_z_scores = x_minus_mean;
                    dispatch(target_device, BinaryOpInPlace::Div, normalized_z_scores, inv_std);

                    Tensor<T> gamma_reshaped = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                    Tensor<T> beta_reshaped = lobotomized_reshape_view(m_beta, m_normalized_shape);
                    Tensor<T> shifted_scaled = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, shifted_scaled, normalized_z_scores, gamma_reshaped);
                    dispatch(target_device, BinaryOpInPlace::Add, shifted_scaled, beta_reshaped);

                    if (m_parent.requires_grad() || m_gamma.requires_grad() || m_beta.requires_grad()) {
                        m_inv_std = inv_std;
                        m_normalized_z_scores = normalized_z_scores;
                    }

                    return shifted_scaled;
                }

                
                


            }
    };
}