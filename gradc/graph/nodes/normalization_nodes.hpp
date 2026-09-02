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
            RedMeta m_red_meta;
            std::vector<int64_t> m_normalized_shape;
            T m_eps;
            Tensor<T> m_inv_std;
            Tensor<T> m_z_scores;
        public:
            LayerNormNode(Tensor<T> parent, Tensor<T> gamma, Tensor<T> beta, RedMeta red_meta, std::vector<int64_t> normalized_shape, T eps)
             : m_parent(std::move(parent)), m_gamma(std::move(gamma)), m_beta(std::move(beta)), m_red_meta(std::move(red_meta)), m_normalized_shape(std::move(normalized_shape)), m_eps(eps) {}

            Tensor<T> realize() override {
                m_parent.realize();
                m_gamma.realize();
                m_beta.realize();

                Device target_device = m_parent.device();

                bool can_mutate_parent = m_parent.is_exclusive();
                Tensor<T> x_minus_mean;
                if (can_mutate_parent) {
                    x_minus_mean = m_parent;
                }
                else {
                    x_minus_mean = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                }
                
                // LayerNorm is: (X - mean) / std. Literally z_scores. Then multiplied by gamma and beta is added
                
                Tensor<T> mean = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized);
                dispatch(target_device, ReduceOp::Sum, m_red_meta, mean, m_parent);
                dispatch(target_device, BinaryOpInPlace::Div, mean, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));
                
                if (can_mutate_parent) {
                    dispatch(target_device, BinaryOpInPlace::Sub, x_minus_mean, mean);
                }
                else {
                    dispatch(target_device, BinaryOp::Sub, x_minus_mean, m_parent, mean);
                }
                

                Tensor<T> squared_x_minus_mean = Tensor<T>(x_minus_mean.shape(), target_device, uninitialized);
                dispatch(target_device, UnaryOp::Square, squared_x_minus_mean, x_minus_mean);
                Tensor<T>& inv_std = mean;
                dispatch(target_device, ReduceOp::Sum, m_red_meta, inv_std, squared_x_minus_mean);
                dispatch(target_device, BinaryOpInPlace::Div, inv_std, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));
                dispatch(target_device, BinaryOpInPlace::Add, inv_std, Tensor<T>(static_cast<T>(m_eps), target_device));
                dispatch(target_device, UnaryOpInPlace::Sqrt, inv_std);
                dispatch(target_device, BinaryOpInPlace::IDiv, inv_std, Tensor<T>(static_cast<T>(1.0), target_device));

                Tensor<T>& normalized_z_scores = x_minus_mean;
                dispatch(target_device, BinaryOpInPlace::Mul, normalized_z_scores, inv_std);

                Tensor<T> gamma_reshaped = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                Tensor<T> beta_reshaped = lobotomized_reshape_view(m_beta, m_normalized_shape);
                Tensor<T> shifted_scaled = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                dispatch(target_device, BinaryOp::Mul, shifted_scaled, normalized_z_scores, gamma_reshaped);
                dispatch(target_device, BinaryOpInPlace::Add, shifted_scaled, beta_reshaped);

                if (m_gamma.requires_grad() && !m_parent.requires_grad()) {
                    m_z_scores = normalized_z_scores;
                }
                if (m_parent.requires_grad()) {
                    m_z_scores = normalized_z_scores;
                    m_inv_std = inv_std;
                }

                return shifted_scaled;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                Device target_device = out_grad.device();

                if (m_beta.requires_grad()) {
                    Tensor<T> dbeta = unbroadcast_grad(out_grad, m_normalized_shape); // result always contiguous (result of reduce op)
                    dbeta = lobotomized_reshape_view(dbeta, m_beta.shape());
                    m_beta.accumulate_grad(dbeta);
                }

                Tensor<T> scratchpad;
                if (m_gamma.requires_grad() || m_parent.requires_grad()) {
                    scratchpad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                }

                if (m_gamma.requires_grad()) {
                    Tensor<T>& dgamma_broadcast = scratchpad;
                    dispatch(target_device, BinaryOp::Mul, dgamma_broadcast, out_grad, m_z_scores);
                    Tensor<T> dgamma = unbroadcast_grad(dgamma_broadcast, m_normalized_shape);
                    dgamma = lobotomized_reshape_view(dgamma, m_gamma.shape());
                    m_gamma.accumulate_grad(dgamma);
                }

                if (m_parent.requires_grad()) {
                    
                    Tensor<T>& dx_hat = scratchpad;
                    Tensor<T> reshaped_gamma = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                    dispatch(target_device, BinaryOp::Mul, dx_hat, out_grad, reshaped_gamma);
                    Tensor<T> dx_hat_mean = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized);
                    dispatch(target_device, ReduceOp::Sum, m_red_meta, dx_hat_mean, dx_hat);
                    dispatch(target_device, BinaryOpInPlace::Div, dx_hat_mean, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));
                    Tensor<T> dx = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Sub, dx, dx_hat, dx_hat_mean);

                    Tensor<T> dx_hat_mul_normalized_z_scores = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, dx_hat_mul_normalized_z_scores, dx_hat, m_z_scores);
                    Tensor<T>& dx_hat_mul_normalized_z_scores_mean = dx_hat_mean;
                    dispatch(target_device, ReduceOp::Sum, m_red_meta, dx_hat_mul_normalized_z_scores_mean, dx_hat_mul_normalized_z_scores);
                    dispatch(target_device, BinaryOpInPlace::Div, dx_hat_mul_normalized_z_scores_mean, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));

                    Tensor<T>& norm_mul_mean_dx_hat_mul_norm = dx_hat;
                    dispatch(target_device, BinaryOp::Mul, norm_mul_mean_dx_hat_mul_norm, m_z_scores, dx_hat_mul_normalized_z_scores_mean);
                    dispatch(target_device, BinaryOpInPlace::Sub, dx, norm_mul_mean_dx_hat_mul_norm);
                    dispatch(target_device, BinaryOpInPlace::Mul, dx, m_inv_std);

                    m_parent.accumulate_grad(dx);
                }

                if (!retain_graph) {
                    m_inv_std = Tensor<T>();
                    m_z_scores = Tensor<T>();
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base(), m_gamma._get_state_base(), m_beta._get_state_base()};
            }
    };

    template <typename T>
    class RMSNormNaiveNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            Tensor<T> m_gamma;
            RedMeta m_red_meta;
            std::vector<int64_t> m_normalized_shape;
            T m_eps;

            Tensor<T> m_inv_rms;
        public:
            RMSNormNaiveNode(Tensor<T> parent, Tensor<T> gamma, RedMeta red_meta, std::vector<int64_t> normalized_shape, T eps)
             : m_parent(std::move(parent)), m_gamma(std::move(gamma)), m_red_meta(std::move(red_meta)), m_normalized_shape(std::move(normalized_shape)), m_eps(eps) {}

            Tensor<T> realize() override {
                m_parent.realize();
                m_gamma.realize();

                Device target_device = m_parent.device();

                bool can_mutate_parent = m_parent.is_exclusive() && !m_parent.requires_grad() && !m_gamma.requires_grad();
                Tensor<T> scratchpad = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                
                // RMSNorm is x / rms(x) * gamma
                
                dispatch(target_device, UnaryOp::Square, scratchpad, m_parent);
            
                Tensor<T> inv_rms = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized); // (B, T, 1)
                dispatch(target_device, ReduceOp::Sum, m_red_meta, inv_rms, scratchpad);
                dispatch(target_device, BinaryOpInPlace::Div, inv_rms, Tensor<T>(m_red_meta.reduced_vol, target_device));
                dispatch(target_device, BinaryOpInPlace::Add, inv_rms, Tensor<T>(m_eps, target_device));
                dispatch(target_device, UnaryOpInPlace::Sqrt, inv_rms);
                dispatch(target_device, BinaryOpInPlace::IDiv, inv_rms, Tensor<T>(static_cast<T>(1.0), target_device));
                
                Tensor<T> result;
                if (can_mutate_parent) {
                    result = m_parent;
                    dispatch(target_device, BinaryOpInPlace::Mul, result, inv_rms);
                }
                else {
                    result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, result, m_parent, inv_rms);
                }
                Tensor<T> reshaped_gamma = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                dispatch(target_device, BinaryOpInPlace::Mul, result, reshaped_gamma);

                if (m_parent.requires_grad() || m_gamma.requires_grad()) {
                    m_inv_rms = std::move(inv_rms);
                }

                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                Device target_device = out_grad.device();

                Tensor<T> x_norm;
                if (m_gamma.requires_grad() || m_parent.requires_grad()) {
                    x_norm = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, x_norm, m_parent, m_inv_rms);
                }

                Tensor<T> scratchpad;
                if (m_gamma.requires_grad() || m_parent.requires_grad()) {
                    scratchpad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                }

                if (m_gamma.requires_grad()) {
                    Tensor<T> dgamma_broadcast = scratchpad;
                    dispatch(target_device, BinaryOp::Mul, dgamma_broadcast, out_grad, x_norm);

                    Tensor<T> dgamma = unbroadcast_grad(dgamma_broadcast, m_normalized_shape);
                    dgamma = lobotomized_reshape_view(dgamma, m_gamma.shape());
                    m_gamma.accumulate_grad(dgamma);
                }

                if (m_parent.requires_grad()) {
                    
                    Tensor<T>& dx_hat = scratchpad; // dx_norm/dL
                    Tensor<T> reshaped_gamma = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                    dispatch(target_device, BinaryOp::Mul, dx_hat, out_grad, reshaped_gamma);

                    Tensor<T> dx_inter = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, dx_inter, dx_hat, x_norm);

                    Tensor<T> sum_term = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized);
                    dispatch(target_device, ReduceOp::Sum, m_red_meta, sum_term, dx_inter);
                    dispatch(target_device, BinaryOpInPlace::Div, sum_term, Tensor<T>(static_cast<T>(m_red_meta.reduced_vol), target_device));

                    dispatch(target_device, BinaryOp::Mul, dx_inter, x_norm, sum_term);
                    dispatch(target_device, BinaryOpInPlace::Sub, dx_hat, dx_inter);
                    dispatch(target_device, BinaryOpInPlace::Mul, dx_hat, m_inv_rms);

                    m_parent.accumulate_grad(dx_hat);
                }

                if (!retain_graph) {
                    m_inv_rms = Tensor<T>();
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base(), m_gamma._get_state_base()};
            }
    };

    template <typename T>
    class RMSNormFastNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            Tensor<T> m_gamma;
            RedMeta m_red_meta;
            std::vector<int64_t> m_normalized_shape;
            T m_eps;

            Tensor<T> m_inv_rms;
        public:
            RMSNormFastNode(Tensor<T> parent, Tensor<T> gamma, RedMeta red_meta, std::vector<int64_t> normalized_shape, T eps)
             : m_parent(std::move(parent)), m_gamma(std::move(gamma)), m_red_meta(std::move(red_meta)), m_normalized_shape(std::move(normalized_shape)), m_eps(eps) {}

            Tensor<T> realize() override {
                m_parent.realize();
                m_gamma.realize();

                Device target_device = m_parent.device();

                if (m_parent.requires_grad() || m_gamma.requires_grad()) {
                    m_inv_rms = Tensor<T>(m_normalized_shape, target_device, uninitialized);
                }

                Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
                Tensor<T> reshaped_gamma = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                
                dispatch_rmsnorm_forward(target_device, result, m_inv_rms, m_parent, reshaped_gamma, m_red_meta, m_normalized_shape, m_eps);

                return result;
            }

            void backward(const Tensor<T>& out_grad, bool retain_graph) override {
                Device target_device = out_grad.device();

                Tensor<T> dx = m_parent.requires_grad() ? Tensor<T>(m_parent.shape(), target_device, uninitialized) : Tensor<T>();;
                Tensor<T> dgamma = m_gamma.requires_grad() ? Tensor<T>(m_gamma.shape(), T(0), target_device) : Tensor<T>();
                Tensor<T> reshaped_gamma = lobotomized_reshape_view(m_gamma, m_normalized_shape);
                Tensor<T> reshaped_dgamma = lobotomized_reshape_view(dgamma, m_normalized_shape);

                dispatch_rmsnorm_backward(target_device, dx, reshaped_dgamma, out_grad, m_parent, reshaped_gamma, m_inv_rms, m_red_meta, m_normalized_shape);

                if (m_parent.requires_grad()) {m_parent.accumulate_grad(dx);}
                if (m_gamma.requires_grad()) {m_gamma.accumulate_grad(dgamma);}

                if (!retain_graph) {
                    m_inv_rms = Tensor<T>();
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base(), m_gamma._get_state_base()};
            }
    };

}