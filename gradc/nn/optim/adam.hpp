#pragma once

#include "optimizer.hpp"
#include "gradc/backend/dispatcher.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class AdamW : public Optimizer<T> {
        private:
            Tensor<T> m_beta1;
            Tensor<T> m_beta2;
            Tensor<T> m_weight_decay;
            Tensor<T> m_eps;
            Tensor<T> m_beta1_exp;
            Tensor<T> m_beta2_exp;
            std::unordered_map<std::string, Tensor<T>> m_first_moment;
            std::unordered_map<std::string, Tensor<T>> m_second_moment;
        public:
            AdamW(std::unordered_map<std::string, Parameter<T>*> named_params, T lr, T beta1 = static_cast<T>(0.9), T beta2 = static_cast<T>(0.999), T weight_decay = static_cast<T>(0.0), T eps = static_cast<T>(1e-7)) {
                this->m_named_params = std::move(named_params);
                this->assert_valid_params();
                Device target_device = this->optim_device();

                this->m_lr = Tensor<T>(lr, target_device);
                m_beta1 = Tensor<T>(beta1, target_device);
                m_beta2 = Tensor<T>(beta2, target_device);
                m_weight_decay = Tensor<T>(weight_decay, target_device);
                m_eps = Tensor<T>(eps, target_device);

                m_beta1_exp = Tensor<T>(T(1.0), target_device);
                m_beta2_exp = Tensor<T>(T(1.0), target_device);

                for (const auto& [name, p_ptr] : this->m_named_params) {
                    m_first_moment[name] = Tensor<T>(p_ptr->tensor().shape(), T(0), target_device);
                    m_second_moment[name] = Tensor<T>(p_ptr->tensor().shape(), T(0), target_device);
                }
            }

            void step() {
                Device target_device = this->optim_device();

                Tensor<T> penalty_factor = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Mul, penalty_factor, this->m_lr, m_weight_decay);
                dispatch(target_device, BinaryOpInPlace::ISub, penalty_factor, Tensor<T>(static_cast<T>(1.0), target_device));

                Tensor<T> one_minus_beta1 = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, one_minus_beta1, Tensor<T>(static_cast<T>(1.0), target_device), m_beta1);

                Tensor<T> one_minus_beta2 = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, one_minus_beta2, Tensor<T>(static_cast<T>(1.0), target_device), m_beta2);

                dispatch(target_device, BinaryOpInPlace::Mul, m_beta1_exp, m_beta1);
                dispatch(target_device, BinaryOpInPlace::Mul, m_beta2_exp, m_beta2);
                // both exp are B^t 
                // now calc 1 - B^t
                Tensor<T> mean_bias = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, mean_bias, Tensor<T>(static_cast<T>(1.0), target_device), m_beta1_exp);

                Tensor<T> var_bias = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, var_bias, Tensor<T>(static_cast<T>(1.0), target_device), m_beta2_exp);

                    
                for (auto& [name, p_ptr] : this->m_named_params) {
                    // first moment is the mean direction (no square)
                    // second moment is the variance (same as in RMSProp)
                    // both have bias correction to account for how EMA works and not pump down the var/mean
                    // W = W - lr * first_cor[t] / second_cor[t] - you get mean direction / std what means
                    // z-score of the mean direction

                    // mean[t] = B1 * mean[t-1] + (1-B1) * grad
                    // var[t] = B2 * var[t-1] + (1-B2) * grad**2

                    if (!p_ptr->grad().has_value()) {
                        continue;
                    }

                    if (!p_ptr->no_decay()) {
                        dispatch(target_device, BinaryOpInPlace::Mul, p_ptr->tensor(), penalty_factor);
                    }
                    
                    // first update the mean and do the bias correction
                    Tensor<T>& mean = m_first_moment[name];
                    dispatch(target_device, BinaryOpInPlace::Mul, mean, m_beta1);

                    Tensor<T> new_mean_part = Tensor<T>(mean.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, new_mean_part, p_ptr->grad().value(), one_minus_beta1);
                    dispatch(target_device, BinaryOpInPlace::Add, mean, new_mean_part);

                    Tensor<T>& mean_corrected = new_mean_part;
                    dispatch(target_device, BinaryOp::Div, mean_corrected, mean, mean_bias);

                    // second update the var
                    Tensor<T>& var = m_second_moment[name];
                    dispatch(target_device, BinaryOpInPlace::Mul, var, m_beta2);

                    Tensor<T> new_var_part = Tensor<T>(var.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Square, new_var_part, p_ptr->grad().value());
                    dispatch(target_device, BinaryOpInPlace::Mul, new_var_part, one_minus_beta2);
                    dispatch(target_device, BinaryOpInPlace::Add, var, new_var_part);

                    Tensor<T>& var_corrected = new_var_part;
                    dispatch(target_device, BinaryOp::Div, var_corrected, var, var_bias);

                    Tensor<T> update = Tensor<T>(var.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Sqrt, update, var_corrected);
                    dispatch(target_device, BinaryOpInPlace::Add, update, m_eps);
                    dispatch(target_device, BinaryOpInPlace::IDiv, update, this->m_lr);
                    dispatch(target_device, BinaryOpInPlace::Mul, update, mean_corrected);
                    dispatch(target_device, BinaryOpInPlace::Sub, p_ptr->tensor(), update);   
                }
            }

            void sync_state_param_device() override {
                Device target_device = this->optim_device();
                
                Tensor<T> moved_lr = this->m_lr.to(target_device);
                moved_lr.realize();
                this->m_lr = moved_lr.detach();

                Tensor<T> moved_beta1 = m_beta1.to(target_device);
                moved_beta1.realize();
                m_beta1 = moved_beta1.detach();

                Tensor<T> moved_beta2 = m_beta2.to(target_device);
                moved_beta2.realize();
                m_beta2 = moved_beta2.detach();

                Tensor<T> moved_weight_decay = m_weight_decay.to(target_device);
                moved_weight_decay.realize();
                m_weight_decay = moved_weight_decay.detach();

                Tensor<T> moved_eps = m_eps.to(target_device);
                moved_eps.realize();
                m_eps = moved_eps.detach();

                Tensor<T> moved_beta1_exp = m_beta1_exp.to(target_device);
                moved_beta1_exp.realize();
                m_beta1_exp = moved_beta1_exp.detach();

                Tensor<T> moved_beta2_exp = m_beta2_exp.to(target_device);
                moved_beta2_exp.realize();
                m_beta2_exp = moved_beta2_exp.detach();
                
                for (auto& [name, mean] : m_first_moment) {
                    Tensor<T> moved_mean = mean.to(target_device);
                    moved_mean.realize();
                    mean = moved_mean.detach();
                }

                for (auto& [name, var] : m_second_moment) {
                    Tensor<T> moved_var = var.to(target_device);
                    moved_var.realize();
                    var = moved_var.detach();
                }
            }
            
            std::unordered_map<std::string, Tensor<T>> state_dict(Device device = Device(DeviceType::CPU)) override {
                std::unordered_map<std::string, Tensor<T>> result;
                for (const auto& [name, mean] : m_first_moment) {
                    Tensor<T> moved_mean = mean.to(device);
                    moved_mean.realize();
                    result[name + ".adamw_first"] = moved_mean.detach();
                }

                for (const auto& [name, var] : m_second_moment) {
                    Tensor<T> moved_var = var.to(device);
                    moved_var.realize();
                    result[name + ".adamw_second"] = moved_var.detach();
                }

                Tensor<T> moved_beta1_exp = m_beta1_exp.to(device);
                moved_beta1_exp.realize();
                result["beta1_exp"] = moved_beta1_exp.detach();

                Tensor<T> moved_beta2_exp = m_beta2_exp.to(device);
                moved_beta2_exp.realize();
                result["beta2_exp"] = moved_beta2_exp.detach();

                return result;
            }

            void load_state_dict(const std::unordered_map<std::string, Tensor<T>>& state_dict) override {
                Device target_device = this->optim_device();
                for (auto& [name, mean] : m_first_moment) {
                    auto it = state_dict.find(name + ".adamw_first");
                    if (it != state_dict.end()) {
                        Tensor<T> moved_tensor = it->second.to(target_device);
                        moved_tensor.realize();
                        mean = moved_tensor.detach();
                    }
                    else {
                        std::cout << "Warning: '" + name + "' is in the AdamW first moment, but couldn't find its counterpart in state_dict during loading." << std::endl;
                    }
                }
                for (auto& [name, var] : m_second_moment) {
                    auto it = state_dict.find(name + ".adamw_second");
                    if (it != state_dict.end()) {
                        Tensor<T> moved_tensor = it->second.to(target_device);
                        moved_tensor.realize();
                        var = moved_tensor.detach();
                    }
                    else {
                        std::cout << "Warning: '" + name + "' is in the AdamW second moment, but couldn't find its counterpart in state_dict during loading." << std::endl;
                    }
                }

                auto it1 = state_dict.find("beta1_exp");
                if (it1 != state_dict.end()) {
                    Tensor<T> moved_beta = it1->second.to(target_device);
                    moved_beta.realize();
                    m_beta1_exp = moved_beta.detach();
                }
                else {
                    throw std::runtime_error("Tried loading AdamW optimizer with state_dict without key 'beta1_exp'");
                }

                auto it2 = state_dict.find("beta2_exp");
                if (it2 != state_dict.end()) {
                    Tensor<T> moved_beta = it2->second.to(target_device);
                    moved_beta.realize();
                    m_beta2_exp = moved_beta.detach();
                }
                else {
                    throw std::runtime_error("Tried loading AdamW optimizer with state_dict without key 'beta2_exp'");
                }
            }
    };
}