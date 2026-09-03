#pragma once

#include "optimizer.hpp"
#include "gradc/backend/dispatcher.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class RMSProp : public Optimizer<T> {
        private:
            Tensor<T> m_beta;
            Tensor<T> m_eps;
            std::unordered_map<std::string, Tensor<T>> m_uncentered_variances;
        public:
            RMSProp(std::unordered_map<std::string, Parameter<T>*> named_params, T lr, T beta = static_cast<T>(0.9), T eps = static_cast<T>(1e-8)) {
                this->m_named_params = std::move(named_params);
                this->assert_valid_params();
                Device target_device = this->optim_device();

                this->m_lr = Tensor<T>(lr, target_device);
                m_beta = Tensor<T>(beta, target_device);
                m_eps = Tensor<T>(eps, target_device);

                for (const auto& [name, p_ptr] : this->m_named_params) {
                    m_uncentered_variances[name] = Tensor<T>(p_ptr->tensor().shape(), T(0), target_device);
                }
            }

            void step() {
                Device target_device = this->optim_device();

                Tensor<T> one_minus_beta = Tensor<T>(std::vector<int64_t>{}, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Sub, one_minus_beta, Tensor<T>(static_cast<T>(1.0), target_device), m_beta);

                for (auto& [name, p_ptr] : this->m_named_params) {
                    // var[t] = B*var[t-1] + (1-B)*g[t]^2
                    // then W = W - lr / (sqrt(var[t])) * g[t]
                    // when sqrt(var[t]) is small = small grads = plateau, then you divide by a smaller number
                    // what means acceleration
                    // when sqrt(var[t]) is big = big grads = ravine, then you divide by a bigger number = handbrake

                    // in addition: you divide by sqrt(variance) = std. This means you get z-score of the grad.
                    // When variance is small and suddenly the grad is big, you get a big z-score
                    // if grad is as every other, z-score is 1, step is just by -lr.

                    // at first RMSProp can take big steps bc the variance is artificially small. You can introduce
                    // bias correction just like in Adam
                    if (!p_ptr->grad().has_value()) {
                        continue;
                    }
                    
                    Tensor<T>& var = m_uncentered_variances[name];
                    dispatch(target_device, BinaryOpInPlace::Mul, var, m_beta);

                    Tensor<T> new_var_part = Tensor<T>(var.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Square, new_var_part, p_ptr->grad().value());
                    dispatch(target_device, BinaryOpInPlace::Mul, new_var_part, one_minus_beta);
                    dispatch(target_device, BinaryOpInPlace::Add, var, new_var_part);

                    // variance is updated at this point

                    Tensor<T> update = Tensor<T>(var.shape(), target_device, uninitialized);
                    dispatch(target_device, UnaryOp::Sqrt, update, var);
                    dispatch(target_device, BinaryOpInPlace::Add, update, m_eps);
                    dispatch(target_device, BinaryOpInPlace::IDiv, update, this->m_lr);
                    dispatch(target_device, BinaryOpInPlace::Mul, update, p_ptr->grad().value());
                    dispatch(target_device, BinaryOpInPlace::Sub, p_ptr->tensor(), update);
                }
            }

            void sync_state_param_device() override {
                Device target_device = this->optim_device();
                
                Tensor<T> moved_lr = this->m_lr.to(target_device);
                moved_lr.realize();
                this->m_lr = moved_lr.detach();

                Tensor<T> moved_beta = m_beta.to(target_device);
                moved_beta.realize();
                m_beta = moved_beta.detach();
                
                Tensor<T> moved_eps = m_eps.to(target_device);
                moved_eps.realize();
                m_eps = moved_eps.detach();

                for (auto& [name, var] : m_uncentered_variances) {
                    Tensor<T> moved_var = var.to(target_device);
                    moved_var.realize();
                    var = moved_var.detach();
                }
            }

            std::unordered_map<std::string, Tensor<T>> state_dict(Device device = Device(DeviceType::CPU)) override {
                std::unordered_map<std::string, Tensor<T>> result;
                for (const auto& [name, var] : m_uncentered_variances) {
                    Tensor<T> moved_var = var.to(device);
                    moved_var.realize();
                    result[name + ".rmsprop_var"] = moved_var.detach();
                }
                return result;
            }

            void load_state_dict(const std::unordered_map<std::string, Tensor<T>>& state_dict) override {
                Device target_device = this->optim_device();
                for (auto& [name, var] : m_uncentered_variances) {
                    auto it = state_dict.find(name + ".rmsprop_var");
                    if (it != state_dict.end()) {
                        Tensor<T> moved_tensor = it->second.to(target_device);
                        moved_tensor.realize();
                        var = moved_tensor.detach();
                    }
                    else {
                        std::cout << "Warning: '" + name + "' is in the RMSProp variance, but couldn't find its counterpart in state_dict during loading." << std::endl;
                    }
                }
            }
    };
}