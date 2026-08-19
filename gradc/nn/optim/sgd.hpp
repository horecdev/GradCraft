#pragma once

#include "optimizer.hpp"
#include "gradc/backend/dispatcher.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class SGD : public Optimizer<T> {
        public:
            SGD(std::unordered_map<std::string, Parameter<T>*> named_params, T lr) {
                this->m_named_params = std::move(named_params);
                this->assert_valid_params();
                Device target_device = this->optim_device();

                this->m_lr = Tensor<T>(lr, target_device);
            }

            void step() {
                Device target_device = this->optim_device();
                for (auto& [name, p_ptr] : this->m_named_params) {
                    if (!p_ptr->grad().has_value()) {
                        continue;
                    }
                    Tensor<T> grad_mul_lr = Tensor<T>(p_ptr->tensor().shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, grad_mul_lr, p_ptr->grad().value(), Tensor<T>(m_lr, target_device));
                    dispatch(target_device, BinaryOpInPlace::Sub, p_ptr->tensor(), grad_mul_lr);
                }
            }   

            void sync_state_param_device() override {
                Device target_device = this->optim_device();

                Tensor<T> moved_lr = this->m_lr.to(target_device);
                moved_lr.realize();
                this->m_lr = moved_lr.detach();
            }
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class SGDMomentum : public Optimizer<T> {
        private:
            Tensor<T> m_beta;
            std::unordered_map<std::string, Tensor<T>> m_velocities;
        public:
            SGDMomentum(std::unordered_map<std::string, Parameter<T>*> named_params, T lr, T beta) {
                this->m_named_params = std::move(named_params);
                this->assert_valid_params();
                Device target_device = this->optim_device();
                
                this->m_lr = Tensor<T>(lr, target_device);
                m_beta = Tensor<T>(beta, target_device);
                for (auto& [name, p_ptr] : this->m_named_params) {
                    m_velocities[name] = Tensor<T>(p_ptr->tensor().shape(), T(0), target_device);
                }
            }

            void step() {
                Device target_device = this->optim_device();
                for (auto& [name, p_ptr] : this->m_named_params) {
                    // v[t] = beta * v[t-1] + grad;
                    // this means we retain 90% of prev momentum and add grad. If beta=1 we accumulate infinitely
                    // Setting beta < 1 sets a speed limit
                    // v[1] = g1
                    // v[2] = Bv1 + g2 = Bg1 + g2
                    // v[3] = B^2v1 + Bg2 + g3
                    // so on. Limit is (for constant g) g / (1 - beta). This means if grad is constant, speed limit = 10 x g
                    // You retain the DIRECTION of the grad. do W = W - lr * vel
                    if (p_ptr->grad().has_value()) {
                        continue;
                    }

                    Tensor<T>& vel = m_velocities[name];

                    dispatch(target_device, BinaryOpInPlace::Mul, vel, m_beta);
                    dispatch(target_device, BinaryOpInPlace::Add, vel, p_ptr->grad().value());
                    Tensor<T> vel_mul_lr = Tensor<T>(p_ptr->tensor().shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, vel_mul_lr, vel, this->m_lr);
                    dispatch(target_device, BinaryOpInPlace::Sub, p_ptr->tensor(), vel_mul_lr);
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

                for (auto& [name, vel] : m_velocities) {
                    Tensor<T> moved_vel = vel.to(target_device);
                    moved_vel.realize();
                    vel = moved_vel.detach();
                }
            }

            std::unordered_map<std::string, Tensor<T>> state_dict(Device device = Device(DeviceType::CPU)) override {
                std::unordered_map<std::string, Tensor<T>> result;
                for (const auto& [name, vel] : m_velocities) {
                    Tensor<T> moved_vel = vel.to(device);
                    moved_vel.realize();
                    result[name + ".sgdm_vel"] = moved_vel.detach();
                }
                return result;
            }

            void load_state_dict(const std::unordered_map<std::string, Tensor<T>>& state_dict) override {
                Device target_device = this->optim_device();
                for (auto& [name, vel] : m_velocities) {
                    auto it = state_dict.find(name + ".sgdm_vel");
                    if (it != state_dict.end()) {
                        Tensor<T> moved_tensor = it->second.to(target_device);
                        moved_tensor.realize();
                        vel = moved_tensor.detach();
                    }
                    else {
                        std::cout << "Warning: '" + name + "' is in the SGDMomentum velocity, but couldn't find its counterpart in state_dict during loading." << std::endl;
                    }
                }

            }
    };
}