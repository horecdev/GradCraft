#pragma once

#include "optimizer.hpp"
#include "gradc/backend/dispatcher.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class SGD : public Optimizer<T> {
        private:
            Tensor<T> m_lr;
        public:
            SGD(std::vector<Parameter<T>*> params, T lr) {
                this->m_params = std::move(params);
                this->assert_same_device_params();
                m_lr = Tensor<T>(lr, this->m_params[0]->device());
            }

            void step() {
                for (Parameter<T>* p_ptr : this->m_params) {
                    if (!p_ptr->grad().has_value()) {
                        continue;
                    }
                    Device target_device = p_ptr->device();
                    Tensor<T> grad_mul_lr = Tensor<T>(p_ptr->tensor().shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, grad_mul_lr, p_ptr->grad().value(), Tensor<T>(m_lr, target_device));
                    dispatch(target_device, BinaryOpInPlace::Sub, p_ptr->tensor(), grad_mul_lr);
                }
            }    
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class SGDMomentum : public Optimizer<T> {
        private:
            Tensor<T> m_lr;
            Tensor<T> m_beta;
            std::unordered_map<Parameter<T>*, Tensor<T>> m_velocities;
        public:
            SGDMomentum(std::vector<Parameter<T>*> params, T lr, T beta) {
                this->m_params = std::move(params);
                this->assert_same_device_params();
                Device target_device = this->m_params[0]->device();
                
                m_beta = Tensor<T>(beta, target_device);
                m_lr = Tensor<T>(lr, target_device);
                for (Parameter<T>* p_ptr : this->m_params) {
                    m_velocities[p_ptr] = Tensor<T>(p_ptr->tensor().shape(), T(0), p_ptr->device());
                }
            }

            void step() {
                Device target_device = this->m_params[0]->device();
                for (auto& [p_ptr, vel] : m_velocities) {
                    // v[t] = beta * v[t-1] + grad;
                    // this means we retain 90% of prev momentum and add grad. If beta=1 we accumulate infinitely
                    // Setting beta < 1 sets a speed limit
                    // v[1] = g1
                    // v[2] = Bv1 + g2 = Bg1 + g2
                    // v[3] = B^2v1 + Bg2 + g3
                    // so on. Limit is (for constant g) g / (1 - beta). This means if grad is constant, speed limit = 10 x g
                    // You retain the DIRECTION of the grad. do W = W - lr * vel
                    // Vel keeps the average (EMA) direction of the grad.
                    if (!p_ptr->grad().has_value()) {
                        continue;
                    }

                    dispatch(target_device, BinaryOpInPlace::Mul, vel, m_beta);
                    dispatch(target_device, BinaryOpInPlace::Add, vel, p_ptr->grad().value());
                    Tensor<T> vel_mul_lr = Tensor<T>(p_ptr->tensor().shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, vel_mul_lr, vel, m_lr);
                    dispatch(target_device, BinaryOpInPlace::Sub, p_ptr->tensor(), vel_mul_lr);
                }
            }
    };
}