#pragma once

#include "optimizer.hpp"
#include "gradc/backend/dispatcher.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class SGD : public Optimizer<T> {
        private:
            T m_lr;
        public:
            SGD(std::vector<Parameter<T>*> params, T lr) : m_lr(lr) {
                this->m_params = std::move(params);
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
}