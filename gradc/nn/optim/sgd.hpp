#pragma once

#include "optimizer.hpp"

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
                    if (!p_ptr.)
                }
            }

            
    };
}