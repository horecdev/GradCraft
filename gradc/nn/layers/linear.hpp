#pragma once
#include "../base/module.hpp"
#include "initializers.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Linear : public Module<T> {
        private:
            Parameter<T> m_weight;
            Parameter<T> m_bias;
        public:
            Linear(int64_t input_dim, int64_t output_dim, const Initializer<T>& initializer) {
                m_weight = Parameter<T>(initializer.generate({input_dim, output_dim}, Device(DeviceType::CPU))); // default generate on the CPU
                m_bias = Parameter<T>(initializer.generate({output_dim}, Device(DeviceType::CPU)));

                this->register_parameter("W", &m_weight);
                this->register_parameter("b", &m_bias);
            }

            


    };
}