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
            Linear(int64_t input_dim, int64_t output_dim, const Initializer<T>& w_init, const Initializer<T>& b_init) {
                m_weight = Parameter<T>(w_init.generate({input_dim, output_dim}, Device(DeviceType::CPU))); // default generate on the CPU
                m_bias = Parameter<T>(b_init.generate({output_dim}, Device(DeviceType::CPU)));

                this->register_parameter("W", &m_weight);
                this->register_parameter("b", &m_bias);
            }

            Tensor<T> forward(Tensor<T> x) {
                Tensor<T> y = matmul(std::move(x), m_weight.tensor()) + m_bias.tensor();
                return y;
            }
    };

    template <typename T>
    class ReLU : public Module<T> {
        public:
            ReLU() = default;

            Tensor<T> forward(Tensor<T> x) {
                return x.relu();
            }
    };
}