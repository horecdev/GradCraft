#pragma once
#include "../base/module.hpp"
#include "../base/initializers.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class LayerNorm : public Module<T> {
        private:
            Parameter<T> m_gamma;
            Parameter<T> m_beta;
            std::vector<int64_t> m_axes;
            T m_eps;
        public:
            LayerNorm(const std::vector<int64_t>& axes, const Initializer<T>& gamma_init, const Initializer<T>& beta_init, T eps) {
                m_gamma = Parameter<T>(gamma_init.generate(axes, Device(DeviceType::CPU))); 
                m_beta = Parameter<T>(beta_init.generate(axes, Device(DeviceType::CPU)));
                m_axes = std::move(axes);
                m_eps = eps;

                this->register_parameter("gamma", &m_gamma);
                this->register_parameter("beta", &m_beta);
            }

            Tensor<T> forward(Tensor<T> x) {
                Tensor<T> normalized = layernorm(x, m_gamma, m_beta, m_axes, m_eps);
                return normalized;
            }
    };
}