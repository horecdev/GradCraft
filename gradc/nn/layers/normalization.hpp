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
            // param_shape does NOT have any dims set as 1. It gets reshaped to fit axes later
            LayerNorm(const std::vector<int64_t>& param_shape, const std::vector<int64_t>& axes, const Initializer<T>& gamma_init, const Initializer<T>& beta_init, T eps = static_cast<T>(1e-5)) {
                m_gamma = Parameter<T>(gamma_init.generate(param_shape, Device(DeviceType::CPU))); 
                m_beta = Parameter<T>(beta_init.generate(param_shape, Device(DeviceType::CPU)));
                m_axes = std::move(axes);
                m_eps = eps;

                this->register_parameter("gamma", &m_gamma);
                this->register_parameter("beta", &m_beta);
            }

            Tensor<T> forward(Tensor<T> x) {
                Tensor<T> normalized = layernorm(x, m_gamma, m_beta, m_axes, m_eps);
            }
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class RMSNorm : public Module<T> {
        private:
            Parameter<T> m_gamma;
            std::vector<int64_t> m_axes;
            T m_eps;
        public:
            RMSNorm(const std::vector<int64_t>& param_shape, const std::vector<int64_t>& axes, const Initializer<T>& gamma_init, T eps = static_cast<T>(1e-6)) {
                m_gamma = gamma_init.generate(param_shape, Device(DeviceType::CPU));
                m_axes = std::move(axes);
                m_eps = eps;

                this->register_parameter("gamma", &m_gamma);
            }

            Tensor<T> forward(Tensor<T> x) {
                return rmsnorm(x, m_gamma, m_axes, m_eps);
            }
    };
}