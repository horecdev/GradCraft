#pragma once

#include "gradc/core/tensor.hpp"

namespace gradc {

    template <typename T>
    requires std::is_floating_point_v<T>
    class Initializer {
        public:
            virtual Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const = 0;
    };

    template <typename T>
    class ZerosInit : public Initializer<T> {
        public:
            ZerosInit() = default;

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::zeros(shape, device);
            }
    };

    template <typename T>
    class OnesInit : public Initializer<T> {
        public:
            OnesInit() = default;

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::ones(shape, device);
            }
    };
    

    template <typename T>
    class NormalInit : public Initializer<T> {
        private:
            T m_mean;
            T m_std;
        public:
            NormalInit(T mean, T std) : m_mean(mean), m_std(std) {}

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::normal(shape, m_mean, m_std, device);
            }
    };

    template <typename T>
    class UniformInit : public Initializer<T> {
        private:
            T m_low;
            T m_high;
        public:
            UniformInit(T low, T high) : m_low(low), m_high(high) {}

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::uniform(shape, m_low, m_high, device);
            }
    };

    template <typename T>
    class XavierNormalInit : public Initializer<T> {
        public:
            XavierNormalInit() = default;

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::xavier_normal(shape, shape[0], shape[1], device);
            }
    };

    template <typename T>
    class XavierUniformInit : public Initializer<T> {
        public:
            XavierUniformInit() = default;

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::xavier_uniform(shape, shape[0], shape[1], device);
            }
    };

    template <typename T>
    class KaimingNormalInit : public Initializer<T> {
        private:
            T m_a;
        public:
            KaimingNormalInit(T a) : m_a(a) {}

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::kaiming_normal(shape, shape[0], m_a, device);
            }
    };

    template <typename T>
    class KaimingUniformInit : public Initializer<T> {
        private:
            T m_a;
        public:
            KaimingUniformInit(T a) : m_a(a) {}

            Tensor<T> generate(const std::vector<int64_t>& shape, Device device) const override {
                return Tensor<T>::kaiming_uniform(shape, shape[0], m_a, device);
            }
    };

    
    
}