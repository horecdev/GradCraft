#pragma once

#include "gradc/core/tensor.hpp"

namespace gradc {
    
    template <typename T>
    requires std::is_floating_point_v<T>
    class Parameter {
        private:
            Tensor<T> m_tensor;
            bool m_no_decay = false;

            void validate_dense_tensor() const {
            if (!m_tensor.is_dense()) {
                throw std::runtime_error("Parameters must be dense tensors (contiguous, volume == storage size).");
            }
        }
        public:
            Parameter() = default;
            Parameter(const Tensor<T>& tensor) : m_tensor(tensor) {validate_dense_tensor(); m_tensor.set_requires_grad(true);} // copy tensor constructor
            Parameter(Tensor<T>&& tensor) : m_tensor(std::move(tensor)) {validate_dense_tensor(); m_tensor.set_requires_grad(true);} // move tensor constructor
            
            Parameter(Parameter<T>&& other) noexcept = default;
            Parameter& operator=(Parameter<T>&& other) noexcept = default;

            Parameter(const Parameter<T>& other) = delete; // cannot copy a parameter - its useless we only care about underlying tensors.
            Parameter& operator=(const Parameter<T>& other) = delete;

            operator Tensor<T>() {
                return m_tensor;
            }
            operator Tensor<T>() const {
                return m_tensor;
            }
            
            Tensor<T>& tensor() {return m_tensor;}
            const Tensor<T>& tensor() const {return m_tensor;}

            std::optional<Tensor<T>> grad() const {
                return m_tensor.grad();
            }

            void zero_grad()  {
                m_tensor.zero_grad();
            }

            bool requires_grad() const {
                return m_tensor.requires_grad();
            }

            void set_requires_grad(bool value) {
                m_tensor.set_requires_grad(value);
            }

            bool no_decay() const {
                return m_no_decay;
            }

            bool set_no_decay(bool value) {
                m_no_decay = value;
            }
    };
}