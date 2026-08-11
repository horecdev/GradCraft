#pragma once

#include "tensor.hpp"

namespace gradc {
    
    template <typename T>
    class Parameter {
        private:
            Tensor<T> m_tensor;
        public:
        // TODO: FIGURE OUT ALL THESE CONVERSIONS ON PARAMS, TURN PARAMETER INTO TENSOR WHEN FUNC TAKES BY VALUE
            Parameter() = default;
            Parameter(const Tensor<T>& tensor) : m_tensor(tensor) {m_tensor.set_requires_grad(true);} // copy tensor constructor
            Parameter(Tensor<T>&& tensor) : m_tensor(std::move(tensor)) {m_tensor.set_requires_grad(true);} // move tensor constructor
            // NO NEED FOR ASSIGNMENT OPERATORS BECAUSE YOU WILL NEVER CREATE PARAMETER AND THEN ASSIGN IT A VALUE / CHANGE EXISTING VALUE
            
            Parameter(Parameter<T>&& other) { // move constructor
                m_tensor = std::move(other.m_tensor);
            }

            Parameter(const Parameter<T>& other) = delete; // cannot copy a parameter - its useless we only care about underlying tensors.

            std::optional<Tensor<T>> grad() const {
                return m_tensor.grad();
            }

            void zero_grad()  {
                m_tensor.zero_grad();
            }

            bool requires_grad() {
                return m_tensor.requires_grad();
            }

            void set_requires_grad(bool value) {
                m_tensor.set_requires_grad(value);
            }
    };
}