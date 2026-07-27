#pragma once

#include "../core/tensor.hpp"
#include "../core/tensor_state.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T>::Tensor() : m_shape({}), m_strides({}), m_offset(0), m_state(nullptr), m_requires_grad(false) {} // default constructor
    
    template <typename T>
    Tensor<T>::Tensor(T value, Device device)
        : m_shape({}), m_strides({}), m_offset(0), m_state(std::make_shared<TensorState<T>>(1, value, device, true, true)), m_requires_grad(false) {}

    template <typename T>
    Tensor<T>::Tensor(std::vector<int64_t> shape, T init_val, Device device) 
        // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
        : m_shape(std::move(shape)), m_strides(std::ssize(m_shape)), m_offset(0), m_requires_grad(false) {
            if (std::ssize(m_shape) == 0) { // a scalar (0-dimensional)
                m_state = std::make_shared<TensorState<T>>(1, init_val, device);
            }
            else {
                m_strides[std::ssize(m_shape) - 1] = 1; 
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_state = std::make_shared<TensorState<T>>(m_shape[0] * m_strides[0], init_val, device);
            }
        }

    template <typename T>
    Tensor<T>::Tensor(std::vector<int64_t> shape, Device device, UninitializedTag)
        : m_shape(std::move(shape)), m_strides(std::ssize(m_shape)), m_offset(0), m_requires_grad(false) {
            if (std::ssize(m_shape) == 0) { // a scalar (0-dimensional)
                m_state = std::make_shared<TensorState<T>>(1, T(0), device, true, false);
            }
            else {
                m_strides[std::ssize(m_shape) - 1] = 1; 
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_state = std::make_shared<TensorState<T>>(m_shape[0] * m_strides[0], T(0), device, true, false);
            }
        }
    

    template <typename T>
    Tensor<T>::Tensor(std::initializer_list<int64_t> shape, T init_val, Device device) : Tensor(std::vector<int64_t>(shape), init_val, device) {}

    template <typename T>
    Tensor<T>::Tensor(std::vector<int64_t> shape, bool requires_grad, LazyTag, Device device)
        // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
        : m_shape(std::move(shape)), m_strides(std::ssize(m_shape)), m_offset(0), m_requires_grad(validate_requires_grad(requires_grad)) {
            if (std::ssize(m_shape) == 0) { // a scalar (0-dimensional)
                m_state = std::make_shared<TensorState<T>>(1, T(0), device, false);
            }
            else {
                m_strides[std::ssize(m_shape) - 1] = 1; 
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_state = std::make_shared<TensorState<T>>(m_shape[0] * m_strides[0], T(0), device, false);
            }
        }

    template <typename T> // backdoor
    Tensor<T>::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<TensorState<T>> state, bool requires_grad) 
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset), m_state(std::move(state)), m_requires_grad(validate_requires_grad(requires_grad)) {} 

    template <typename T> // lobotomy constructor
    Tensor<T>::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<Storage<T>> storage, bool requires_grad) 
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset), m_requires_grad(validate_requires_grad(requires_grad)) {
            m_state = std::make_shared<TensorState<T>>(std::move(storage)); // op is nullptr
        }
    
    template <typename T>
    Tensor<T>::~Tensor() {
        // std::cout << "Tensor Destroyed" << std::endl;
    }

    template <typename T> // realizing one alias realizes all (shared TensorState)
    Tensor<T>::Tensor(const Tensor& source) 
        : m_shape(source.m_shape), m_strides(source.m_strides), m_offset(source.m_offset), m_state(source.m_state), m_requires_grad(source.m_requires_grad) {}


    template <typename T>
    Tensor<T>::Tensor(Tensor&& source) // move constructor [Tensor c = a + b]
        : m_shape(std::move(source.m_shape)), m_strides(std::move(source.m_strides)), m_offset(source.m_offset), m_state(std::move(source.m_state)), m_requires_grad(source.m_requires_grad) {}

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor& source) { // copy assignment operator [c = b]
        if (this != &source) {
            m_shape = source.m_shape;
            m_strides = source.m_strides;
            m_offset = source.m_offset;
            m_state = source.m_state;
            m_requires_grad = source.m_requires_grad;
        }
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& source) { // move assignment operator [a = b + c]
        if (this != &source) { // [a = transpose(a)], and transpose modifies it in-place, then it would trigger
            m_shape = std::move(source.m_shape);
            m_strides = std::move(source.m_strides);
            m_offset = source.m_offset;
            m_state = std::move(source.m_state);
            m_requires_grad = source.m_requires_grad;
        }
        return *this;
    }
}