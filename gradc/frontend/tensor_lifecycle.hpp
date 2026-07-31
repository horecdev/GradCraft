#pragma once

#include "../core/tensor.hpp"
#include "../core/tensor_state.hpp"
#include "../backend/cpu/cpu_utils.hpp"
#include "../backend/cuda/cuda_utils.hpp"

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
    Tensor<T> Tensor<T>::zeros(std::vector<int64_t> shape, Device device) {
        return Tensor<T>(std::move(shape), T(0), device);
    }

    template <typename T>
    Tensor<T> Tensor<T>::ones(std::vector<int64_t> shape, Device device) {
        return Tensor<T>(std::move(shape), T(1), device);
    }

    template <typename T>
    Tensor<T> Tensor<T>::full(std::vector<int64_t> shape, T fill_val, Device device) {
        return Tensor<T>(std::move(shape), fill_val, device);
    }

    template <typename T>
    Tensor<T> Tensor<T>::arange(T start, T stop, T step, Device device) {
        int64_t n_elems = (stop - start) / step;
        if (n_elems == 0) {
            throw std::runtime_error("n_elems in .arange() is 0.");
        }
        Tensor<T> result = Tensor<T>({n_elems}, device, uninitialized);
        T* ptr = result._get_storage()->data();
        if (device.is_cpu()) {
            CPUUtils::fill_arange(ptr, start, step, n_elems);
        }
        else if (device.is_cuda()) {
            CUDAUtils::fill_arange(ptr, start, step, n_elems, device);
        }
        return result;
    }

    template <typename T>
    template <typename U>
    requires std::is_floating_point_v<U> // compiler treats as signature, so must put in both declaration and out of line definition
    Tensor<T> Tensor<T>::normal(std::vector<int64_t> shape, T mean, T std, Device device) {
        Tensor<T> result = Tensor<T>(shape, device, uninitialized);
        T* ptr = result._get_storage()->data();
        int64_t n_elems = result.volume();
        if (device.is_cpu()) {
            CPUUtils::fill_normal(ptr, mean, std, n_elems);
        }
        else if (device.is_cuda()) {
            CUDAUtils::fill_normal(ptr, mean, std, n_elems);
        }
        return result;
    }

    template <typename T>
    template <typename U>
    requires std::is_floating_point_v<U>
    Tensor<T> Tensor<T>::uniform(std::vector<int64_t> shape, T low, T high, Device device) {
        Tensor<T> result = Tensor<T>(shape, device, uninitialized);
        T* ptr = result._get_storage()->data();
        int64_t n_elems = result.volume();
        if (device.is_cpu()) {
            CPUUtils::fill_uniform(ptr, low, high, n_elems);
        }
        else if (device.is_cuda()) {
            CUDAUtils::fill_uniform(ptr, low, high, n_elems);
        }
        return result;
    }

    template <typename T>
    template <typename U>
    requires std::is_floating_point_v<U>
    Tensor<T> Tensor<T>::xavier_uniform(std::vector<int64_t> shape, int64_t fan_in, int64_t fan_out, Device device) {
        T bound = static_cast<T>(std::sqrt(6.0 / static_cast<double>(fan_in + fan_out)));
        return Tensor<T>::uniform(std::move(shape), -bound, bound, device);
    }

    template <typename T>
    template <typename U>
    requires std::is_floating_point_v<U>
    Tensor<T> Tensor<T>::xavier_normal(std::vector<int64_t> shape, int64_t fan_in, int64_t fan_out, Device device) {
        T std = static_cast<T>(std::sqrt(2.0 / static_cast<double>(fan_in + fan_out)));
        return Tensor<T>::normal(std::move(shape), T(0), std, device);
    }

    template <typename T>
    template <typename U>
    requires std::is_floating_point_v<U>
    Tensor<T> Tensor<T>::kaiming_uniform(std::vector<int64_t> shape, int64_t fan_in, T a, Device device) {
        T gain = static_cast<T>(std::sqrt(2.0 / (1.0 + static_cast<double>(a * a))));
        T bound = gain * static_cast<T>(std::sqrt(3.0 / static_cast<double>(fan_in)));
        return Tensor<T>::uniform(std::move(shape), -bound, bound, device);
    }

    template <typename T>
    template <typename U>
    requires std::is_floating_point_v<U>
    Tensor<T> Tensor<T>::kaiming_normal(std::vector<int64_t> shape, int64_t fan_in, T a, Device device) {
        T gain = static_cast<T>(std::sqrt(2.0 / (1.0 + static_cast<double>(a * a))));
        T std = gain / static_cast<T>(std::sqrt(static_cast<double>(fan_in)));
        return Tensor<T>::normal(std::move(shape), T(0), std, device);
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