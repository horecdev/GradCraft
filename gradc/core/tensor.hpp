#pragma once

#include "storage.hpp"
#include "types.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gradc {
    struct TensorStateBase;
    template <typename T> struct TensorState;

    class CPUBackend;
    class CUDAMath;
    class CUDAUtils;
    
    template <typename T>
    class Tensor {
        static_assert(is_supported_tensor_type_v<T>, "FATAL: Attempted creating a Tensor with unsupported data type.");

        template <typename U> friend class Tensor; // otherwise Tensor A cant look into T. B
        template <typename U> friend class Node;

        private:
            std::vector<int64_t> m_shape;
            std::vector<int64_t> m_strides;
            int64_t m_offset;
            std::shared_ptr<TensorState<T>> m_state;
            bool m_requires_grad;

            static bool validate_requires_grad(bool requires_grad) {
                if constexpr (!std::is_floating_point_v<T>) {
                    if (requires_grad) {
                        throw std::runtime_error("FATAL: Cannot require gradients on non-floating Tensors.");
                    }
                }
                return requires_grad;
            }

        public:
            void realize();

            void backward();

            void accumulate_grad(const Tensor<T>& incoming_grad, bool is_sub = false);
            void accumulate_grad_matmul(const Tensor<T>& left, const Tensor<T>& right, BLASGEMMMeta& blas_meta, const std::vector<int64_t>& orig_shape) requires std::is_floating_point_v<T>;

            void zero_grad();

            bool is_exclusive();

            Tensor detach();
            

            // LIFECYCLE 
            Tensor();
            explicit Tensor(T value, Device device = Device(DeviceType::CPU));
            explicit Tensor(std::vector<int64_t> shape, T init_val = T(0), Device device = Device(DeviceType::CPU));
            explicit Tensor(std::vector<int64_t> shape, Device device = Device(DeviceType::CPU), UninitializedTag = uninitialized);

            Tensor(std::initializer_list<int64_t> shape, T init_val = T(0), Device device = Device(DeviceType::CPU));
            Tensor(std::vector<int64_t> shape, bool requires_grad, LazyTag, Device device = Device(DeviceType::CPU));
            Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<Storage<T>> storage, bool requires_grad);
            Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<TensorState<T>> state, bool requires_grad);
            ~Tensor();
            Tensor(const Tensor& source);
            Tensor(Tensor&& source);
            Tensor& operator=(const Tensor& source);
            Tensor& operator=(Tensor&& source);

            static Tensor zeros(std::vector<int64_t> shape, Device device = Device(DeviceType::CPU));
            static Tensor ones(std::vector<int64_t> shape, Device device = Device(DeviceType::CPU));
            static Tensor full(std::vector<int64_t> shape, T fill_val, Device device = Device(DeviceType::CPU));
            static Tensor arange(T start, T stop, T step, Device device = Device(DeviceType::CPU));

            
            static Tensor normal(std::vector<int64_t> shape, T mean = T(0), T std = T(1), Device device = Device(DeviceType::CPU)) requires std::is_floating_point_v<T>;
            static Tensor uniform(std::vector<int64_t> shape, T low, T high, Device device = Device(DeviceType::CPU)) requires std::is_floating_point_v<T>;
            static Tensor xavier_uniform(std::vector<int64_t> shape, int64_t fan_in, int64_t fan_out, Device device) requires std::is_floating_point_v<T>;
            static Tensor xavier_normal(std::vector<int64_t> shape, int64_t fan_in, int64_t fan_out, Device device) requires std::is_floating_point_v<T>;
            static Tensor kaiming_uniform(std::vector<int64_t> shape, int64_t fan_in, T a, Device device) requires std::is_floating_point_v<T>;
            static Tensor kaiming_normal(std::vector<int64_t> shape, int64_t fan_in, T a, Device device) requires std::is_floating_point_v<T>;
            
            Tensor clone() const;

            // INDEXING
            template <typename... Args>
            Tensor operator[](Args... args) const;

            T item() const;

            // GETTERS

            const std::vector<int64_t>& shape() const {return m_shape;}

            const std::vector<int64_t>& strides() const {return m_strides;}

            int64_t offset() const {return m_offset;}

            bool requires_grad() const {return m_requires_grad;}

            Device device() const {return m_state->m_storage->device();}

            TensorStateBase* _get_state_base() const {
                return m_state.get();
            }
            
            std::shared_ptr<TensorState<T>> _get_state() const {
                return m_state;
            }

            std::shared_ptr<Storage<T>> _get_storage() const;

            std::optional<Tensor<T>> grad() const;

            DType dtype() const {return type_to_dtype<T>();}

            int64_t volume() const {
                int64_t vol = 1;
                for (int64_t dim : m_shape) {
                    vol *= dim;
                }
                return vol;
            }

            // SETTERS
            void set_data(std::initializer_list<T> data);
            
            Tensor& set_requires_grad(bool value) {
                m_requires_grad = validate_requires_grad(value);
                return *this;
            }

            // SHAPES
            bool is_contiguous() const {
                if (m_shape.empty()) return true;
                if (m_strides[std::ssize(m_strides) - 1] != 1) {
                    return false;
                }
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    if (m_strides[i - 1] != m_shape[i] * m_strides[i]) {
                        return false;
                    }
                }
                return true;
            }

            Tensor contiguous() const;
            Tensor transpose(const int64_t dim0, const int64_t dim1) const;
            Tensor permute(const std::vector<int64_t>& axes) const;
            Tensor reshape(const std::vector<int64_t>& target_shape) const;
            
            template <typename U> friend Tensor<T> lazy_concat(std::vector<Tensor<T>> &tensor_list, int64_t concat_dim);

            // NN
            Tensor relu() const;

            Tensor sigmoid() const requires std::is_floating_point_v<T>;
            Tensor tanh() const requires std::is_floating_point_v<T>;
            Tensor silu() const requires std::is_floating_point_v<T>;
            Tensor gelu() const requires std::is_floating_point_v<T>;
            
            // BACKEND
            friend CPUBackend;
            friend CUDAUtils;
            friend CUDAMath;

            // MATH 
            Tensor operator-() const;
            Tensor exp() const requires std::is_floating_point_v<T>;
            Tensor log() const requires std::is_floating_point_v<T>;

            template <typename U, typename W> friend auto operator+(Tensor<U> left, Tensor<W> right); // we befriend whole family of functions named operator+. 
            template <typename U, typename W> friend auto operator-(Tensor<U> left, Tensor<W> right);
            template <typename U, typename W> friend auto operator*(Tensor<U> left, Tensor<W> right); // It operates on type U and U can be virtually anything
            template <typename U, typename W> friend auto operator/(Tensor<U> left, Tensor<W> right);

            template <typename U, typename W> friend Tensor<U>& operator+=(Tensor<U>& main, Tensor<W> other);
            template <typename U, typename W> friend Tensor<U>& operator-=(Tensor<U>& main, Tensor<W> other);
            template <typename U, typename W> friend Tensor<U>& operator*=(Tensor<U>& main, Tensor<W> other);
            template <typename U, typename W> friend Tensor<U>& operator/=(Tensor<U>& main, Tensor<W> other);

            template <typename U>
            friend auto infer_blas_meta(Tensor<U> left, Tensor<U> right, bool accumulate);

            template <typename U, typename W>
            requires (std::is_floating_point_v<U> && std::is_floating_point_v<W>)
            friend Tensor<U> matmul(Tensor<U> left, Tensor<W> right);


            // REDUCTIONS

            Tensor sum(const std::vector<int64_t>& axes, bool keepdims) const;
            template <typename OutT = std::conditional_t<std::is_integral_v<T>, float, T>> // if int: float. Otherwise T.
            Tensor<OutT> mean(const std::vector<int64_t>& axes, bool keepdims) const;
            Tensor max(const std::vector<int64_t>& red_axes, bool keepdims) const;
            Tensor<int64_t> argmax(int64_t dim, bool keepdims);

            // UTILS
            template <typename U> friend Tensor<U> lobotomized_broadcast_view(const Tensor<U>& source, const std::vector<int64_t>& target_shape);
            template <typename U> friend Tensor<U> lobotomized_contiguous_alloc(const Tensor<U>& source);
            template <typename U> friend Tensor<U> lobotomized_transpose_view(const Tensor<U>& source, int64_t dim0, int64_t dim1);
            template <typename U> friend Tensor<U> lobotomized_reshape_view(const Tensor<U>& source, const std::vector<int64_t>& target_shape);
            template <typename U> friend Tensor<U> lobotomized_permute_view(const Tensor<U>& source, const std::vector<int64_t>& axes);
            template <typename InT, typename OutT> friend Tensor<OutT> lobotomized_cast_alloc(const Tensor<InT>& source);
            template <typename U> friend Tensor<U> create_lobotomized_slice_view(const Tensor<U>& source, const std::vector<IndexDesc>& descriptors);
            template <typename U> friend Tensor<U> lobotomized_concat_alloc(const std::vector<Tensor<U>>& tensor_list, int64_t concat_dim, const std::vector<int64_t>& final_shape);
            template <typename U> friend std::ostream& print_tensor(std::ostream& stream, const Tensor<U>& source, PrintOptions opts);
            template <typename U> friend void print_dim(std::ostream& stream, const Tensor<U>& source, const PrintOptions& opts, int64_t current_dim, int64_t base_offset, bool is_last);
            template <typename U> friend Tensor<U> unbroadcast_grad(const Tensor<U>& raw_grad, const std::vector<int64_t>& orig_shape);
            template <typename U> friend Tensor<U> lazy_concat(std::vector<Tensor<U>>& tensor_list, int64_t concat_dim);

            template <typename TargetT> Tensor<TargetT> cast() const;

            Tensor to(Device device) const;

    };      
}

#include "tensor_state.hpp"

namespace gradc { // here is what dereferences m_state, since before its created it has no idea how big it is and what variables it has
    template <typename T>
    void Tensor<T>::realize() {
        if (m_state->m_creation_op != nullptr && m_state->m_is_realized != true) { // is not leaf and wasnt realized yet
            Tensor computed_result = m_state->m_creation_op->realize();
            if (m_state->m_storage != computed_result.m_state->m_storage) {
                std::swap(m_state->m_storage->m_data, computed_result.m_state->m_storage->m_data);
            }
        }

        m_state->m_is_realized = true; // so we dont realize() twice (reflected across multiple aliases)
    }

    template <typename T>
    bool Tensor<T>::is_exclusive() {
        return m_state.use_count() == 1 && m_state->m_storage.use_count() == 1;
    }

    template <typename T>
    std::shared_ptr<Storage<T>> Tensor<T>::_get_storage() const {
        return m_state->m_storage;
    }

    template <typename T>
    std::optional<Tensor<T>> Tensor<T>::grad() const {
        return m_state->m_grad;
    }

    template <typename T>
    void Tensor<T>::set_data(std::initializer_list<T> data) {
        if (std::ssize(data) != this->_get_storage()->size()) {
            throw std::runtime_error("set_data must set data for the entire underlying storage.");
        }

        T* raw_ptr = m_state->m_storage->data();
        int64_t bytes = sizeof(T) * std::ssize(data);
        Device dev = this->device();

        if (dev.is_cpu()) {
            std::memcpy(raw_ptr, data.begin(), bytes);
        }
        else if (dev.is_cuda()) {
            cudaSetDevice(dev.index);
            cudaMemcpy(raw_ptr, data.begin(), bytes, cudaMemcpyHostToDevice);
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::detach() {
        Tensor<T> result = Tensor<T>(m_shape, m_strides, m_offset, this->_get_storage(), false); // no grad, no creation_op. Lobotomized.
        return result;
    }
}



   