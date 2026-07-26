#pragma once

#include "op_types.hpp"
#include "cpu/apply.hpp"
#include "../core/tensor.hpp"
#include "../core/types.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gradc {
    template <typename T1, typename T2>
    inline Device infer_assert_device(const Tensor<T1>& t1, const Tensor<T2>& t2) {
        if (t1.device() != t2.device()) {
            throw std::runtime_error("Operation failed: both (2) Tensors must be on the same device.");
        }
        return t1.device();
    }


    template <typename T>
    inline Device infer_assert_device(const std::vector<Tensor<T>>& tensors) {
        if (tensors.empty()) {
            throw std::runtime_error("Tried inferring device from an empty Tensor list.");
        }
        // supposes that input vector is NOT empty
        Device target_device = tensors[0].device();
        for (int64_t i = 1; i < std::ssize(tensors); ++i) {
            if (tensors[i].device() != target_device) {
                throw std::runtime_error("Operation failed: all (2+) Tensors must be on the same device.");
            }
        }

        return target_device;
    }

    template <typename T>
    inline void dispatch(Device device, UnaryOp op, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case UnaryOp::Identity:
                    CPUBackend::apply_unary_out_of_place(out, in, [](T a){return a;});
                    break;

                case UnaryOp::Exp:
                    CPUBackend::apply_unary_out_of_place(out, in, [](T a){return std::exp(a);});
                    break;

                case UnaryOp::Log:
                    CPUBackend::apply_unary_out_of_place(out, in, [](T a){return std::log(a);});
                    break;

                case UnaryOp::ReLU:
                    CPUBackend::apply_unary_out_of_place(out, in, [](T a){return a > 0 ? a : 0;});
                    break;

                default:
                    throw std::runtime_error("Unsupported UnaryOp in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            // same but for CUDA
        }
    }

    template <typename T>
    inline void dispatch(Device device, UnaryOpInPlace op, Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case UnaryOpInPlace::Exp:
                    CPUBackend::apply_unary_in_place(in, [](T& a){a = std::exp(a);});
                    break;

                case UnaryOpInPlace::Log:
                    CPUBackend::apply_unary_in_place(in, [](T& a){a = std::log(a);});
                    break;

                case UnaryOpInPlace::ReLU:
                    CPUBackend::apply_unary_in_place(in, [](T& a){a = a > 0 ? a : 0;});
                    break;

                default:
                    throw std::runtime_error("Unsupported UnaryOpInPlace in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            // same but for CUDA
        }
    }

    template <typename T>
    inline void dispatch(Device device, BinaryOp op, Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right) {
        if (device.is_cpu()) {
            switch (op) {
                case BinaryOp::Add:
                    CPUBackend::apply_binary_out_of_place(out, left, right, [](T a, T b){return a + b;});
                    break;

                case BinaryOp::Sub:
                    CPUBackend::apply_binary_out_of_place(out, left, right, [](T a, T b){return a - b;});
                    break;

                case BinaryOp::Mul:
                    CPUBackend::apply_binary_out_of_place(out, left, right, [](T a, T b){return a * b;});
                    break;

                case BinaryOp::Div:
                    CPUBackend::apply_binary_out_of_place(out, left, right, [](T a, T b){return a / b;});
                    break;

                case BinaryOp::ReLUBackward:
                    CPUBackend::apply_binary_out_of_place(out, left, right, [](T grad, T value){return value > 0 ? grad : 0;});
                    break;

                default:
                    throw std::runtime_error("Unsupported BinaryOp in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            // same but for CUDA
        }
    }

    template <typename T>
    inline void dispatch(Device device, BinaryOpInPlace op, Tensor<T>& left, const Tensor<T>& right) {
        if (device.is_cpu()) {
            switch (op) {
                case BinaryOpInPlace::Add:
                    CPUBackend::apply_binary_in_place(left, right, [](T& a, T b){a += b;});
                    break;

                case BinaryOpInPlace::Sub:
                    CPUBackend::apply_binary_in_place(left, right, [](T& a, T b){a -= b;});
                    break;

                case BinaryOpInPlace::Mul:
                    CPUBackend::apply_binary_in_place(left, right, [](T& a, T b){a *= b;});
                    break;

                case BinaryOpInPlace::Div:
                    CPUBackend::apply_binary_in_place(left, right, [](T& a, T b){a /= b;});
                    break;

                default:
                    throw std::runtime_error("Unsupported BinaryOpInPlace in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            // same but for CUDA
        }
    }

    template <typename T>
    inline void dispatch(Device device, ReduceOp op, ReductionMetadata& reduction_metadata, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case ReduceOp::Sum:
                    CPUBackend::apply_reduction_operation(out, in, reduction_metadata, T(), [](T a, T b){return a + b;});
                    break;

                case ReduceOp::Mean:
                    CPUBackend::apply_reduction_operation(out, in, reduction_metadata, T(), [](T a, T b){return a + b;});
                    CPUBackend::apply_binary_in_place(out, Tensor<T>(static_cast<T>(reduction_metadata.reduced_vol), out.device()), [](T& a, T b){a /= b;});
                    break;

                default:
                    throw std::runtime_error("Unsupported ReduceOp in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            // same but for CUDA
        }
    }

    template <typename OutT, typename InT>
    inline void dispatch_cast(Device device, Tensor<OutT>& out, const Tensor<InT>& in) {
        if (device.is_cpu()) {
            CPUBackend::apply_unary_out_of_place(out, in, [](InT a){return static_cast<OutT>(a);});
        }
        else if (device.is_cuda()) {}
    }

}