#pragma once

#include "gradc/backend/op_types.hpp"
#include "gradc/backend/cpu/math_functors.hpp"
#include "cpu/apply.hpp"
#include "gradc/backend/cuda/cuda_math.hpp"
#include "../core/tensor.hpp"
#include "../core/types.hpp"

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
    inline void dispatch(Device device, BinaryOp op, Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right) {
        if (device.is_cpu()) {
            switch (op) {
                case BinaryOp::Add:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::Add<T>());
                    break;

                case BinaryOp::Sub:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::Sub<T>());
                    break;

                case BinaryOp::Mul:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::Mul<T>());
                    break;

                case BinaryOp::Div:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::Div<T>());
                    break;

                case BinaryOp::ReLUBackward:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::ReLUBackward<T>());
                    break;

                case BinaryOp::EqMask:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::EqMask<T>());
                    break;

                default:
                    throw std::runtime_error("Unsupported BinaryOp in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_binary_out_of_place(out, left, right, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, BinaryOpInPlace op, Tensor<T>& left, const Tensor<T>& right) {
        if (device.is_cpu()) {
            switch (op) {
                case BinaryOpInPlace::Add:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::Add<T>());
                    break;

                case BinaryOpInPlace::Sub:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::Sub<T>());
                    break;

                case BinaryOpInPlace::Mul:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::Mul<T>());
                    break;

                case BinaryOpInPlace::Div:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::Div<T>());
                    break;

                default:
                    throw std::runtime_error("Unsupported BinaryOpInPlace in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_binary_in_place(left, right, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, UnaryOp op, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case UnaryOp::Identity:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Identity<T>());
                    break;

                case UnaryOp::Exp:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Exp<T>());
                    break;

                case UnaryOp::Log:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Log<T>());
                    break;

                case UnaryOp::ReLU:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::ReLU<T>());
                    break;

                default:
                    throw std::runtime_error("Unsupported UnaryOp in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_unary_out_of_place(out, in, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, UnaryOpInPlace op, Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case UnaryOpInPlace::Exp:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::Exp<T>());
                    break;

                case UnaryOpInPlace::Log:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::Log<T>());
                    break;

                case UnaryOpInPlace::ReLU:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::ReLU<T>());
                    break;

                default:
                    throw std::runtime_error("Unsupported UnaryOpInPlace in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_unary_in_place(in, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, ReduceOp op, ReductionMetadata& reduction_metadata, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case ReduceOp::Sum:
                    CPUBackend::apply_reduction_operation(out, in, reduction_metadata, T(0), cpu_functors::RED::Sum<T>());
                    break;

                case ReduceOp::Mean:
                    CPUBackend::apply_reduction_operation(out, in, reduction_metadata, T(0), cpu_functors::RED::Sum<T>());
                    CPUBackend::apply_binary_in_place(out, Tensor<T>(static_cast<T>(reduction_metadata.reduced_vol), out.device()), cpu_functors::BIP::Div<T>());
                    break;
                
                case ReduceOp::Max:
                    CPUBackend::apply_reduction_operation(out, in, reduction_metadata, std::numeric_limits<T>::lowest(), cpu_functors::RED::Max<T>());
                    break;

                case ReduceOp::Min:
                    CPUBackend::apply_reduction_operation(out, in, reduction_metadata, std::numeric_limits<T>::max(), cpu_functors::RED::Min<T>());
                    break;

                default:
                    throw std::runtime_error("Unsupported ReduceOp in CPU Dispatcher.");
            }
        }

        else if (device.is_cuda()) {
            switch (op) {
                case ReduceOp::Sum:
                    CUDAMath::apply_reduction_operation(out, in, reduction_metadata, T(0), ReduceOp::Sum);
                    break;

                case ReduceOp::Mean:
                    CUDAMath::apply_reduction_operation(out, in, reduction_metadata, T(0), ReduceOp::Sum);
                    CUDAMath::apply_binary_in_place(out, Tensor<T>(static_cast<T>(reduction_metadata.reduced_vol), out.device()), BinaryOpInPlace::Div);
                    break;

                case ReduceOp::Max:
                    CUDAMath::apply_reduction_operation(out, in, reduction_metadata, std::numeric_limits<T>::lowest(), ReduceOp::Max);
                    break;

                case ReduceOp::Min:
                    CUDAMath::apply_reduction_operation(out, in, reduction_metadata, std::numeric_limits<T>::max(), ReduceOp::Min);
                    break;

                default:
                    throw std::runtime_error("Unsupported ReduceOp in CPU Dispatcher.");
            }
        }
    }

    template <typename T>
    inline void dispatch(Device device, ArgExtrOp op, int64_t dim, Tensor<int64_t>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            switch (op) {
                case ArgExtrOp::ArgMax:
                    CPUBackend::apply_arg_extr_operation(out, in, dim, std::numeric_limits<T>::lowest(), cpu_functors::ARGEXTR::ArgMax<T>());
                    break;
                case ArgExtrOp::ArgMin:
                    CPUBackend::apply_arg_extr_operation(out, in, dim, std::numeric_limits<T>::lowest(), cpu_functors::ARGEXTR::ArgMin<T>());
                    break;
            }
        }
        else if (device.is_cuda()) {
            switch (op) {
                case ArgExtrOp::ArgMax:
                    CUDAMath::apply_arg_extr_operation(out, in, dim, std::numeric_limits<T>::lowest(), ArgExtrOp::ArgMax);
                    break;
                case ArgExtrOp::ArgMin:
                    CUDAMath::apply_arg_extr_operation(out, in, dim, std::numeric_limits<T>::lowest(), ArgExtrOp::ArgMin);
                    break;
            }
        }
    }

    template <typename OutT, typename InT>
    inline void dispatch_cast(Device device, Tensor<OutT>& out, const Tensor<InT>& in) {
        if (device.is_cpu()) {
            CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Cast<InT, OutT>());
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_unary_out_of_place(out, in, UnaryOp::Cast);
        }
    }

}