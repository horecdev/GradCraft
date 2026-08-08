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
    template <typename T>
    inline Device infer_assert_device(const std::vector<const Tensor<T>>& tensors) {
        if (tensors.empty()) {
            throw std::runtime_error("Tried inferring device from an empty Tensor list.");
        }
        // supposes that input vector is NOT empty
        Device target_device = *(tensors[0]).device();
        for (int64_t i = 1; i < std::ssize(tensors); ++i) {
            if (*(tensors[i]).device() != target_device) {
                throw std::runtime_error("Operation failed: all (2+) Tensors must be on the same device.");
            }
        }
        return target_device;
    }

    template <typename T, typename... Args>
    inline Device infer_assert_device(const Tensor<T>& first, const Args&... rest) {
        Device target_device = first.device();
        bool all_match = ((rest.device() == target_device) && ...); // compare each thing in bucket "rest"
        if (all_match == false) {
            throw std::runtime_error("Operation failed: all tensors have to live on the same device.");
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

                case BinaryOp::EqMask:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::EqMask<T>());
                    break;

                case BinaryOp::BExp:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BExp<T>());
                    break;

                case BinaryOp::BLog:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BLog<T>());
                    break;

                case BinaryOp::BSin:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BSin<T>());
                    break;
                
                case BinaryOp::BCos:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BCos<T>());
                    break;

                case BinaryOp::BSquare:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BSquare<T>());
                    break;

                case BinaryOp::BReLU:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BReLU<T>());
                    break;

                case BinaryOp::BSigmoid:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BSigmoid<T>());
                    break;

                case BinaryOp::BTanH:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BTanH<T>());
                    break;

                case BinaryOp::BSiLU:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BSiLU<T>());
                    break;

                case BinaryOp::BGeLU:
                    CPUBackend::apply_binary_out_of_place(out, left, right, cpu_functors::BOOP::BGeLU<T>());
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

                case BinaryOpInPlace::ISub: // left: to_be_subbed. right: main
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::ISub<T>());
                    break;

                case BinaryOpInPlace::Mul:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::Mul<T>());
                    break;

                case BinaryOpInPlace::Div:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::Div<T>());
                    break;

                case BinaryOpInPlace::IDiv:
                    CPUBackend::apply_binary_in_place(left, right, cpu_functors::BIP::IDiv<T>());
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

                case UnaryOp::Sin:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Sin<T>());
                    break;

                case UnaryOp::Cos:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Cos<T>());
                    break;

                case UnaryOp::Square:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Square<T>());
                    break;

                case UnaryOp::ReLU:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::ReLU<T>());
                    break;

                case UnaryOp::Sigmoid:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::Sigmoid<T>());
                    break;

                case UnaryOp::TanH:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::TanH<T>());
                    break;

                case UnaryOp::SiLU:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::SiLU<T>());
                    break;

                case UnaryOp::GeLU:
                    CPUBackend::apply_unary_out_of_place(out, in, cpu_functors::UOOP::GeLU<T>());
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

                case UnaryOpInPlace::Sin:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::Sin<T>());
                    break;

                case UnaryOpInPlace::Cos:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::Cos<T>());
                    break;

                case UnaryOpInPlace::Square:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::Square<T>());
                    break;

                case UnaryOpInPlace::ReLU:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::ReLU<T>());
                    break;

                case UnaryOpInPlace::Sigmoid:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::Sigmoid<T>());
                    break;

                case UnaryOpInPlace::TanH:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::TanH<T>());
                    break;

                case UnaryOpInPlace::SiLU:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::SiLU<T>());
                    break;

                case UnaryOpInPlace::GeLU:
                    CPUBackend::apply_unary_in_place(in, cpu_functors::UIP::GeLU<T>());
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

    template <typename T>
    inline void dispatch_batched_gemm(Device device, Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const BLASGEMMMeta& blas_meta) {
        if (device.is_cpu()) {
            CPUBackend::apply_batched_gemm(out, left, right, blas_meta);
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_batched_gemm(out, left, right, blas_meta);
        }
    }
}