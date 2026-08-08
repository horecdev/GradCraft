#pragma once

#include "gradc/backend/op_types.hpp"
#include "gradc/backend/cpu/math_functors.hpp"
#include "gradc/backend/cpu/function_mapper.hpp"
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
            cpu_mapper::map_boop<T>(op, [&](auto functor) {CPUBackend::apply_binary_out_of_place(out, left, right, functor);});
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_binary_out_of_place(out, left, right, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, BinaryOpInPlace op, Tensor<T>& left, const Tensor<T>& right) {
        if (device.is_cpu()) {
            cpu_mapper::map_bip<T>(op, [&](auto functor) {CPUBackend::apply_binary_in_place(left, right, functor);});
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_binary_in_place(left, right, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, UnaryOp op, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            cpu_mapper::map_uoop<T>(op, [&](auto functor) {CPUBackend::apply_unary_out_of_place(out, in, functor);});
        }
        
        else if (device.is_cuda()) {
            CUDAMath::apply_unary_out_of_place(out, in, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, UnaryOpInPlace op, Tensor<T>& in) {
        if (device.is_cpu()) {
            cpu_mapper::map_uip<T>(op, [&](auto functor) {CPUBackend::apply_unary_in_place(in, functor);});
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_unary_in_place(in, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, ReduceOp op, ReductionMetadata& reduction_metadata, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            cpu_mapper::map_red<T>(op, [&](auto functor, T init_value) {CPUBackend::apply_reduction_operation(out,in, reduction_metadata, init_value, functor);});
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_reduction_operation(out, in, reduction_metadata, op);
        }
    }

    template <typename T>
    inline void dispatch(Device device, ArgExtrOp op, int64_t dim, Tensor<int64_t>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            cpu_mapper::map_argextr<T>(op, [&](auto functor, T init_value) {CPUBackend::apply_arg_extr_operation(out, in, dim, init_value, functor);});
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_arg_extr_operation(out, in, dim, op);
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