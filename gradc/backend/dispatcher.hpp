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
    inline void dispatch(Device device, ReduceOp op, RedMeta& red_meta, Tensor<T>& out, const Tensor<T>& in) {
        if (device.is_cpu()) {
            cpu_mapper::map_red<T>(op, [&](auto functor, T init_value) {CPUBackend::apply_reduction_operation(out,in, red_meta, init_value, functor);});
        }

        else if (device.is_cuda()) {
            CUDAMath::apply_reduction_operation(out, in, red_meta, op);
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
    inline void dispatch_normal_gemm(Device device, Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const NMMMeta& blas_meta) {
        if (device.is_cpu()) {
            CPUBackend::apply_normal_gemm(out, left, right, blas_meta);
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_normal_gemm(out, left, right, blas_meta);
        }
    }

    template <typename T>
    inline void dispatch_batched_gemm(Device device, Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const BMMMeta& blas_meta) {
        if (device.is_cpu()) {
            CPUBackend::apply_batched_gemm(out, left, right, blas_meta);
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_batched_gemm(out, left, right, blas_meta);
        }
    }

    template <typename T>
    inline void dispatch_embed(Device device, Tensor<T>& out, const Tensor<int64_t>& indices, const Tensor<T>& embeds, int64_t embed_vol) {
        if (device.is_cpu()) {
            CPUBackend::apply_embed(out, indices, embeds, embed_vol);
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_embed(out, indices, embeds, embed_vol);
        }
    }

    template <typename T>
    inline void dispatch_scatter_add(Device device, Tensor<T>& dembeds, const Tensor<int64_t>& indices, const Tensor<T>& out_grad, int64_t embed_vol) {
        if (device.is_cpu()) {
            CPUBackend::apply_scatter_add(dembeds, indices, out_grad, embed_vol);
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_scatter_add(dembeds, indices, out_grad, embed_vol);
        }
    }

    template <typename T>
    inline void dispatch_cudnn_layernorm_forward(Device device, Tensor<T>& out, Tensor<T>& saved_mean, Tensor<T>& saved_inv_var, const Tensor<T>& x, const Tensor<T>& gamma, const Tensor<T>& beta, T eps, bool save_intermediates) {
        if (device.is_cpu()) {
            throw std::runtime_error("cuDNN LayerNorm forward dispatch called on CPU");
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_cudnn_layernorm_forward(out, saved_mean, saved_inv_var, x, gamma, beta, eps, save_intermediates);
        }
    }

    template <typename T>
    inline void dispatch_cudnn_layernorm_backward(Device device, Tensor<T>& dx, Tensor<T>& dgamma, Tensor<T>& dbeta, const Tensor<T>& out_grad, const Tensor<T>& x, const Tensor<T>& gamma, const Tensor<T>& saved_mean, const Tensor<T>& saved_inv_var) {
        if (device.is_cpu()) {
            throw std::runtime_error("cuDNN LayerNorm backward dispatch called on CPU");
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_cudnn_layernorm_backward(dx, dgamma, dbeta, out_grad, x, gamma, saved_mean, saved_inv_var);
        }
    }

    template <typename T>
    inline void dispatch_cudnn_sdpa_forward(Device device, Tensor<T>& out, Tensor<T>& saved_lse, const Tensor<T>& q, const Tensor<T>& k, const Tensor<T>& v, T scale, bool is_causal, bool save_intermediates) {
        if (device.is_cpu()) {
            throw std::runtime_error("cuDNN SDPA forward dispatch called on CPU");
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_cudnn_sdpa_forward(out, saved_lse, q, k, v, scale, is_causal, save_intermediates);
        }
    } 

    template <typename T>
    inline void dispatch_cudnn_sdpa_backward(Device device, Tensor<T>& dq, Tensor<T>& dk, Tensor<T>& dv, const Tensor<T>& out_grad, const Tensor<T>& q, const Tensor<T>& k, const Tensor<T>& v, const Tensor<T> out, const Tensor<T>& saved_lse, T scale, bool is_causal) {
        if (device.is_cpu()) {
            throw std::runtime_error("cuDNN SDPA backward dispatch called on CPU");
        }
        else if (device.is_cuda()) {
            CUDAMath::apply_cudnn_sdpa_backward(dq, dk, dv, out_grad, q, k, v, out, saved_lse, scale, is_causal);
        }
    }

    
}