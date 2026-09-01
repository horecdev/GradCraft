#pragma once

#include "gradc/core/tensor.hpp"
#include "gradc/backend/op_types.hpp"

namespace gradc {
    class CUDAMath {
        public:
            template <typename T>
            static void apply_binary_out_of_place(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, BinaryOp op);
            template <typename T>
            static void apply_binary_in_place(Tensor<T>& left, const Tensor<T>& right, BinaryOpInPlace op);
            template <typename T>
            static void apply_unary_out_of_place(Tensor<T>& out, const Tensor<T>& source, UnaryOp op);
            template <typename OutT, typename InT>
            static void apply_cast_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source);
            template <typename T>
            static void apply_unary_in_place(Tensor<T>& source, UnaryOpInPlace op);
            template <typename T>
            static void apply_reduction_operation(Tensor<T>& out, const Tensor<T>& source, const RedMeta& red_meta, ReduceOp op);
            template <typename T>
            static void apply_arg_extr_operation(Tensor<int64_t>& out, const Tensor<T>& source, int64_t dim, ArgExtrOp op);
            template <typename T> 
            requires std::is_floating_point_v<T>
            static void apply_normal_gemm(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const NMMMeta& blas_meta);
            template<typename T>
            requires std::is_floating_point_v<T>
            static void apply_batched_gemm(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const BMMMeta& blas_meta);
            template <typename T>
            static void apply_embed(Tensor<T>& out, const Tensor<int64_t>& indices, const Tensor<T>& embeds, int64_t embed_vol);
            template <typename T>
            static void apply_scatter_add(Tensor<T>& dembeds, const Tensor<int64_t>& indices, const Tensor<T>& out_grad, int64_t embed_vol);
            template <typename T>
            requires std::is_floating_point_v<T>
            static void apply_rmsnorm_forward(Tensor<T>& out, Tensor<T>& inv_rms, const Tensor<T>& parent, const Tensor<T>& gamma, const RedMeta& red_meta, T eps);
            template <typename T>
            requires std::is_floating_point_v<T>
            static void apply_rmsnorm_backward(Tensor<T>& dx, Tensor<T>& dgamma, const Tensor<T>& out_grad, const Tensor<T>& parent, const Tensor<T>& gamma, const Tensor<T>& inv_rms, const RedMeta& red_meta);
    };
}
