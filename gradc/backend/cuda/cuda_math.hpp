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
            template <typename OutT, typename InT>
            static void apply_unary_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source, UnaryOp op);
            template <typename T>
            static void apply_unary_in_place(Tensor<T>& source, UnaryOpInPlace op);
            template <typename T>
            static void apply_reduction_operation(Tensor<T>& out, const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, ReduceOp op);
            template <typename T>
            static void apply_arg_extr_operation(Tensor<int64_t>& out, const Tensor<T>& source, int64_t dim, T init_value, ArgExtrOp op);
            template <typename T> 
            static void apply_batched_gemm(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const BLASGEMMMeta& blas_meta);
    };
}
