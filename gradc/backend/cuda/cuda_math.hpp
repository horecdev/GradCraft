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
            template <typename OutT, typename InT, typename Func>
            static void apply_unary_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source, Func op);
    };
}
