#pragma once

#include "gradc/core/tensor.hpp"
#include "gradc/backend/op_types.hpp"

namespace gradc {
    class CUDAMath {
        public:
            template <typename T>
            static void apply_binary_out_of_place(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, BinaryOp op);
    };
}
