#pragma once

#include "core/print.hpp" // IWYU pragma: keep
#include "core/tensor.hpp" // IWYU pragma: keep
#include "core/types.hpp" // IWYU pragma: keep

#include "frontend/tensor_activation.hpp" // IWYU pragma: keep
#include "frontend/tensor_indexing.hpp" // IWYU pragma: keep
#include "frontend/tensor_lifecycle.hpp" // IWYU pragma: keep
#include "frontend/tensor_math.hpp" // IWYU pragma: keep
#include "frontend/tensor_memory.hpp" // IWYU pragma: keep
#include "frontend/tensor_reduction.hpp" // IWYU pragma: keep
#include "frontend/tensor_shape.hpp" // IWYU pragma: keep

#include "graph/autograd_engine.hpp" // IWYU pragma: keep

namespace gradc {
    template class Tensor<float>;
    template class Tensor<double>;
    template class Tensor<int32_t>;
    template class Tensor<int64_t>;
}