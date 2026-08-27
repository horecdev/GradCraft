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
#include "frontend/tensor_normalization.hpp" // IWYU pragma: keep
#include "frontend/tensor_loss.hpp" // IWYU pragma: keep
#include "frontend/tensor_other.hpp" // IWYU pragma: keep

#include "graph/autograd_engine.hpp" // IWYU pragma: keep

#include "nn/base/module.hpp" // IWYU pragma: keep
#include "nn/base/parameter.hpp" // IWYU pragma: keep
#include "nn/layers/linear.hpp" // IWYU pragma: keep
#include "nn/layers/normalization.hpp" // IWYU pragma: keep
#include "nn/layers/other.hpp" // IWYU pragma: keep
#include "nn/optim/adam.hpp" // IWYU pragma: keep
#include "nn/optim/optimizer.hpp" // IWYU pragma: keep
#include "nn/optim/rmsprop.hpp" // IWYU pragma: keep
#include "nn/optim/sgd.hpp" // IWYU pragma: keep
#include "nn/utils/checkpoint.hpp" // IWYU pragma: keep


namespace gradc {
    template class Tensor<float>;
    template class Tensor<double>;
    template class Tensor<int32_t>;
    template class Tensor<int64_t>;
}