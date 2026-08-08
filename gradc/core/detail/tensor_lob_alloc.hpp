#pragma once

#include "../../backend/dispatcher.hpp"
#include "../tensor.hpp"
#include "../types.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T> lobotomized_contiguous_alloc(const Tensor<T>& source) {
        Tensor<T> result = Tensor<T>(source.m_shape, source.device(), uninitialized);
        dispatch(source.device(), UnaryOp::Identity, result, source);
        return result;
    }

    template <typename InT, typename OutT>
    Tensor<OutT> lobotomized_cast_alloc(const Tensor<InT>& source) {
        // source can have weird strides, but result is contiguous
        Tensor<OutT> result = Tensor<OutT>(source.m_shape, source.device(), uninitialized);
        dispatch_cast(source.device(), result, source);
        return result;
    }

    template <typename T>
    Tensor<T> lobotomized_concat_alloc(const std::vector<Tensor<T>>& tensor_list, int64_t concat_dim, const std::vector<int64_t>& final_shape) {
        // target_shape must be known and concat_dim must be normalized already.
        Device target_device = tensor_list[0].device();
        Tensor<T> result = Tensor<T>(final_shape, target_device, uninitialized);
        int64_t current_offset = 0;

        for (const Tensor<T>& parent : tensor_list) {
            std::vector<int64_t> view_shape = parent.m_shape;

            Tensor<T> chunk_view = Tensor<T>(std::move(view_shape), result.m_strides, current_offset, result.m_state->m_storage, false);
            // by keeping strides it makes apply func naturally skip indices for next tensors in list

            dispatch(target_device, UnaryOp::Identity, chunk_view, parent);

            current_offset += parent.m_shape[concat_dim] * result.m_strides[concat_dim];
            // if you have, say, 10, 5, 7 split into [10, 2, 7] and [10, 3, 7], increase offset, then you start writing at [0, 2, 0]. You never go back to [0, 0, 0] or [0, 1, 0]
            // or anything with the concat dim less than your current progress
            // It naturally slides the starting point to the next tensors destination. Then physically builds the tensor here, skipping over already filled place by prev tensors.
        }

        return result;
    }

    template <typename T>
    Tensor<T> lobotomized_stack_alloc(const std::vector<Tensor<T>>& tensor_list, int64_t stack_dim, const std::vector<int64_t>& final_shape) {

        Device target_device = tensor_list[0].device();
        Tensor<T> result = Tensor<T>(final_shape, target_device, uninitialized);
        int64_t current_offset = 0;

        for (const Tensor<T>& parent : tensor_list) {
            Tensor<T> unsqueezed_parent = lobotomized_unsqueeze_view(parent, stack_dim);
            std::vector<int64_t> view_shape = unsqueezed_parent.m_shape;

            Tensor<T> chunk_view = Tensor<T>(std::move(view_shape), result.m_strides, current_offset, result.m_state->m_storage, false);

            dispatch(target_device, UnaryOp::Identity, chunk_view, unsqueezed_parent);

            current_offset += result.m_strides[stack_dim];
        }

        return result;
    }

}