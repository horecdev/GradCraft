#pragma once

#include "../../core/tensor.hpp"
#include "../../core/detail/tensor_lob_view.hpp"
#include "../../core/detail/shape_inference.hpp"

namespace gradc {

    class CPUBackend {
        public:
        // RULE: EVERY OUT TENSOR IS MEMORY ALLOCATED. SUPPOSED TO NOT BE INITIALIZED EXCEPT IN-PLACE OPPS. OUT IS OR ISNT CONTIGUOUS (RESPECT OFFSET, STRIDES)
        // UNARY/BINARY OUTOF/IN APPLY FOR IT. REDUCE OP EXPECTS OUT TO BE CONTIGUOUS
        template <typename T, typename Func>
        static void apply_binary_out_of_place(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, Func op) {
            if (std::ssize(out.m_shape) == 0) { // op on 2 scalars
                (out.m_state->m_storage->m_data)[out.m_offset] = op((left.m_state->m_storage->m_data)[left.m_offset], (right.m_state->m_storage->m_data)[right.m_offset]);
                return;
            }

            if (out.m_shape == left.m_shape && left.m_shape == right.m_shape && out.is_contiguous() && left.is_contiguous() && right.is_contiguous()) {
                // fast path
                int64_t total_size = out.volume();

                T* __restrict p_out = out._get_storage()->data() + out.m_offset;
                const T* __restrict p_left = left._get_storage()->data() + left.m_offset;
                const T* __restrict p_right = right._get_storage()->data() + right.m_offset;
                
                for (int64_t i = 0; i < total_size; ++i) {
                    p_out[i] = op(p_left[i], p_right[i]);
                }
                return;
            }

            std::vector<int64_t>* left_strides = &left.m_strides;
            std::vector<int64_t>* right_strides = &right.m_strides;
            int64_t left_offset = left.m_offset;
            int64_t right_offset = right.m_offset; 
            std::shared_ptr<Storage<T>> left_storage = left.m_state->m_storage; // broadcasting does not alter memory
            std::shared_ptr<Storage<T>> right_storage = right.m_state->m_storage;

            Tensor<T> broad_right;
            Tensor<T> broad_left;

            if (left.m_shape != out.m_shape) {
                broad_left = lobotomized_broadcast_view(left, out.m_shape);

                left_strides = &broad_left.m_strides;
                left_offset = broad_left.m_offset;
            }
            if (right.m_shape != out.m_shape) {
                broad_right = lobotomized_broadcast_view(right, out.m_shape);

                right_strides = &broad_right.m_strides;
                right_offset = broad_right.m_offset;
            }

            FusedView fused = fuse_dimensions(out.m_shape, {&out.m_strides, left_strides, right_strides});
            std::vector<int64_t>* out_strides = &fused.strides[0] ;
            left_strides = &fused.strides[1];
            right_strides = &fused.strides[2];

            const int64_t n_dim = std::ssize(fused.shared_shape);
            std::vector<int64_t> odometer(n_dim, 0);
            while (odometer[0] < fused.shared_shape[0]) {
                int64_t left_strided_idx = left_offset; 
                int64_t right_strided_idx = right_offset;
                int64_t out_strided_idx = out.m_offset;

                for (int64_t i = 0; i < n_dim; ++i) {
                    left_strided_idx += odometer[i] * (*left_strides)[i];
                    right_strided_idx += odometer[i] * (*right_strides)[i];
                    out_strided_idx += odometer[i] * (*out_strides)[i];
                }
                (out.m_state->m_storage->m_data)[out_strided_idx] = op((left_storage->m_data)[left_strided_idx], (right_storage->m_data)[right_strided_idx]); // copied straight into CPU registers from RAM
                ++odometer[n_dim - 1];
                int64_t i = n_dim - 1;
                while ((odometer[i] == fused.shared_shape[i]) && i > 0) {
                    odometer[i] = 0;
                    ++odometer[i - 1];
                    --i;
                }
            }
        }

        template <typename T, typename Func> // two types so you can cast one to another in lambda
        static void apply_binary_in_place(Tensor<T>& left, const Tensor<T>& right, Func op) { 
            if (left.m_shape.empty() && right.m_shape.empty()) {
                op(left.m_state->m_storage->m_data[left.m_offset], right.m_state->m_storage->m_data[right.m_offset]);
                return;
            }

            if (left.m_shape == right.m_shape && left.is_contiguous() && right.is_contiguous()) {
                // fast path
                int64_t total_size = left.volume();

                T* __restrict p_left = left._get_storage()->data() + left.m_offset;
                const T* __restrict p_right = right._get_storage()->data() + right.m_offset;
                
                for (int64_t i = 0; i < total_size; ++i) {
                    op(p_left[i], p_right[i]);
                }
                return;
            }

            std::vector<int64_t>* right_strides;
            int64_t right_offset;

            Tensor<T> broad_right;
            if (left.m_shape == right.m_shape) {
                right_strides = &right.m_strides;
                right_offset = right.m_offset;
            }
            else {
                broad_right = lobotomized_broadcast_view(right, left.m_shape);

                right_strides = &broad_right.m_strides;
                right_offset = broad_right.m_offset;
            }

            FusedView fused = fuse_dimensions(left.m_shape, {&left.m_strides, right_strides});
            std::vector<int64_t>* left_strides = &fused.strides[0];
            right_strides = &fused.strides[1];

            const int64_t n_dim = std::ssize(fused.shared_shape);
            std::vector<int64_t> odometer(n_dim, 0);
            while (odometer[0] < fused.shared_shape[0]) {
                int64_t left_strided_idx = left.m_offset; 
                int64_t right_strided_idx = right_offset;

                for (int64_t i = 0; i < n_dim; ++i) {
                    left_strided_idx += odometer[i] * (*left_strides)[i];
                    right_strided_idx += odometer[i] * (*right_strides)[i];
                }
                op((left.m_state->m_storage->m_data)[left_strided_idx], (right.m_state->m_storage->m_data)[right_strided_idx]);
                ++odometer[n_dim - 1];
                int64_t i = n_dim - 1;
                while ((odometer[i] == fused.shared_shape[i]) && i > 0) {
                    odometer[i] = 0;
                    ++odometer[i - 1];
                    --i;
                }
            }
        }

        template <typename OutT, typename InT, typename Func>
        static void apply_unary_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source, Func op) {
            if (source.m_shape.empty()) {
                (out.m_state->m_storage->m_data)[out.m_offset] = op((source.m_state->m_storage->m_data)[source.m_offset]);
                return;
            }
            
            if (out.is_contiguous() && source.is_contiguous()) {
                // fast path
                int64_t total_size = out.volume();

                OutT* __restrict p_out = out._get_storage()->data() + out.m_offset;
                const InT* __restrict p_source = source._get_storage()->data() + source.m_offset;
                
                for (int64_t i = 0; i < total_size; ++i) {
                    p_out[i] = op(p_source[i]);
                }
                return;
            }

            FusedView fused = fuse_dimensions(out.m_shape, {&out.m_strides, &source.right_strides});
            std::vector<int64_t>* out_strides = &fused.strides[0];
            std::vector<int64_t>* source_strides = &fused.strides[1];

            const int64_t n_dim = std::ssize(fused.shared_shape);
            std::vector<int64_t> odometer(n_dim, 0);
            while (odometer[0] < fused.shared_shape[0]) {
                int64_t strided_in_idx = source.m_offset;
                int64_t strided_out_idx = out.m_offset;

                for (int64_t i = 0; i < n_dim; ++i) {
                    strided_in_idx += odometer[i] * (*source_strides)[i];
                    strided_out_idx += odometer[i] * (*out_strides)[i];
                }
                (out.m_state->m_storage->m_data)[strided_out_idx] = op((source.m_state->m_storage->m_data)[strided_in_idx]); 
                ++odometer[n_dim - 1];
                int64_t i = n_dim - 1;
                while ((odometer[i] == fused.shared_shape[i]) && i > 0) {
                    odometer[i] = 0;
                    ++odometer[i - 1];
                    --i;
                }
            }
        }

        template <typename T, typename Func>
        static void apply_unary_in_place(Tensor<T>& source, Func op) {
            if (source.m_shape.empty()) {
                op(source.m_state->m_storage->m_data[source.m_offset]);
                return;
            }

            if (source.is_contiguous()) {
                // fast path
                int64_t total_size = source.volume();

                const T* __restrict p_source = source._get_storage()->data() + source.m_offset;
                
                for (int64_t i = 0; i < total_size; ++i) {
                    op(p_source[i]);
                }
                return;
            }

            FusedView fused = fuse_dimensions(source.m_shape, {&source.m_strides});
            std::vector<int64_t>* source_strides = &fused.strides[1];

            const int64_t n_dim = std::ssize(fused.shared_shape);
            std::vector<int64_t> odometer(n_dim, 0);
            while (odometer[0] < fused.shared_shape[0]) {
                int64_t strided_idx = source.m_offset;

                for (int64_t i = 0; i < n_dim; ++i) {
                    strided_idx += odometer[i] * (*source_strides)[i];
                }
                op((source.m_state->m_storage->m_data)[strided_idx]);
                ++odometer[n_dim - 1];
                int64_t i = n_dim - 1;
                while ((odometer[i] == fused.shared_shape[i]) && i > 0) {
                    odometer[i] = 0;
                    ++odometer[i - 1];
                    --i;
                }
            }
        }

        template <typename T, typename Func>
        static void apply_reduction_operation(Tensor<T>& out, const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, Func op) {
            const int64_t n_dim = std::ssize(source.m_shape);
            if (n_dim == 0) {
                throw std::runtime_error("Tried reducing a 0-Dimensional Tensor.");
            }

            std::shared_ptr<Storage<T>> storage = out._get_storage();
            std::fill(storage->data(), storage->data() + storage->size(), init_value); // initialize garbage memory

            std::vector<int64_t> odometer(n_dim, 0);
            while (odometer[0] < source.m_shape[0]) {
                int64_t in_strided_idx = source.m_offset; 
                int64_t out_strided_idx = 0;

                for (int64_t i = 0; i < n_dim; ++i) {
                    in_strided_idx += odometer[i] * source.m_strides[i];
                    out_strided_idx += odometer[i] * reduction_metadata.temp_strides[i];
                }

                (out.m_state->m_storage->m_data)[out_strided_idx] = op((out.m_state->m_storage->m_data)[out_strided_idx], (source.m_state->m_storage->m_data)[in_strided_idx]);
                ++odometer[n_dim - 1];
                int64_t i = n_dim - 1;
                while ((odometer[i] == source.m_shape[i]) && i > 0) {
                    odometer[i] = 0;
                    ++odometer[i - 1];
                    --i;
                }
            }
        }
    };

}