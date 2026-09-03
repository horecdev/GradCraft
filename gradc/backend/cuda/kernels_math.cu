#include "gradc/backend/cuda/cuda_math.hpp"
#include "gradc/core/detail/shape_inference.hpp"
#include "gradc/backend/cuda/math_functors.cuh"
#include "gradc/backend/cuda/cuda_mapper.cuh"
#include "gradc/backend/cuda/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace gradc {
    constexpr int MAX_DIMS = 8;

    struct CUDAMeta { // when you copy a struct with an array, it doesnt decay. The compiler respects struct's size.
        int64_t data[MAX_DIMS];
        int64_t size;
    };

    inline CUDAMeta to_cuda_meta(const std::vector<int64_t>& vec) { // turning CPU vector into primitive struct
        CUDAMeta meta;
        meta.size = std::ssize(vec);
        for (int64_t i = 0; i < std::ssize(vec); ++i) {
            meta.data[i] = vec[i];
        }
        return meta;
    }

    inline cublasHandle_t get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS handle initialization failed.");
            }
        }
        return handle;
    }
    

    #pragma region BINARY OUT OF PLACE

    template <typename T, typename Func>
    __global__ void binary_out_of_place_kernel(
        T* __restrict__ p_out, const T* __restrict__ p_left, const T* __restrict__ p_right, 
        CUDAMeta shared_shape, 
        CUDAMeta out_strides, CUDAMeta left_strides, CUDAMeta right_strides,
        int64_t out_offset, int64_t left_offset, int64_t right_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {

            int64_t temp_idx = linear_idx;

            int64_t out_strided_idx = out_offset;
            int64_t left_strided_idx = left_offset;
            int64_t right_strided_idx = right_offset;

            for (int64_t i = shared_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % shared_shape.data[i];
                temp_idx /= shared_shape.data[i];

                out_strided_idx += coord * out_strides.data[i];
                left_strided_idx += coord * left_strides.data[i];
                right_strided_idx += coord * right_strides.data[i];
            }

            p_out[out_strided_idx] = op(p_left[left_strided_idx], p_right[right_strided_idx]);
        }
    }

    template <typename T, typename Func>
    __global__ void binary_out_of_place_kernel_fast(
        T* __restrict__ p_out, const T* __restrict__ p_left, const T* __restrict__ p_right, 
        int64_t out_offset, int64_t left_offset, int64_t right_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            p_out[out_offset + linear_idx] = op(p_left[left_offset + linear_idx], p_right[right_offset + linear_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_binary_out_of_place(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, BinaryOp op) {
        cudaSetDevice(out.device().index);
        int64_t total_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_out = out._get_storage()->data();
        const T* p_left = left._get_storage()->data();
        const T* p_right = right._get_storage()->data(); 
        
        if (out.is_contiguous() && left.is_contiguous() && right.is_contiguous() && out.m_shape == left.m_shape && left.m_shape == right.m_shape) {
            cuda_mapper::map_boop<T>(op, [&](auto functor) {binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, functor);});
            return;
        }

        const std::vector<int64_t>* left_strides = &left.m_strides;
        const std::vector<int64_t>* right_strides = &right.m_strides;
        int64_t left_offset = left.m_offset;
        int64_t right_offset = right.m_offset; 

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

        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_out_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_left_strides = to_cuda_meta(fused.strides[1]);
        CUDAMeta gpu_right_strides = to_cuda_meta(fused.strides[2]);

        cuda_mapper::map_boop<T>(op, [&](auto functor) {binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, functor);});
    }

    #pragma endregion BINARY OUT OF PLACE
    #pragma region BINARY IN PLACE

    template <typename T, typename Func>
    __global__ void binary_in_place_kernel_fast(
        T* __restrict__ p_left, const T* __restrict__ p_right, 
        int64_t left_offset, int64_t right_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            op(p_left[left_offset + linear_idx], p_right[right_offset + linear_idx]);
        }
    }

    template <typename T, typename Func>
    __global__ void binary_in_place_kernel(
        T* __restrict__ p_left, const T* __restrict__ p_right, 
        CUDAMeta shared_shape, 
        CUDAMeta left_strides, CUDAMeta right_strides,
        int64_t left_offset, int64_t right_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {

            int64_t temp_idx = linear_idx;

            int64_t left_strided_idx = left_offset;
            int64_t right_strided_idx = right_offset;

            for (int64_t i = shared_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % shared_shape.data[i];
                temp_idx /= shared_shape.data[i];

                left_strided_idx += coord * left_strides.data[i];
                right_strided_idx += coord * right_strides.data[i];
            }

            op(p_left[left_strided_idx], p_right[right_strided_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_binary_in_place(Tensor<T>& left, const Tensor<T>& right, BinaryOpInPlace op) {
        cudaSetDevice(left.device().index);
        int64_t total_elems = left.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_left = left._get_storage()->data();
        const T* p_right = right._get_storage()->data(); 

        if (left.m_shape == right.m_shape && left.is_contiguous() && right.is_contiguous()) {
            cuda_mapper::map_bip<T>(op, [&](auto functor) {binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, functor);});
            return;
        }

        const std::vector<int64_t>* right_strides;
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

        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_left_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_right_strides = to_cuda_meta(fused.strides[1]);
        
        cuda_mapper::map_bip<T>(op, [&](auto functor) {binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, functor);});
    }

    #pragma endregion BINARY IN PLACE
    #pragma region UNARY OUT OF PLACE

    template <typename OutT, typename InT, typename Func>
    __global__ void unary_out_of_place_kernel(
        OutT* __restrict__ p_out, const InT* __restrict__ p_source, 
        CUDAMeta shared_shape, 
        CUDAMeta out_strides, CUDAMeta source_strides,
        int64_t out_offset, int64_t source_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {

            int64_t temp_idx = linear_idx;

            int64_t out_strided_idx = out_offset;
            int64_t source_strided_idx = source_offset;

            for (int64_t i = shared_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % shared_shape.data[i];
                temp_idx /= shared_shape.data[i];

                out_strided_idx += coord * out_strides.data[i];
                source_strided_idx += coord * source_strides.data[i];
            }

            p_out[out_strided_idx] = op(p_source[source_strided_idx]);
        }
    }

    template <typename OutT, typename InT, typename Func>
    __global__ void unary_out_of_place_kernel_fast(
        OutT* __restrict__ p_out, const InT* __restrict__ p_source, 
        int64_t out_offset, int64_t source_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            p_out[out_offset + linear_idx] = op(p_source[source_offset + linear_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_unary_out_of_place(Tensor<T>& out, const Tensor<T>& source, UnaryOp op) {
        cudaSetDevice(out.device().index);
        int64_t total_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_out = out._get_storage()->data();
        const T* p_source = source._get_storage()->data(); 

        if (out.is_contiguous() && source.is_contiguous() && out.m_shape == source.m_shape) {
            cuda_mapper::map_uoop<T>(op, [&](auto functor) {unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, functor);});
            return;
        }

        FusedView fused = fuse_dimensions(out.m_shape, {&out.m_strides, &source.m_strides});
        const std::vector<int64_t>* out_strides = &fused.strides[0];
        const std::vector<int64_t>* source_strides = &fused.strides[1];

        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_out_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[1]);

        cuda_mapper::map_uoop<T>(op, [&](auto functor) {unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, functor);});
    }

    template <typename OutT, typename InT>
    void CUDAMath::apply_cast_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source) {
        cudaSetDevice(out.device().index);
        int64_t total_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        OutT* p_out = out._get_storage()->data();
        const InT* p_source = source._get_storage()->data(); 

        if (out.is_contiguous() && source.is_contiguous() && out.m_shape == source.m_shape) {
            unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Cast<InT, OutT>());
        }

        FusedView fused = fuse_dimensions(out.m_shape, {&out.m_strides, &source.m_strides});
        const std::vector<int64_t>* out_strides = &fused.strides[0];
        const std::vector<int64_t>* source_strides = &fused.strides[1];

        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_out_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[1]);

        unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Cast<InT, OutT>());
    }

    #pragma endregion UNARY OUT OF PLACE
    #pragma region UNARY IN PLACE

    template <typename T, typename Func>
    __global__ void unary_in_place_kernel(
        T* __restrict__ p_source, 
        CUDAMeta shared_shape, 
        CUDAMeta source_strides,
        int64_t source_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {

            int64_t temp_idx = linear_idx;

            int64_t source_strided_idx = source_offset;

            for (int64_t i = shared_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % shared_shape.data[i];
                temp_idx /= shared_shape.data[i];

                source_strided_idx += coord * source_strides.data[i];
            }

            op(p_source[source_strided_idx]);
        }
    }

    template <typename T, typename Func>
    __global__ void unary_in_place_kernel_fast(
        T* __restrict__ p_source, 
        int64_t source_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            op(p_source[source_offset + linear_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_unary_in_place(Tensor<T>& source, UnaryOpInPlace op) {
        cudaSetDevice(source.device().index);
        int64_t total_elems = source.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_source = source._get_storage()->data();

        if (source.is_contiguous()) {
            cuda_mapper::map_uip<T>(op, [&](auto functor) {unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, functor);});
            return;
        }

        FusedView fused = fuse_dimensions(source.m_shape, {&source.m_strides});
        
        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[0]);

        cuda_mapper::map_uip<T>(op, [&](auto functor) {unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, functor);});
    }

    #pragma endregion UNARY IN PLACE
    #pragma region REDUCTIONS

    template <typename T, typename Func>
    __global__ void reduction_kernel(
        T* __restrict__ p_out, const T* __restrict__ p_source, 
        CUDAMeta source_shape, 
        CUDAMeta out_strides, CUDAMeta source_strides,
        int64_t source_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {

            int64_t temp_idx = linear_idx;

            int64_t out_strided_idx = 0;
            int64_t source_strided_idx = source_offset;

            for (int64_t i = source_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % source_shape.data[i];
                temp_idx /= source_shape.data[i];

                out_strided_idx += coord * out_strides.data[i];
                source_strided_idx += coord * source_strides.data[i];
            }

            op.atomic(&p_out[out_strided_idx], p_source[source_strided_idx]);
        }
    }

    template <typename T, typename Func> 
    __global__ void reduction_kernel_whole(
        T* __restrict__ p_out, const T* __restrict__ p_source, 
        int64_t source_offset, T init_value,
        int64_t total_elements, Func op) {
        
        __shared__ T shared_data[256];

        int64_t t_idx = threadIdx.x;
        int64_t  global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (global_idx < total_elements) {
            shared_data[t_idx] = p_source[source_offset + global_idx];
        }
        else {
            shared_data[t_idx] = init_value;
        }
        __syncthreads(); // all threads wait for init to finish

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (t_idx < stride) {
                shared_data[t_idx] = op(shared_data[t_idx], shared_data[t_idx + stride]); // not atomic since its thread safe
            } 
            __syncthreads(); // wait till halving is done
        }

        if (t_idx == 0) {
            op.atomic(&p_out[0], shared_data[0]);
        }
    }

    template <typename T>
    void CUDAMath::apply_reduction_operation(Tensor<T>& out, const Tensor<T>& source, const RedMeta& red_meta, ReduceOp op) {
        cudaSetDevice(out.device().index);
        int64_t total_elems = source.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_out = out._get_storage()->data();
        T* p_source = source._get_storage()->data();

        if (out.volume() == 1 && source.is_contiguous()) {
            cuda_mapper::map_red<T>(op, [&](auto functor, T init_value) {
                CUDAUtils::fill(p_out, init_value, out.volume(), out.device());
                reduction_kernel_whole<<<blocks, threads>>>(p_out, p_source, source.m_offset, init_value, total_elems, functor); 
            });
            return;
        }

        CUDAMeta source_shape = to_cuda_meta(source.m_shape);
        CUDAMeta out_strides = to_cuda_meta(red_meta.temp_strides);
        CUDAMeta source_strides = to_cuda_meta(source.m_strides);

        cuda_mapper::map_red<T>(op, [&](auto functor, T init_value) {
            CUDAUtils::fill(p_out, init_value, out.volume(), out.device());
            reduction_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, out_strides, source_strides, source.m_offset, total_elems, functor);
        });
    }

    template <typename T, typename Func>
    __global__ void argextr_kernel_whole(
        int64_t* __restrict__ p_out, const T* __restrict__ p_source, 
        int64_t source_offset, T init_value,
        int64_t total_elements, Func op) {

        __shared__ T shared_vals[1024];
        __shared__ int64_t shared_idxs[1024];

        int64_t t_idx = threadIdx.x;
        // no global idx because we are launching one block
        
        T thread_best_val = init_value;
        int64_t thread_best_idx = 0;

        int64_t current_idx = t_idx;

        while (current_idx < total_elements) {
            T val = p_source[source_offset + current_idx];

            if (op(val, thread_best_val)) { // op takes: new val, old val
                thread_best_val = val;
                thread_best_idx = current_idx;
            }
            current_idx += blockDim.x; // += 1024
        }

        shared_vals[t_idx] = thread_best_val;
        shared_idxs[t_idx] = thread_best_idx;
        __syncthreads();

        for (int stride = blockDim.x; stride > 0; stride >>= 1) {
            if (t_idx < stride) {
                if (op(shared_vals[t_idx + stride], shared_vals[t_idx])) {
                    shared_vals[t_idx] = shared_vals[t_idx + stride];
                    shared_idxs[t_idx] = shared_idxs[t_idx + stride];
                }
            }
            __syncthreads();
        }

        if (t_idx == 0) {
            p_out[0] = shared_idxs[0];
        }
    }

    template <typename T, typename Func>
    __global__ void argextr_kernel(
        int64_t* __restrict__ p_out, const T* __restrict__ p_source, 
        CUDAMeta source_shape, CUDAMeta source_strides,
        int64_t source_offset, int64_t dim, T init_value,
        int64_t out_elems, Func op) {

        int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x; // each out element gets its own thread

        if (out_idx < out_elems) {
            int64_t source_idx = source_offset;
            int64_t temp_idx = out_idx;

            // figure out where in out we are
            for (int64_t i = source_shape.size - 1; i >= 0; --i) {
                if (i == dim) {continue;}
                int64_t coord = temp_idx % source_shape.data[i];
                temp_idx /= source_shape.data[i];

                source_idx += coord * source_strides.data[i];
            }

            int64_t dim_size = source_shape.data[dim];
            int64_t dim_stride = source_strides.data[dim];

            T best_val = init_value;
            int64_t best_idx = 0;
            // find best along the dimension
            for (int64_t i = 0; i < dim_size; ++i) {
                T new_val = p_source[source_idx + (i * dim_stride)];
                if (op(new_val, best_val)) {
                    best_val = p_source[source_idx];
                    best_idx = i;
                }
            }
            p_out[out_idx] = best_idx;
        }
    }
    
    

   template <typename T>
    void CUDAMath::apply_arg_extr_operation(Tensor<int64_t>& out, const Tensor<T>& source, int64_t dim, ArgExtrOp op) {
        cudaSetDevice(out.device().index);
        int64_t total_elems = source.volume();
        int64_t* p_out = out._get_storage()->data();
        T* p_source = source._get_storage()->data();

        if (out.volume() == 1 && source.is_contiguous()) {
            cuda_mapper::map_argextr<T>(op, [&](auto functor, T init_value) {argextr_kernel_whole<<<1, 1024>>>(p_out, p_source, source.m_offset, init_value, total_elems, functor);});
            return;
        }

        int64_t out_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (out_elems + threads - 1) / threads;

        CUDAMeta source_shape = to_cuda_meta(source.m_shape);
        CUDAMeta source_strides = to_cuda_meta(source.m_strides);

        cuda_mapper::map_argextr<T>(op, [&](auto functor, T init_value) {argextr_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, source_strides, source.m_offset, dim, init_value, out_elems, functor);});
    }

    #pragma endregion REDUCTIONS

    #pragma region EMBEDDINGS

    template <typename T>
    __global__ void embed_kernel_fast(
        T* __restrict__ p_out, int64_t* __restrict__ p_indices, T* __restrict__ p_embeds, 
        int64_t indices_offset, int64_t out_vol, int64_t embed_vol
    ) {
        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear_idx < out_vol) {
            int64_t token_pos = linear_idx / embed_vol;
            int64_t embed_dim_idx = linear_idx % embed_vol;
            
            int64_t token_id = p_indices[indices_offset + token_pos];

            p_out[linear_idx] = p_embeds[token_id * embed_vol + embed_dim_idx];
        }
    }

    template <typename T>
    __global__ void embed_kernel(
        T* __restrict__ p_out, int64_t* __restrict__ p_indices, T* __restrict__ p_embeds, 
        CUDAMeta indices_shape, CUDAMeta indices_strides,
        int64_t indices_offset, int64_t out_vol, int64_t embed_vol
    ) {
        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < out_vol) {
            int64_t token_pos = linear_idx / embed_vol; // linear idx in the indices
            int64_t embed_dim_idx = linear_idx % embed_vol;

            int64_t temp_idx = token_pos;

            int64_t idx_strided_idx = indices_offset;

            for (int64_t i = indices_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % indices_shape.data[i];
                temp_idx /= indices_shape.data[i];

                idx_strided_idx += coord * indices_strides.data[i];
            }

            int64_t token_id = p_indices[idx_strided_idx];

            p_out[linear_idx] = p_embeds[token_id * embed_vol + embed_dim_idx];
        }
    }

    template <typename T>
    void CUDAMath::apply_embed(Tensor<T>& out, const Tensor<int64_t>& indices, const Tensor<T>& embeds, int64_t embed_vol) {
        // OUT MUST BE DENSE. EMBEDDINGS MUST BE DENSE.
        cudaSetDevice(out.device().index);
        int64_t out_vol = out.volume();
        int64_t threads = 256;
        int64_t blocks = (out_vol + threads - 1) / threads;

        T* p_out = out._get_storage()->data();
        int64_t* p_indices = indices._get_storage()->data();
        T* p_embeds = embeds._get_storage()->data();

        if (indices.is_contiguous()) {
            embed_kernel_fast<<<blocks, threads>>>(p_out, p_indices, p_embeds, indices.m_offset, out_vol, embed_vol);
            return;
        }

        FusedView fused = fuse_dimensions(indices.m_shape, {&indices.m_strides});
        
        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_indices_strides = to_cuda_meta(fused.strides[0]);

        embed_kernel<<<blocks, threads>>>(p_out, p_indices, p_embeds, gpu_shape, gpu_indices_strides, indices.m_offset, out_vol, embed_vol);
    }

    template <typename T>
    __global__ void scatter_add_kernel_fast(
        T* __restrict__ p_dembeds, int64_t* __restrict__ p_indices, T* __restrict__ p_out_grad, 
        int64_t indices_offset, int64_t out_grad_vol, int64_t embed_vol
    ) {
        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear_idx < out_grad_vol) {
            int64_t token_pos = linear_idx / embed_vol;
            int64_t embed_dim_idx = linear_idx % embed_vol;
            
            int64_t token_id = p_indices[indices_offset + token_pos];

            cuda_functors::RED::Sum<T>().atomic(&p_dembeds[token_id * embed_vol + embed_dim_idx], p_out_grad[linear_idx]);
        }
    }

    template <typename T>
    __global__ void scatter_add_kernel(
        T* __restrict__ p_dembeds, int64_t* __restrict__ p_indices, T* __restrict__ p_out_grad, 
        CUDAMeta indices_shape, CUDAMeta indices_strides,
        int64_t indices_offset, int64_t out_grad_vol, int64_t embed_vol
    ) {
        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < out_grad_vol) {
            int64_t token_pos = linear_idx / embed_vol;
            int64_t embed_dim_idx = linear_idx % embed_vol;

            int64_t temp_idx = token_pos;

            int64_t idx_strided_idx = indices_offset;

            for (int64_t i = indices_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % indices_shape.data[i];
                temp_idx /= indices_shape.data[i];

                idx_strided_idx += coord * indices_strides.data[i];
            }

            int64_t token_id = p_indices[idx_strided_idx];

            cuda_functors::RED::Sum<T>().atomic(&p_dembeds[token_id * embed_vol + embed_dim_idx], p_out_grad[linear_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_scatter_add(Tensor<T>& dembeds, const Tensor<int64_t>& indices, const Tensor<T>& out_grad, int64_t embed_vol) {
        // OUT_GRAD MUST BE DENSE. DEMBEDDINGS MUST BE DENSE.
        cudaSetDevice(dembeds.device().index);
        int64_t out_grad_vol = out_grad.volume();
        int64_t threads = 256;
        int64_t blocks = (out_grad_vol + threads - 1) / threads;

        T* p_dembeds = dembeds._get_storage()->data();
        int64_t* p_indices = indices._get_storage()->data();
        T* p_out_grad = out_grad._get_storage()->data();

        if (indices.is_contiguous()) {
            scatter_add_kernel_fast<<<blocks, threads>>>(p_dembeds, p_indices, p_out_grad, indices.m_offset, out_grad_vol, embed_vol);
            return;
        }

        FusedView fused = fuse_dimensions(indices.m_shape, {&indices.m_strides});
        
        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_indices_strides = to_cuda_meta(fused.strides[0]);

        scatter_add_kernel<<<blocks, threads>>>(p_dembeds, p_indices, p_out_grad, gpu_shape, gpu_indices_strides, indices.m_offset, out_grad_vol, embed_vol);
    }

    #pragma endregion EMBEDDINGS

    #pragma region MATRIX MULTIPLY

    template <typename T>
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_normal_gemm(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const NMMMeta& blas_meta) {
        T* p_out = out._get_storage()->data() + out.m_offset;
        const T* p_left = left._get_storage()->data() + left.m_offset;
        const T* p_right = right._get_storage()->data() + right.m_offset;

        cublasHandle_t handle = get_cublas_handle();
        cublasOperation_t op_left = (blas_meta.left_op == MatrixTensorOp::Normal) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t op_right = (blas_meta.right_op == MatrixTensorOp::Normal) ? CUBLAS_OP_N : CUBLAS_OP_T;

        T typed_alpha = static_cast<T>(blas_meta.alpha);
        T typed_beta = static_cast<T>(blas_meta.beta);

        // row major = transposed col major
        // AB = B^T A^T so just swap. col major cublas will treat them as transposed
        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, op_right, op_left, blas_meta.N, blas_meta.M, blas_meta.K,
                 &typed_alpha, p_right, blas_meta.ldb, 
                 p_left, blas_meta.lda, &typed_beta, p_out, blas_meta.ldc);
        }
        else if constexpr (std::is_same_v<T, double>) {
            cublasDgemm(handle, op_right, op_left, blas_meta.N, blas_meta.M, blas_meta.K, 
                &typed_alpha, p_right, blas_meta.ldb, 
                p_left, blas_meta.lda, &typed_beta, p_out, blas_meta.ldc);
        }
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_batched_gemm(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const BMMMeta& blas_meta) {
        T* p_out = out._get_storage()->data() + out.m_offset;
        const T* p_left = left._get_storage()->data() + left.m_offset;
        const T* p_right = right._get_storage()->data() + right.m_offset;

        cublasHandle_t handle = get_cublas_handle();
        cublasOperation_t op_left = (blas_meta.left_op == MatrixTensorOp::Normal) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t op_right = (blas_meta.right_op == MatrixTensorOp::Normal) ? CUBLAS_OP_N : CUBLAS_OP_T;

        T typed_alpha = static_cast<T>(blas_meta.alpha);
        T typed_beta = static_cast<T>(blas_meta.beta);

        if constexpr (std::is_same_v<T, float>) {
            cublasSgemmStridedBatched(handle, op_right, op_left, blas_meta.N, blas_meta.M, blas_meta.K, 
                &typed_alpha, p_right, blas_meta.ldb, blas_meta.stride_b, p_left, 
                blas_meta.lda, blas_meta.stride_a, &typed_beta, 
                p_out, blas_meta.ldc, blas_meta.stride_c, blas_meta.batch_count);
        }
        else if constexpr (std::is_same_v<T, double>) {
            cublasDgemmStridedBatched(handle, op_right, op_left, blas_meta.N, blas_meta.M, blas_meta.K, 
                &typed_alpha, p_right, blas_meta.ldb, blas_meta.stride_b, p_left, 
                blas_meta.lda, blas_meta.stride_a, &typed_beta, 
                p_out, blas_meta.ldc, blas_meta.stride_c, blas_meta.batch_count);
        }
    }

    #pragma endregion MATRIX MULTIPLY



    #pragma region RMSNorm

    template <typename T>
    __global__ void rmsnorm_forward_kernel_fast(
        T* __restrict__ p_out, T* __restrict__ p_inv_rms, 
        const T* __restrict__ p_parent, const T* __restrict__ p_gamma,
        int64_t parent_offset, 
        int64_t reduced_vol, T eps
    ) {
        int64_t row = blockIdx.x;

        const T* parent_row = p_parent + parent_offset + (row * reduced_vol);
        T* out_row = p_out + (row * reduced_vol);
        const T* gamma_row = p_gamma; // also contiguous flat 

        int64_t tid = threadIdx.x;
        T thread_sq_sum = 0;

        // loop over reduced_vol elements. Thread 0 does 0, 256, 512...
        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            T val = parent_row[i];
            thread_sq_sum += val * val;
        }

        __shared__ T s_sum[256];

        s_sum[tid] = thread_sq_sum;
        __syncthreads();

        for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
            }
            _syncthreads();
        }

        T inv_rms = 0;
        if (tid == 0) {
            inv_rms = static_cast<T>(1.0) / sqrt((s_sum[0] / static_cast<T>(reduced_vol)) + eps);
            
            // if we dont save inv_rms then its nullptr
            if (p_inv_rms != nullptr) {
                p_inv_rms[row] = inv_rms;
            }

            s_sum[0] = inv_rms;
        }
        __syncthreads();
        inv_rms = s_sum[0]; // now all 0-255 threads have the same inv_rms (before it was 0 and tid=0 only had it)

        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            out_row[i] = parent_row[i] * inv_rms * gamma_row[i];
        }
    }

    template <typename T>
    __global__ void rmsnorm_forward_kernel_strided(
        T* __restrict__ p_out, T* __restrict__ p_inv_rms,
        const T* __restrict__ p_parent, const T* __restrict__ p_gamma,
        CUDAMeta reduced_shape, CUDAMeta normalized_shape,
        CUDAMeta out_strides, CUDAMeta parent_strides,
        int64_t parent_offset,
        int64_t reduced_vol, T eps
    ) {
        int64_t row = blockIdx.x;
        int64_t tid = blockDim.x;

        int64_t temp_idx = row;
        int64_t base_parent = parent_offset;
        int64_t base_out = 0;
        int64_t base_gamma = 0;

        // first figure out the base (where are we in the reduced_shape e.g. [B, 1, C] in the B, C dims)
        for (int64_t d = reduced_shape.size; d >= 0; --d) {
            int64_t coord = temp_idx % reduced_shape.data[d];
            temp_idx /= reduced_shape.data[d];

            base_parent += coord * parent_strides.data[d];
            base_out += coord * out_strides.data[d];
        }

        // thread local sum - 256 threads have partial results of summation across the reduced dims
        T thread_sq_sum = 0;
        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            int64_t temp_i = i;
            int64_t mov_parent = 0;

            // sum over the normalized_shape into 256 diff threads
            for (int64_t d = normalized_shape.size - 1; d >= 0; --d) {
                int64_t coord = temp_i % normalized_shape.data[d];
                temp_i /= normalized_shape.data[d];
                mov_parent += coord * parent_strides.data[d];
            }

            T val = p_parent[base_parent + mov_parent];
            thread_sq_sum += val * val;
        }

        // sum into one value
        __shared__ T s_sum[256];
        s_sum[tid] = thread_sq_sum;
        __syncthreads();

        for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
            }
            __syncthreads();
        }

        T inv_rms = 0;
        if (tid == 0) {
            inv_rms = static_cast<T>(1.0) / sqrt((s_sum[0] / static_cast<T>(reduced_vol)) + eps);

            // if we dont save inv_rms then its nullptr
            if (p_inv_rms != nullptr) {
                p_inv_rms[row] = inv_rms;
            }
            
            s_sum[0] = inv_rms;
        }
        __syncthreads();
        inv_rms = s_sum[0]; // share to all threads

        // where we are in the normalized shape again (to use our inv_rms and actually update)
        // we already have where we are in the reduced_shape. 
        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            int64_t temp_i = 0;
            int64_t mov_parent = 0;
            int64_t mov_out = 0;

            for (int64_t d = normalized_shape.size - 1; d >= 0; --d) {
                int64_t coord = temp_i % normalized_shape.data[d];
                temp_i /= normalized_shape.data[d];
                mov_parent += coord * parent_strides.data[d];
                mov_out += coord * out_strides.data[d];
            }

            T val = p_parent[base_parent + mov_parent];
            T g = p_gamma[base_gamma + i];

            p_out[base_out + mov_out] = val * g * inv_rms;
        }
    }

    template <typename T> 
    requires std::is_floating_point_v<T>
    // out, inv_rms, gamma ALWAYS DENSE
    void CUDAMath::apply_rmsnorm_forward(Tensor<T>& out, Tensor<T>& inv_rms, const Tensor<T>& parent, const Tensor<T>& gamma, const RedMeta& red_meta, const std::vector<int64_t>& normalized_shape, T eps) {
        cudaSetDevice(out.device().index);
        int64_t threads = 256;
        // you launch result_vol blocks, 256 threads each
        int64_t blocks = red_meta.result_vol;
        // say you launch block 100. It starts in RAM exactly at 100 * reduced_vol (skip over the tokens normalized by previous blocks)
        // for [10, 30, 500] it starts at [4, 10, 0] - skipped over 100 * 500 elems
        // To make it work all the result_vol elems (say B, T) must be contiguous. In addition all the dims being reduced must be trailing and contiguous too.

        bool is_fast = parent.is_contiguous();
        // all ones have to be on the left, and all numbers have to be trailing
        if (is_fast) {
            bool found_1 = false;
            for (int64_t i = std::ssize(normalized_shape) - 1; i >= 0; --i) {
                if (normalized_shape[i] != 1) {
                    if (found_1 == true) { // there was one to the right of it
                        is_fast = false;
                        break;
                    }
                }
                else {
                    found_1 = true;
                }
            }
        }

        T* p_out = out._get_storage()->data();
        T* p_inv_rms = inv_rms._get_storage()->data();
        const T* p_parent = parent._get_storage()->data();
        const T* p_gamma = gamma._get_storage()->data();

        if (is_fast) {
            rmsnorm_forward_kernel_fast<<<blocks, threads>>>(p_out, p_inv_rms, p_parent, p_gamma, parent.offset(), red_meta.reduced_vol, eps);
            // fire up the fast kernel
        }
        else {
            CUDAMeta gpu_reduced_shape = to_cuda_meta(red_meta.temp_shape);
            CUDAMeta gpu_normalized_shape = to_cuda_meta(normalized_shape);
            CUDAMeta gpu_out_strides = to_cuda_meta(out.strides());
            CUDAMeta gpu_parent_strides = to_cuda_meta(parent.strides());
            rmsnorm_forward_kernel_strided<<<blocks, threads>>>(p_out, p_inv_rms, p_parent, p_gamma, gpu_reduced_shape, gpu_normalized_shape, gpu_out_strides, gpu_parent_strides, parent.offset(), red_meta.reduced_vol, eps);
        }
    }

    template <typename T>
    // dx, dgamma, dy, gamma, inv_rms are always DENSE.
    __device__ void rmsnorm_backward_kernel_fast(
        T* __restrict__ p_dx, T* __restrict__ p_dgamma, 
        const T* __restrict__  p_out_grad, const T* __restrict__ p_parent, 
        const T* __restrict__ p_gamma, const T* __restrict__ p_inv_rms,
        int64_t parent_offset, int64_t reduced_vol
    ) {
        int64_t row = blockIdx.x;
        int64_t tid = threadIdx.x;

        const T* out_grad_row = p_out_grad + (row * reduced_vol);
        const T* parent_row = p_parent + parent_offset + (row * reduced_vol);
        T* dx_row = p_dx != nullptr ? p_dx + (row * reduced_vol) : nullptr;
        const T* gamma_row = p_gamma;

        T inv_rms = p_inv_rms[row];

        // sum for each thread of dx_hat * x_norm
        T thread_sum = 0;
        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            T dx_hat = out_grad_row[i] * gamma_row[i];
            T x_norm = parent_row[i] * inv_rms;
            thread_sum += dx_hat * x_norm;
        }

        __shared__ T s_sum[256];
        s_sum[tid] = thread_sum;
        __syncthreads();

        for (int64_t s = blockDim.x / 2; s > 0; s >>= 1)  { // shift right by 1 = div 2
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
            }
            __syncthreads();
        }

        T sum_term = 0;
        if (tid == 0) {
            sum_term = s_sum[0] / static_cast<T>(reduced_vol);
            s_sum[0] = sum_term;
        }
        __syncthreads();
        sum_term = s_sum[0]; // share

        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            T x_norm = parent_row[i] * inv_rms;

            if (p_dx != nullptr) {
                T dx_hat = out_grad_row[i] * gamma_row[i];
                dx_row[i] = inv_rms * (dx_hat - x_norm * sum_term);
            }

            if (p_gamma != nullptr) {
                T dgamma_val = out_grad_row[i] * x_norm;

                cuda_functors::RED::Sum<T>().atomic(&p_dgamma[i], dgamma_val);
                // memory of dgamma must be initialized to zeros beforehand
            }    
        }
    }

    template <typename T>
    __device__ void rmsnorm_backward_kernel_strided(
        T* __restrict__ p_dx, T* __restrict__ p_dgamma, 
        const T* __restrict__ p_out_grad, const T* __restrict__ p_parent, 
        const T* __restrict__ p_gamma, const T* __restrict__ p_inv_rms,
        CUDAMeta reduced_shape, CUDAMeta normalized_shape, 
        CUDAMeta dx_strides, CUDAMeta out_grad_strides, CUDAMeta parent_strides,
        int64_t parent_offset, int64_t reduced_vol
    ) {
        int64_t row = blockIdx.x;
        int64_t tid = threadIdx.x;

        // get location in reduced_shape
        int64_t temp_idx = row;
        int64_t base_dx = 0;
        int64_t base_out_grad = 0;
        int64_t base_parent = parent_offset;

        for (int64_t d = reduced_shape.size - 1; d >= 0; --d) {
            int64_t coord = temp_idx % reduced_shape.data[d];
            temp_idx /= reduced_shape.data[d];

            base_dx += coord * dx_strides.data[d];
            base_out_grad += coord * out_grad_strides.data[d];
            base_parent += coord * parent_strides.data[d];
        }

        T inv_rms = p_inv_rms[row];

        // get location in normalized_shape
        T thread_sum = 0;
        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            int64_t temp_i = i;
            int64_t mov_out_grad = 0;
            int64_t mov_parent = 0;

            for (int64_t d = normalized_shape.size; d >= 0; --d) {
                int64_t coord = temp_i % normalized_shape.data[d];
                temp_i /= normalized_shape.data[d];

                mov_out_grad += coord * out_grad_strides.data[d];
                mov_parent += coord * parent_strides.data[d];
            }

            T x_val = p_parent[base_parent + mov_parent];
            T out_grad_val = p_out_grad[base_out_grad + mov_out_grad];
            T gamma_val = p_gamma[i];

            T dx_hat = out_grad_val * gamma_val;
            T x_norm = inv_rms * x_val;

            thread_sum += dx_hat * x_norm;
        }

        __shared__ T s_sum[256];
        s_sum[tid] = thread_sum;
        __syncthreads();

        for (int64_t s = blockDim.x / 2; s >= 0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
            }
            __syncthreads();
        }

        T sum_term = 0;
        if (tid == 0) {
            sum_term = s_sum[0] / static_cast<T>(reduced_vol);
            s_sum[0] = sum_term;
        }
        __syncthreads();
        sum_term = s_sum[0];

        for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
            int64_t temp_i = i;
            int64_t mov_dx = 0;
            int64_t mov_out_grad = 0;
            int64_t mov_parent = 0;

            for (int64_t d = normalized_shape.size - 1; d >= 0; --d) {
                int64_t coord = temp_i % normalized_shape.data[d];
                temp_i /= normalized_shape.data[d];

                mov_dx += coord * dx_strides.data[d];
                mov_out_grad += coord * out_grad_strides.data[d];
                mov_parent += coord * parent_strides.data[d];
            }

            T x_val = p_parent[base_parent + mov_parent];
            T out_grad_val = p_out_grad[base_out_grad + mov_out_grad];
            T x_norm = x_val * inv_rms;

            if (p_dx != nullptr) {
                T gamma_val = p_gamma[i];
                T dx_hat = out_grad_val * gamma_val;
                p_dx[base_dx + mov_dx] = inv_rms * (dx_hat - x_norm * sum_term);
            }

            if (p_dgamma != nullptr) {
                T dgamma_val = out_grad_val * x_norm;
                cuda_functors::RED::Sum<T>().atomic(&p_dgamma[i], dgamma_val);
            }
        }
    }

    template <typename T> 
    requires std::is_floating_point_v<T>
    // dx, out_grad, dgamma, gamma, inv_rms always DENSE
    void CUDAMath::apply_rmsnorm_backward(Tensor<T>& dx, Tensor<T>& dgamma, const Tensor<T>& out_grad, const Tensor<T>& parent, const Tensor<T>& gamma, const Tensor<T>& inv_rms, const RedMeta& red_meta, const std::vector<int64_t>& normalized_shape) {
        cudaSetDevice(out_grad.device().index);
        int64_t threads = 256;
        int64_t blocks = red_meta.result_vol;

        bool is_fast = parent.is_contiguous();
        if (is_fast) {
            bool found_1 = false;
            for (int64_t i = std::ssize(normalized_shape) - 1; i >= 0; --i) {
                if (normalized_shape[i] != 1) {
                    if (found_1 == true) {
                        is_fast = false;
                        break;
                    }
                }
                else {
                    found_1 = true;
                }
            }
        }

        T* p_dx = dx._get_storage()->data();
        T* p_dgamma = dgamma._get_storage()->data();
        const T* p_out_grad = out_grad._get_storage()->data();
        const T* p_parent = parent._get_storage()->data();
        const T* p_gamma = gamma._get_storage()->data();
        const T* p_inv_rms = inv_rms._get_storage()->data();

        if (is_fast) {
            rmsnorm_backward_kernel_fast<<<blocks, threads>>>(p_dx, p_dgamma, p_out_grad, p_parent, p_gamma, p_inv_rms, parent.offset(), red_meta.reduced_vol);
        }
        else {
            CUDAMeta gpu_reduced_shape = to_cuda_meta(red_meta.temp_shape);
            CUDAMeta gpu_normalized_shape = to_cuda_meta(normalized_shape);
            CUDAMeta gpu_dx_strides = to_cuda_meta(dx.strides());
            CUDAMeta gpu_out_grad_strides = to_cuda_meta(out_grad.strides());
            CUDAMeta gpu_parent_strides = to_cuda_meta(parent.strides());
            rmsnorm_backward_kernel_strided<<<blocks, threads>>>(p_dx, p_dgamma, p_out_grad, p_parent, p_gamma, p_inv_rms, gpu_reduced_shape, gpu_normalized_shape, gpu_dx_strides, gpu_out_grad_strides, gpu_parent_strides, parent.offset(), red_meta.reduced_vol);
        }
    }

    #pragma endregion RMSNorm

    #pragma region CAUSAL SOFTMAX

    template <typename T>
    requires std::is_floating_point_v<T>
    // probs and p_scores are dense
    __global__ void causal_softmax_forward_kernel_fast(
        T* __restrict__ p_probs, const T* p_scores,
        T scale, int64_t seq_len
    ) {
        // launch B * num_heads * T blocks
        int64_t row = blockIdx.x;
        int64_t tid = threadIdx.x;

        int64_t seq_row = row % seq_len; // which row in the T x T matrix (causal condition)
        const T* scores_row = p_scores + (row * seq_len); // jump into our row in scores
        T* probs_row = p_probs + (row * seq_len); // jump into our row in probs

        T thread_max = -INFINITY;
        for (int64_t i = tid; i < seq_len; i += blockDim.x) {
            if (i <= seq_row) { // causal condition - if (i >= j)
                T val = scores_row[i] * scale;
                thread_max = max(thread_max, val);
            }
        }

        __shared__ T s_scratch[256];
        s_scratch[tid] = thread_max;
        __syncthreads();

        for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_scratch[tid] = max(s_scratch[tid], s_scratch[tid + s]);
            }
            __sync_threads();
        }

        T row_max = s_scratch[0];
        __syncthreads();
        // sync so that thread_sum doesnt overwrite s_scratch[0]

        T thread_sum = 0;
        for (int64_t i = tid; i < seq_len; i += blockDim.x) {
            if (i <= seq_row) {
                thread_sum += exp((p_scores[i] * scale) - row_max);
            }
        }
        s_scratch[tid] = thread_sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s >= 0; s >>= 1) {
            if (tid < s) {
                s_scratch[tid] += s_scratch[tid + s];
            }
            __syncthreads();
        }

        T row_sum = s_scratch[0];
        __syncthreads();

        for (int64_t i = tid; i < seq_len; i += blockDim.x) {
            if (i <= seq_row) {
                p_probs[i] = exp((p_scores[i] * scale) - row_max) / row_sum;
            }
            else {
                p_probs[i] = 0;
            }
        }
    }

    template <typename T> 
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_causal_softmax_forward(Tensor<T>& probs, const Tensor<T>& scores, T scale, int64_t seq_len) {
        cudaSetDevice(probs.device().index);
        int64_t threads = 256;
        int64_t blocks = probs.shape()[0] * probs.shape()[1] * probs.shape()[2];


        T* p_probs = probs._get_storage()->data();
        const T* p_scores = scores._get_storage()->data();

        causal_softmax_forward_kernel_fast<<<blocks, threads>>>(p_probs, p_scores, scale, seq_len);
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    // dx, out_grad, probs are dense
    __global__ void causal_softmax_backward_kernel_fast(
        T* __restrict__ p_dx, 
        const T* __restrict__ p_out_grad, const T* __restrict__ p_probs,
        T scale, int64_t seq_len
    ) {
        // launch B * num_heads * T blocks again
        int64_t row = blockIdx.x;
        int64_t tid = threadIdx.x;

        int64_t seq_row = row % seq_len;

        const T* out_grad_row = p_out_grad + (row * seq_len);
        const T* probs_row = p_probs + (row * seq_len);
        T* dx_row = p_dx + (row * seq_len);

        __shared__ T s_scratch[256];

        T thread_sum = 0;
        for (int64_t i = tid; i < seq_len; i += blockDim.x) {
            if (i <= seq_row) {
                thread_sum += out_grad_row[i] * probs_row[i];
            }
        }
        s_scratch[tid] = thread_sum;
        __syncthreads();

        for (int64_t s = blockDim.x / 2; s >= 0; s >>= 1) {
            if (tid < s) {
                s_scratch[tid] += s_scratch[tid + s];
            }
            __syncthreads();
        }

        T row_sum = s_scratch[0];
        
        for (int64_t i = tid; i < seq_len; i += blockDim.x) {
            if (i <= seq_row) {
                dx_row = scale * probs_row[i] * (out_grad_row[i] - row_sum);
            }
            else {
                dx_row[i] = 0;
            }
        }
    }
    
    template <typename T> 
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_causal_softmax_backward(Tensor<T>& dx, const Tensor<T>& out_grad, const Tensor<T>& probs, T scale, int64_t seq_len) {
        cudaSetDevice(probs.device().index);
        int64_t threads = 256;
        int64_t blocks = probs.shape()[0] * probs.shape()[1] * probs.shape()[2];

        T* p_dx = dx._get_storage()->data();
        const T* p_out_grad = out_grad._get_storage()->data();
        const T* p_probs = probs._get_storage()->data();

        causal_softmax_backward_kernel_fast<<<blocks, threads>>>(p_dx, p_out_grad, p_probs, scale, seq_len);
    }

    

    #pragma endregion CAUSAL SOFTMAX

    #pragma region TEMPLATING

    #define INSTANTIATE_CUDA_MATH_SINGLE(T) \
        template void CUDAMath::apply_binary_out_of_place<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, BinaryOp); \
        template void CUDAMath::apply_binary_in_place<T>(Tensor<T>&, const Tensor<T>&, BinaryOpInPlace); \
        template void CUDAMath::apply_unary_out_of_place<T>(Tensor<T>&, const Tensor<T>&, UnaryOp); \
        template void CUDAMath::apply_unary_in_place<T>(Tensor<T>&, UnaryOpInPlace); \
        template void CUDAMath::apply_reduction_operation<T>(Tensor<T>&, const Tensor<T>&, const RedMeta&, ReduceOp); \
        template void CUDAMath::apply_arg_extr_operation<T>(Tensor<int64_t>&, const Tensor<T>&, int64_t, ArgExtrOp); \
        template void CUDAMath::apply_embed<T>(Tensor<T>&, const Tensor<int64_t>&, const Tensor<T>&, int64_t); \
        template void CUDAMath::apply_scatter_add(Tensor<T>&, const Tensor<int64_t>&, const Tensor<T>&, int64_t);

    #define INSTANTIATE_CAST(OutT, InT) \
        template void CUDAMath::apply_cast_out_of_place<OutT, InT>(Tensor<OutT>&, const Tensor<InT>&);

    #define INSTANTIATE_CUDA_MATMUL(T) \
        template void CUDAMath::apply_normal_gemm<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const NMMMeta&); \
        template void CUDAMath::apply_batched_gemm<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const BMMMeta&);

    #define INSTANTIATE_CUDA_CASTS_ALL_OUTS(InT) \
        INSTANTIATE_CAST(float, InT) \
        INSTANTIATE_CAST(double, InT) \
        INSTANTIATE_CAST(int32_t, InT) \
        INSTANTIATE_CAST(int64_t, InT) 

    INSTANTIATE_CUDA_MATH_SINGLE(float)
    INSTANTIATE_CUDA_MATH_SINGLE(double)
    INSTANTIATE_CUDA_MATH_SINGLE(int32_t)
    INSTANTIATE_CUDA_MATH_SINGLE(int64_t)

    INSTANTIATE_CUDA_CASTS_ALL_OUTS(float)
    INSTANTIATE_CUDA_CASTS_ALL_OUTS(double)
    INSTANTIATE_CUDA_CASTS_ALL_OUTS(int32_t)
    INSTANTIATE_CUDA_CASTS_ALL_OUTS(int64_t)

    INSTANTIATE_CUDA_MATMUL(float)
    INSTANTIATE_CUDA_MATMUL(double)

    #pragma endregion TEMPLATING
}