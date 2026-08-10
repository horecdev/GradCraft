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
        T* p_out, const T* p_left, const T* p_right, 
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
        T* p_out, const T* p_left, const T* p_right, 
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
        T* p_left, const T* p_right, 
        int64_t left_offset, int64_t right_offset,
        int64_t total_elements, Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            op(p_left[left_offset + linear_idx], p_right[right_offset + linear_idx]);
        }
    }

    template <typename T, typename Func>
    __global__ void binary_in_place_kernel(
        T* p_left, const T* p_right, 
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
        OutT* p_out, const InT* p_source, 
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
        OutT* p_out, const InT* p_source, 
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
        T* p_source, 
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
        T* p_source, 
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
        T* p_out, const T* p_source, 
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
        T* p_out, const T* p_source, 
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

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
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
        int64_t* p_out, const T* p_source, 
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

        for (int stride = blockDim.x; stride > 0; stride /= 2) {
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
        int64_t* p_out, const T* p_source, 
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

    #pragma region MATRIX MULTIPLY

    template <typename T>
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_batched_gemm(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, const NMMMeta& blas_meta) {
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
            cublasSgemm(handle, op_right, op_left, blas_meta.N, blas_meta.M, blas_meta.K, &typed_alpha, p_right, blas_meta.ldb, p_left, blas_meta.lda, &typed_beta, p_out, blas_meta.ldc);
        }
        else if constexpr (std::is_same_v<T, double>) {
            cublasDgemm(handle, op_right, op_left, blas_meta.N, blas_meta.M, blas_meta.K, &typed_alpha, p_right, blas_meta.ldb, p_left, blas_meta.lda, &typed_beta, p_out, blas_meta.ldc);
        }
    }

    #pragma endregion MATRIX MULTIPLY

    #pragma region TEMPLATING

    #define INSTANTIATE_CUDA_MATH_SINGLE(T) \
        template void CUDAMath::apply_binary_out_of_place<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, BinaryOp); \
        template void CUDAMath::apply_binary_in_place<T>(Tensor<T>&, const Tensor<T>&, BinaryOpInPlace); \
        template void CUDAMath::apply_unary_out_of_place<T>(Tensor<T>&, const Tensor<T>&, UnaryOp); \
        template void CUDAMath::apply_unary_in_place<T>(Tensor<T>&, UnaryOpInPlace); \
        template void CUDAMath::apply_reduction_operation<T>(Tensor<T>&, const Tensor<T>&, const RedMeta&, ReduceOp); \
        template void CUDAMath::apply_arg_extr_operation<T>(Tensor<int64_t>&, const Tensor<T>&, int64_t, ArgExtrOp);

    #define INSTANTIATE_CAST(OutT, InT) \
        template void CUDAMath::apply_cast_out_of_place<OutT, InT>(Tensor<OutT>&, const Tensor<InT>&);

    #define INSTANTIATE_CUDA_MATMUL(T) \
        template void CUDAMath::apply_batched_gemm<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const NMMMeta&);

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