#include "gradc/backend/cuda/cuda_math.hpp"
#include <cuda_runtime.h>
#include "gradc/core/detail/shape_inference.hpp"
#include "gradc/backend/cuda/math_functors.cuh"
#include "gradc/backend/cuda/cuda_utils.hpp"

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

    #pragma region BINARY OUT OF PLACE

    template <typename T, typename Func>
    __global__ void binary_out_of_place_kernel(
        T* p_out, const T* p_left, const T* p_right, 
        const CUDAMeta shared_shape, 
        const CUDAMeta out_strides, const CUDAMeta left_strides, const CUDAMeta right_strides,
        const int64_t out_offset, const int64_t left_offset, const int64_t right_offset,
        const int64_t total_elements, const Func op) {

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
        const int64_t out_offset, const int64_t left_offset, const int64_t right_offset,
        const int64_t total_elements, const Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            p_out[out_offset + linear_idx] = op(p_left[left_offset + linear_idx], p_right[right_offset + linear_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_binary_out_of_place(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, BinaryOp op) {
        int64_t total_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_out = out._get_storage()->data();
        const T* p_left = left._get_storage()->data();
        const T* p_right = right._get_storage()->data(); 
        
        if (out.is_contiguous() && left.is_contiguous() && right.is_contiguous() && out.m_shape == left.m_shape && left.m_shape == right.m_shape) {
            switch (op) {
                case BinaryOp::Add:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::Add<T>());
                    break;
                case BinaryOp::Sub:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::Sub<T>());
                    break;
                case BinaryOp::Mul:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::Mul<T>());
                    break;
                case BinaryOp::Div:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::Div<T>());
                    break;
                case BinaryOp::ReLUBackward:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::ReLUBackward<T>());
                    break;
                case BinaryOp::EqMask:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::EqMask<T>());
                    break;
                default:
                    throw std::runtime_error("Unsupported CUDA BOOP FAST");
            }
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

        cudaSetDevice(out.device().index);

        switch (op) {
            case BinaryOp::Add:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::Add<T>());
                break;
            case BinaryOp::Sub:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::Sub<T>());
                break;
            case BinaryOp::Mul:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::Mul<T>());
                break;
            case BinaryOp::Div:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::Div<T>());
                break;
            case BinaryOp::ReLUBackward:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::ReLUBackward<T>());
                break;
            case BinaryOp::EqMask:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::EqMask<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA BOOP SLOW");
        }
    }

    #pragma endregion BINARY OUT OF PLACE
    #pragma region BINARY IN PLACE

    template <typename T, typename Func>
    __global__ void binary_in_place_kernel_fast(
        T* p_left, const T* p_right, 
        const int64_t left_offset, const int64_t right_offset,
        const int64_t total_elements, const Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            op(p_left[left_offset + linear_idx], p_right[right_offset + linear_idx]);
        }
    }

    template <typename T, typename Func>
    __global__ void binary_in_place_kernel(
        T* p_left, const T* p_right, 
        const CUDAMeta shared_shape, 
        const CUDAMeta left_strides, const CUDAMeta right_strides,
        const int64_t left_offset, const int64_t right_offset,
        const int64_t total_elements, const Func op) {

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
        int64_t total_elems = left.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_left = left._get_storage()->data();
        const T* p_right = right._get_storage()->data(); 

        if (left.m_shape == right.m_shape && left.is_contiguous() && right.is_contiguous()) {
            switch (op) {
            case BinaryOpInPlace::Add:
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::Add<T>());
                break;
            case BinaryOpInPlace::Sub:
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::Sub<T>());
                break;
            case BinaryOpInPlace::Mul:
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::Mul<T>());
                break;
            case BinaryOpInPlace::Div:
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::Div<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA BIP FAST");
            }
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
        std::vector<int64_t>* left_strides = &fused.strides[0];
        right_strides = &fused.strides[1];


        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_left_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_right_strides = to_cuda_meta(fused.strides[1]);

        cudaSetDevice(left.device().index);

        switch (op) {
            case BinaryOpInPlace::Add:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::Add<T>());
                break;
            case BinaryOpInPlace::Sub:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::Sub<T>());
                break;
            case BinaryOpInPlace::Mul:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::Mul<T>());
                break;
            case BinaryOpInPlace::Div:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::Div<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA BIP SLOW");
        }
    }

    #pragma endregion BINARY IN PLACE
    #pragma region UNARY OUT OF PLACE

    template <typename OutT, typename InT, typename Func>
    __global__ void unary_out_of_place_kernel(
        OutT* p_out, const InT* p_source, 
        const CUDAMeta shared_shape, 
        const CUDAMeta out_strides, const CUDAMeta source_strides,
        const int64_t out_offset, const int64_t source_offset,
        const int64_t total_elements, const Func op) {

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
        const int64_t out_offset, const int64_t source_offset,
        const int64_t total_elements, const Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            p_out[out_offset + linear_idx] = op(p_source[source_offset + linear_idx]);
        }
    }

    template <typename OutT, typename InT>
    void CUDAMath::apply_unary_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source, UnaryOp op) {
        int64_t total_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        OutT* p_out = out._get_storage()->data();
        const InT* p_source = source._get_storage()->data(); 

        if (out.is_contiguous() && source.is_contiguous() && out.m_shape == source.m_shape) {
            switch (op) {
                case UnaryOp::Identity:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Identity<InT>());
                    break;
                case UnaryOp::Cast:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Cast<InT, OutT>());
                    break;
                case UnaryOp::ReLU:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::ReLU<InT>());
                    break;
                case UnaryOp::Exp:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Exp<InT>());
                    break;
                case UnaryOp::Log:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Log<InT>());
                    break;
                default:
                    throw std::runtime_error("Unsupported CUDA UOP");
            }
            return;
        }

        FusedView fused = fuse_dimensions(out.m_shape, {&out.m_strides, &source.m_strides});
        const std::vector<int64_t>* out_strides = &fused.strides[0];
        const std::vector<int64_t>* source_strides = &fused.strides[1];

        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_out_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[1]);

        cudaSetDevice(out.device().index);

        switch (op) {
            case UnaryOp::Identity:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Identity<InT>());
                break;
            case UnaryOp::Cast:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Cast<InT, OutT>());
                break;
            case UnaryOp::ReLU:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::ReLU<InT>());
                break;
            case UnaryOp::Exp:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Exp<InT>());
                break;
            case UnaryOp::Log:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOP::Log<InT>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA UOP");
        }
    }

    #pragma endregion UNARY OUT OF PLACE
    #pragma region UNARY IN PLACE

    template <typename T, typename Func>
    __global__ void unary_in_place_kernel(
        T* p_source, 
        const CUDAMeta shared_shape, 
        const CUDAMeta source_strides,
        const int64_t source_offset,
        const int64_t total_elements, const Func op) {

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
        const int64_t source_offset,
        const int64_t total_elements, const Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {
            op(p_source[source_offset + linear_idx]);
        }
    }

    template <typename T>
    void CUDAMath::apply_unary_in_place(Tensor<T>& source, UnaryOpInPlace op) {
        int64_t total_elems = source.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_source = source._get_storage()->data();

        if (source.is_contiguous()) {
            switch (op) {
                case UnaryOpInPlace::ReLU:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::ReLU<T>());
                    break;
                case UnaryOpInPlace::Exp:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::Exp<T>());
                    break;
                case UnaryOpInPlace::Log:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::Log<T>());
                    break;
                default:
                    throw std::runtime_error("Unsupported CUDA UIP");
            }
            return;
        }

        FusedView fused = fuse_dimensions(source.m_shape, {&source.m_strides});
        
        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[0]);

        cudaSetDevice(source.device().index);

        switch (op) {
            case UnaryOpInPlace::ReLU:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::ReLU<T>());
                break;
            case UnaryOpInPlace::Exp:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::Exp<T>());
                break;
            case UnaryOpInPlace::Log:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::Log<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA UIP");
        }

    }

    #pragma endregion UNARY IN PLACE
    #pragma region REDUCTIONS

    template <typename T, typename Func>
    __global__ void reduction_kernel(
        T* p_out, const T* p_source, 
        const CUDAMeta shared_shape, 
        const CUDAMeta out_strides, const CUDAMeta source_strides,
        const int64_t source_offset,
        const int64_t total_elements, const Func op) {

        int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (linear_idx < total_elements) {

            int64_t temp_idx = linear_idx;

            int64_t out_strided_idx = 0;
            int64_t source_strided_idx = source_offset;

            for (int64_t i = shared_shape.size - 1; i >= 0; --i) {
                int64_t coord = temp_idx % shared_shape.data[i];
                temp_idx /= shared_shape.data[i];

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
        const int64_t total_elements, const Func op) {
        
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

    // START OFF HERE
    template <typename T, typename Func>
    __global__ void extr_kernel_whole(
        int64_t* p_out, const T* p_source, 
        int64_t source_offset, T init_value,
        const int64_t total_elements, const Func op) {

        __shared__ T shared_maximums[256];
        __shared__ int64_t shared_indices[256];

        int64_t t_idx = threadIdx.x;
        int64_t  global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (global_idx < total_elements) {
            shared_maximums[t_idx] = p_source[source_offset + global_idx];
            shared_indices[t_idx] = -1;
        }
        else {
            shared_maximums[t_idx] = init_value;
            shared_indices[t_idx] = -1;
        }
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (t_idx < stride) {
                shared_maximums[t_idx] = op(shared_data[t_idx], shared_data[t_idx + stride]);
            } 
            __syncthreads(); // wait till halving is done
        }

        if (t_idx == 0) {
            op.atomic(&p_out[0], shared_data[0]);
        }
    }

    template <typename T>
    void CUDAMath::apply_reduction_operation(Tensor<T>& out, const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, ReduceOp op) {
        int64_t total_elems = source.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        T* p_out = out._get_storage()->data();
        T* p_source = source._get_storage()->data();

        if (out.volume() == 1 && source.is_contiguous()) {
            switch (op) {
                case ReduceOp::Sum:
                    reduction_kernel_whole<<<blocks, threads>>>(p_out, p_source, source.m_offset, init_value, total_elems, cuda_functors::RED::Sum<T>());
                    break;
                case ReduceOp::Max:
                    reduction_kernel_whole<<<blocks, threads>>>(p_out, p_source, source.m_offset, init_value, total_elems, cuda_functors::RED::Max<T>());
                    break;
                case ReduceOp::Min:
                    reduction_kernel_whole<<<blocks, threads>>>(p_out, p_source, source.m_offset, init_value, total_elems, cuda_functors::RED::Min<T>());
                    break;
                default:
                    throw std::runtime_error("Unsupported CUDA ReduceOp");  
            }
            return;
        }

        CUDAMeta shared_shape = to_cuda_meta(source.m_shape);
        CUDAMeta out_strides = to_cuda_meta(reduction_metadata.temp_strides);
        CUDAMeta source_strides = to_cuda_meta(source.m_strides);

        CUDAUtils::fill(p_out, init_value, out.volume(), out.device());

        switch (op) {
            case ReduceOp::Sum:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, shared_shape, out_strides, source_strides, source.m_offset, total_elems, cuda_functors::RED::Sum<T>());
                break;
            case ReduceOp::Max:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, shared_shape, out_strides, source_strides, source.m_offset, total_elems, cuda_functors::RED::Max<T>());
                break;
            case ReduceOp::Min:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, shared_shape, out_strides, source_strides, source.m_offset, total_elems, cuda_functors::RED::Min<T>());
                break;
            case ReduceOp::ArgMax:
                break;
            case ReduceOp::ArgMin:
                break;
            default:
                throw std::runtime_error("Unsupported CUDA ReduceOp");
        }
    }

    #pragma endregion REDUCTIONS

    #pragma region TEMPLATING

    #define INSTANTIATE_CUDA_MATH_SINGLE(T) \
        template void CUDAMath::apply_binary_out_of_place<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, BinaryOp); \
        template void CUDAMath::apply_binary_in_place<T>(Tensor<T>&, const Tensor<T>&, BinaryOpInPlace); \
        template void CUDAMath::apply_unary_in_place<T>(Tensor<T>&, UnaryOpInPlace); \
        template void CUDAMath::apply_reduction_operation<T>(Tensor<T>&, const Tensor<T>&, const ReductionMetadata&, T, ReduceOp);

    #define INSTANTIATE_CUDA_MATH_UOP(OutT, InT) \
        template void CUDAMath::apply_unary_out_of_place<OutT, InT>(Tensor<OutT>&, const Tensor<InT>&, UnaryOp);

    #define INSTANTIATE_CUDA_UOP_ALL_OUTS(InT) \
        INSTANTIATE_CUDA_MATH_UOP(float, InT) \
        INSTANTIATE_CUDA_MATH_UOP(double, InT) \
        INSTANTIATE_CUDA_MATH_UOP(int32_t, InT) \
        INSTANTIATE_CUDA_MATH_UOP(int64_t, InT) 

    INSTANTIATE_CUDA_MATH_SINGLE(float)
    INSTANTIATE_CUDA_MATH_SINGLE(double)
    INSTANTIATE_CUDA_MATH_SINGLE(int32_t)
    INSTANTIATE_CUDA_MATH_SINGLE(int64_t)

    INSTANTIATE_CUDA_UOP_ALL_OUTS(float)
    INSTANTIATE_CUDA_UOP_ALL_OUTS(double)
    INSTANTIATE_CUDA_UOP_ALL_OUTS(int32_t)
    INSTANTIATE_CUDA_UOP_ALL_OUTS(int64_t)

    #pragma endregion TEMPLATING
}