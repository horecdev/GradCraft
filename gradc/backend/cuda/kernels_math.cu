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
                case BinaryOp::EqMask:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::EqMask<T>());
                    break;
                case BinaryOp::BExp:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BExp<T>());
                    break;
                case BinaryOp::BLog:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BLog<T>());
                    break;
                case BinaryOp::BReLU:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BReLU<T>());
                    break;
                case BinaryOp::BSigmoid:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BSigmoid<T>());
                    break;
                case BinaryOp::BTanH:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BTanH<T>());
                    break;
                case BinaryOp::BSiLU:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BSiLU<T>());
                    break;
                case BinaryOp::BGeLU:
                    binary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_left, p_right, out.m_offset, left.m_offset, right.m_offset, total_elems, cuda_functors::BOOP::BGeLU<T>());
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
            case BinaryOp::EqMask:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::EqMask<T>());
                break;
            case BinaryOp::BExp:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BExp<T>());
                break;
            case BinaryOp::BLog:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BLog<T>());
                break;
            case BinaryOp::BReLU:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BReLU<T>());
                break;
            case BinaryOp::BSigmoid:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BSigmoid<T>());
                break;
            case BinaryOp::BTanH:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BTanH<T>());
                break;
            case BinaryOp::BSiLU:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BSiLU<T>());
                break;
            case BinaryOp::BGeLU:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left_offset, right_offset, total_elems, cuda_functors::BOOP::BGeLU<T>());
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
            case BinaryOpInPlace::ISub: // left is to_be_subbed. right is main
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::ISub<T>());
                break;
            case BinaryOpInPlace::Mul:
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::Mul<T>());
                break;
            case BinaryOpInPlace::Div:
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::Div<T>());
                break;
            case BinaryOpInPlace::IDiv: // left is divisor. right is main
                binary_in_place_kernel_fast<<<blocks, threads>>>(p_left, p_right, left.m_offset, right.m_offset, total_elems, cuda_functors::BIP::IDiv<T>());
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
            case BinaryOpInPlace::ISub:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::ISub<T>());
                break;
            case BinaryOpInPlace::Mul:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::Mul<T>());
                break;
            case BinaryOpInPlace::Div:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::Div<T>());
                break;
            case BinaryOpInPlace::IDiv:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right_offset, total_elems, cuda_functors::BIP::IDiv<T>());
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
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Identity<InT>());
                    break;
                case UnaryOp::Cast:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Cast<InT, OutT>());
                    break;
                case UnaryOp::ReLU:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::ReLU<InT>());
                    break;
                case UnaryOp::Exp:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Exp<InT>());
                    break;
                case UnaryOp::Log:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Log<InT>());
                    break;
                case UnaryOp::Sigmoid:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Sigmoid<InT>());
                    break;
                case UnaryOp::TanH:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::TanH<InT>());
                    break;
                case UnaryOp::SiLU:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::SiLU<InT>());
                    break;
                case UnaryOp::GeLU:
                    unary_out_of_place_kernel_fast<<<blocks, threads>>>(p_out, p_source, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::GeLU<InT>());
                    break;
                default:
                    throw std::runtime_error("Unsupported CUDA UOOP");
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
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Identity<InT>());
                break;
            case UnaryOp::Cast:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Cast<InT, OutT>());
                break;
            case UnaryOp::ReLU:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::ReLU<InT>());
                break;
            case UnaryOp::Exp:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Exp<InT>());
                break;
            case UnaryOp::Log:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Log<InT>());
                break;
            case UnaryOp::Sigmoid:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::Sigmoid<InT>());
                break;
            case UnaryOp::TanH:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::TanH<InT>());
                break;
            case UnaryOp::SiLU:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::SiLU<InT>());
                break;
            case UnaryOp::GeLU:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, cuda_functors::UOOP::GeLU<InT>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA UOOP");
        }
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
                case UnaryOpInPlace::Sigmoid:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::Sigmoid<T>());
                    break;
                case UnaryOpInPlace::TanH:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::TanH<T>());
                    break;
                case UnaryOpInPlace::SiLU:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::SiLU<T>());
                    break;
                case UnaryOpInPlace::GeLU:
                    unary_in_place_kernel_fast<<<blocks, threads>>>(p_source, source.m_offset, total_elems, cuda_functors::UIP::GeLU<T>());
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
            case UnaryOpInPlace::Sigmoid:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::Sigmoid<T>());
                break;
            case UnaryOpInPlace::TanH:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::TanH<T>());
                break;
            case UnaryOpInPlace::SiLU:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::SiLU<T>());
                break;
            case UnaryOpInPlace::GeLU:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, cuda_functors::UIP::GeLU<T>());
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

        CUDAMeta source_shape = to_cuda_meta(source.m_shape);
        CUDAMeta out_strides = to_cuda_meta(reduction_metadata.temp_strides);
        CUDAMeta source_strides = to_cuda_meta(source.m_strides);

        CUDAUtils::fill(p_out, init_value, out.volume(), out.device());

        switch (op) {
            case ReduceOp::Sum:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, out_strides, source_strides, source.m_offset, total_elems, cuda_functors::RED::Sum<T>());
                break;
            case ReduceOp::Max:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, out_strides, source_strides, source.m_offset, total_elems, cuda_functors::RED::Max<T>());
                break;
            case ReduceOp::Min:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, out_strides, source_strides, source.m_offset, total_elems, cuda_functors::RED::Min<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA ReduceOp");
        }
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
    void CUDAMath::apply_arg_extr_operation(Tensor<int64_t>& out, const Tensor<T>& source, int64_t dim, T init_value, ArgExtrOp op) {
        int64_t total_elems = source.volume();
        int64_t* p_out = out._get_storage()->data();
        T* p_source = source._get_storage()->data();

        if (out.volume() == 1 && source.is_contiguous()) {
            switch (op) {
                case ArgExtrOp::ArgMax:
                    argextr_kernel_whole<<<1, 1024>>>(p_out, p_source, source.m_offset, init_value, total_elems, cuda_functors::ARGEXTR::ArgMax<T>());
                    break;
                case ArgExtrOp::ArgMin:
                    argextr_kernel_whole<<<1, 1024>>>(p_out, p_source, source.m_offset, init_value, total_elems, cuda_functors::ARGEXTR::ArgMin<T>());
                    break;
            }
        }

        int64_t out_elems = out.volume();
        int64_t threads = 256;
        int64_t blocks = (out_elems + threads - 1) / threads;

        CUDAMeta source_shape = to_cuda_meta(source.m_shape);
        CUDAMeta source_strides = to_cuda_meta(source.m_strides);

        switch (op) {
            case ArgExtrOp::ArgMax:
                argextr_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, source_strides, source.m_offset, dim, init_value, out_elems, cuda_functors::ARGEXTR::ArgMax<T>());
            case ArgExtrOp::ArgMin:
                argextr_kernel<<<blocks, threads>>>(p_out, p_source, source_shape, source_strides, source.m_offset, dim, init_value, out_elems, cuda_functors::ARGEXTR::ArgMin<T>());
        }
    }

    #pragma endregion REDUCTIONS

    #pragma region TEMPLATING

    #define INSTANTIATE_CUDA_MATH_SINGLE(T) \
        template void CUDAMath::apply_binary_out_of_place<T>(Tensor<T>&, const Tensor<T>&, const Tensor<T>&, BinaryOp); \
        template void CUDAMath::apply_binary_in_place<T>(Tensor<T>&, const Tensor<T>&, BinaryOpInPlace); \
        template void CUDAMath::apply_unary_in_place<T>(Tensor<T>&, UnaryOpInPlace); \
        template void CUDAMath::apply_reduction_operation<T>(Tensor<T>&, const Tensor<T>&, const ReductionMetadata&, T, ReduceOp); \
        template void CUDAMath::apply_arg_extr_operation<T>(Tensor<int64_t>&, const Tensor<T>&, int64_t, T, ArgExtrOp);

    #define INSTANTIATE_CUDA_MATH_UOOP(OutT, InT) \
        template void CUDAMath::apply_unary_out_of_place<OutT, InT>(Tensor<OutT>&, const Tensor<InT>&, UnaryOp);

    #define INSTANTIATE_CUDA_UOOP_ALL_OUTS(InT) \
        INSTANTIATE_CUDA_MATH_UOOP(float, InT) \
        INSTANTIATE_CUDA_MATH_UOOP(double, InT) \
        INSTANTIATE_CUDA_MATH_UOOP(int32_t, InT) \
        INSTANTIATE_CUDA_MATH_UOOP(int64_t, InT) 

    INSTANTIATE_CUDA_MATH_SINGLE(float)
    INSTANTIATE_CUDA_MATH_SINGLE(double)
    INSTANTIATE_CUDA_MATH_SINGLE(int32_t)
    INSTANTIATE_CUDA_MATH_SINGLE(int64_t)

    INSTANTIATE_CUDA_UOOP_ALL_OUTS(float)
    INSTANTIATE_CUDA_UOOP_ALL_OUTS(double)
    INSTANTIATE_CUDA_UOOP_ALL_OUTS(int32_t)
    INSTANTIATE_CUDA_UOOP_ALL_OUTS(int64_t)

    #pragma endregion TEMPLATING
}