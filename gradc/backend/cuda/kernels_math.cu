#include "gradc/backend/cuda/cuda_math.hpp"
#include <cuda_runtime.h>
#include "gradc/core/detail/shape_inference.hpp"
#include "gradc/backend/math_functors.hpp"
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

    template <typename T>
    void CUDAMath::apply_binary_out_of_place(Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right, BinaryOp op) {
        const std::vector<int64_t>* left_strides = &left.m_strides;
        const std::vector<int64_t>* right_strides = &right.m_strides;
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

        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_out_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_left_strides = to_cuda_meta(fused.strides[1]);
        CUDAMeta gpu_right_strides = to_cuda_meta(fused.strides[2]);

        T* p_out = out._get_storage()->data();
        const T* p_left = left._get_storage()->data();
        const T* p_right = left._get_storage()->data(); 

        int64_t total_elems = out.volume();
        int16_t threads = 256;
        int blocks = (total_elems + threads - 1) / threads;
        cudaSetDevice(out.device().index);

        switch (op) {
            case BinaryOp::Add:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left.m_offset, right.m_offset, total_elems, functors::BOOP::Add<T>());
                break;
            case BinaryOp::Sub:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left.m_offset, right.m_offset, total_elems, functors::BOOP::Sub<T>());
                break;
            case BinaryOp::Mul:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left.m_offset, right.m_offset, total_elems, functors::BOOP::Mul<T>());
                break;
            case BinaryOp::Div:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left.m_offset, right.m_offset, total_elems, functors::BOOP::Div<T>());
                break;
            case BinaryOp::ReLUBackward:
                binary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_left, p_right, gpu_shape, gpu_out_strides, gpu_left_strides, gpu_right_strides, out.m_offset, left.m_offset, right.m_offset, total_elems, functors::BOOP::ReLUBackward<T>());
            default:
                throw std::runtime_error("Unsupported CUDA BOOP");
        }
    }

    #pragma endregion BINARY OUT OF PLACE
    #pragma region BINARY IN PLACE

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

        T* p_left = left._get_storage()->data();
        const T* p_right = left._get_storage()->data(); 

        int64_t total_elems = left.volume();
        int16_t threads = 256;
        int blocks = (total_elems + threads - 1) / threads;
        cudaSetDevice(left.device().index);

        switch (op) {
            case BinaryOpInPlace::Add:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right.m_offset, total_elems, functors::BIP::Add<T>());
                break;
            case BinaryOpInPlace::Sub:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right.m_offset, total_elems, functors::BIP::Sub<T>());
                break;
            case BinaryOpInPlace::Mul:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right.m_offset, total_elems, functors::BIP::Mul<T>());
                break;
            case BinaryOpInPlace::Div:
                binary_in_place_kernel<<<blocks, threads>>>(p_left, p_right, gpu_shape, gpu_left_strides, gpu_right_strides, left.m_offset, right.m_offset, total_elems, functors::BIP::Div<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA BIP");
        }
    }

    #pragma endregion BINARY IN PLACE
    #pragma region UNARY OUT OF PLACE

    template <typename T, typename Func>
    __global__ void unary_out_of_place_kernel(
        T* p_out, const T* p_source, 
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

    template <typename OutT, typename InT>
    void CUDAMath::apply_unary_out_of_place(Tensor<OutT>& out, const Tensor<InT>& source, UnaryOp op) {
        FusedView fused = fuse_dimensions(out.m_shape, {&out.m_strides, &source.m_strides});
        const std::vector<int64_t>* out_strides = &fused.strides[0];
        const std::vector<int64_t>* source_strides = &fused.strides[1];


        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_out_strides = to_cuda_meta(fused.strides[0]);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[1]);

        OutT* p_out = out._get_storage()->data();
        const InT* p_source = source._get_storage()->data(); 

        int64_t total_elems = out.volume();
        int16_t threads = 256;
        int blocks = (total_elems + threads - 1) / threads;
        cudaSetDevice(out.device().index);

        switch (op) {
            case UnaryOp::Identity:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, functors::UOP::Identity<InT>());
                break;
            case UnaryOp::ReLU:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, functors::UOP::ReLU<InT>());
                break;
            case UnaryOp::Exp:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, functors::UOP::Exp<InT>());
                break;
            case UnaryOp::Log:
                unary_out_of_place_kernel<<<blocks, threads>>>(p_out, p_source, gpu_shape, gpu_out_strides, gpu_source_strides, out.m_offset, source.m_offset, total_elems, functors::UOP::Log<InT>());
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

    template <typename T>
    void CUDAMath::apply_unary_in_place(Tensor<T>& source, UnaryOpInPlace op) {

        FusedView fused = fuse_dimensions(source.m_shape, {&source.m_strides});
        
        CUDAMeta gpu_shape = to_cuda_meta(fused.shared_shape);
        CUDAMeta gpu_source_strides = to_cuda_meta(fused.strides[0]);

        T* p_source = source._get_storage()->data();

        int64_t total_elems = source.volume();
        int16_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;
        cudaSetDevice(source.device().index);

        switch (op) {
            case UnaryOpInPlace::ReLU:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, functors::UIP::ReLU<T>());
                break;
            case UnaryOpInPlace::Exp:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, functors::UIP::Exp<T>());
                break;
            case UnaryOpInPlace::Log:
                unary_in_place_kernel<<<blocks, threads>>>(p_source, gpu_shape, gpu_source_strides, source.m_offset, total_elems, functors::UIP::Log<T>());
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

            p_out[out_strided_idx] = op(p_source[source_strided_idx], p_out(out_strided_idx));
        }
    }

    template <typename T>
    static void apply_reduction_operation(Tensor<T>& out, const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, ReduceOp op) {
        

        // first fill the out tensor with init_value
        // then invoke the right op and pass it into the kernel
        T* p_out = out._get_storage()->data();
        T* p_source = source._get_storage()->data();

        CUDAMeta shared_shape = to_cuda_meta(source.m_shape);
        CUDAMeta out_strides = to_cuda_meta(reduction_metadata.temp_strides);
        CUDAMeta source_strides = to_cuda_meta(source.m_strides);

        CUDAUtils::fill(p_out, init_value, out.volume(), out.device());

        int64_t total_elems = source.volume();
        int64_t threads = 256;
        int64_t blocks = (total_elems + threads - 1) / threads;

        switch (op) {
            case ReduceOp::Sum:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, source.shape(), out_strides, source_strides, source.m_offset, total_elems, functors::RED::Sum<T>());
                break;
            case ReduceOp::Max:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, source.shape(), out_strides, source_strides, source.m_offset, total_elems, functors::RED::Max<T>());
                break;
            case ReduceOp::Min:
                reduction_kernel<<<blocks, threads>>>(p_out, p_source, source.shape(), out_strides, source_strides, source.m_offset, total_elems, functors::RED::Min<T>());
                break;
            default:
                throw std::runtime_error("Unsupported CUDA ReduceOp");
        }
    }


    #pragma endregion REDUCTIONS

}