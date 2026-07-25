#include "gradc/backend/cuda/cuda_math.hpp"
#include <cuda_runtime.h>
#include "gradc/core/detail/shape_inference.hpp"

namespace gradc {
    constexpr int MAX_DIMS = 8;

    struct GPUMeta { // when you copy a struct with an array, it doesnt decay. The compiler respects struct's size.
        int64_t data[MAX_DIMS];
        int size;
    };

    template <typename T>
    __global__ void binary_out_of_place_kernel(T* ptr, T val, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            ptr[idx] = val;
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
        std::vector<int64_t>* out_strides = &fused.strides[0] ;
        left_strides = &fused.strides[1];
        right_strides = &fused.strides[2];

        // TODO: Write the function to ship vectors to the GPUMeta struct (it is copied to GPU constant memory when u invoke kernel and pass by value, doesnt live in a vector)
        // Then ship the strides over PCIE bus, pass in raw pointers (cannot pass struct like Storage or Tensor cuz its CPU)
    }
    
    template void CUDAMath::apply_binary_out_of_place<float>(Tensor<float>& out, const Tensor<float>& left, const Tensor<float>& right, BinaryOp op);
    template void CUDAMath::apply_binary_out_of_place<double>(double* ptr, double val, int64_t size, Device device);
    template void CUDAMath::apply_binary_out_of_place<int16_t>(int16_t* ptr, int16_t val, int64_t size, Device device);
    template void CUDAMath::apply_binary_out_of_place<int32_t>(int32_t* ptr, int32_t val, int64_t size, Device device);
    template void CUDAMath::apply_binary_out_of_place<int64_t>(int64_t* ptr, int64_t val, int64_t size, Device device);

    
}