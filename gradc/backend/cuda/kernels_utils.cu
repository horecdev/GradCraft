#include "gradc/backend/cuda/cuda_utils.hpp"
#include <cuda_runtime.h>

namespace gradc {
    template <typename T>
    __global__ void fill_kernel(T* ptr, T val, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            ptr[idx] = val;
        }
    }

    template <typename T>
    void CUDAUtils::fill(T* ptr, T val, int64_t size, Device device) {
        cudaSetDevice(device.index);
        int32_t threads = 256;
        int64_t blocks = (size + threads - 1) / threads;

        fill_kernel<<<blocks, threads>>>(ptr, val, size);
    }
    
    template void CUDAUtils::fill<float>(float* ptr, float val, int64_t size, Device device);
    template void CUDAUtils::fill<double>(double* ptr, double val, int64_t size, Device device);
    template void CUDAUtils::fill<int32_t>(int32_t* ptr, int32_t val, int64_t size, Device device);
    template void CUDAUtils::fill<int64_t>(int64_t* ptr, int64_t val, int64_t size, Device device);

    
}