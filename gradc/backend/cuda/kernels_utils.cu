#include "gradc/backend/cuda/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <curand.h>

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
        int64_t threads = 256;
        int64_t blocks = (size + threads - 1) / threads;

        fill_kernel<<<blocks, threads>>>(ptr, val, size);
    }

    template <typename T>
    __global__ void fill_arange_kernel(T* ptr, T start, T step, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            ptr[idx] = start + step * idx;
        }
    }

    template <typename T>
    void CUDAUtils::fill_arange(T* ptr, T start, T step, int64_t size, Device device) {
       cudaSetDevice(device.index);
       int64_t threads = 256;
       int64_t blocks = (size + threads - 1) / threads;

       fill_arange_kernel<<<blocks, threads>>>(ptr, start, step, size);
    }

    template <typename T>
    void CUDAUtils::fill_normal(T* ptr, T mean, T std, int64_t size, Device device) {
        cudaSetDevice(device.index);
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234);

        if constexpr (std::is_same_v<T, float>) {
            curandGenerateNormal(gen, ptr, size, mean, std);
        }
        else if constexpr (std::is_same_v<T, double>) {
            curandGenerateNormalDouble(gen, ptr, size, mean, std);
        }

        curandDestroyGenerator(gen);
    }

    template <typename T>
    __global__ void scale_uniform_kernel(T* ptr, T low, T high, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < size) {
            ptr[idx] = ptr[idx] * (high - low) + low;
        }
    }

    template <typename T>
    void CUDAUtils::fill_uniform(T* ptr, T low, T high, int64_t size, Device device) {
        cudaSetDevice(device.index);
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234);

        if constexpr (std::is_same_v<T, float>) {
            curandGenerateUniform(gen, ptr, size);
        }
        else if constexpr (std::is_same_v<T, double>) {
            curandGenerateUniformDouble(gen, ptr, size);
        }

        curandDestroyGenerator(gen);

        int64_t threads = 256;
        int64_t blocks = (size + threads - 1) / threads;
        scale_uniform_kernel<<<blocks, threads>>>(ptr, low, high, size);

    }
    
    template void CUDAUtils::fill<float>(float* ptr, float val, int64_t size, Device device);
    template void CUDAUtils::fill<double>(double* ptr, double val, int64_t size, Device device);
    template void CUDAUtils::fill<int32_t>(int32_t* ptr, int32_t val, int64_t size, Device device);
    template void CUDAUtils::fill<int64_t>(int64_t* ptr, int64_t val, int64_t size, Device device);

    template void CUDAUtils::fill_arange<float>(float* ptr, float start, float step, int64_t size, Device device);
    template void CUDAUtils::fill_arange<double>(double* ptr, double start, double step, int64_t size, Device device);
    template void CUDAUtils::fill_arange<int32_t>(int32_t* ptr, int32_t start, int32_t step, int64_t size, Device device);
    template void CUDAUtils::fill_arange<int64_t>(int64_t* ptr, int64_t start, int64_t step, int64_t size, Device device);

    
}