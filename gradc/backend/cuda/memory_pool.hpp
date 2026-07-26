#pragma once

#include <cuda_runtime.h>
#include "../../core/types.hpp"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <format>

namespace gradc {
    class CUDAMemPool {
    private:
        int m_device_count = 0;
        std::vector<std::unordered_map<int64_t, std::vector<void*>>> m_free_blocks;
        CUDAMemPool() {
            cudaGetDeviceCount(&m_device_count);
            m_free_blocks.reserve(m_device_count);
        }
    public:
        CUDAMemPool(const CUDAMemPool&) = delete;
        CUDAMemPool& operator=(const CUDAMemPool) = delete;

        static CUDAMemPool& get() {
            static CUDAMemPool inst;
            return inst;
        }

        void* allocate(int64_t bytes, Device device) {
            if (device.index >= m_device_count) {
                std::string error_msg = std::format("Invalid GPU index (>=): {}. Available GPUs: {}", device.index, m_device_count);
                throw std::runtime_error(error_msg);
            }
            cudaSetDevice(device.index);
            std::vector<void*>& blocks = m_free_blocks[device.index][bytes];
            if (!blocks.empty()) {
                void* ptr = blocks.back();
                blocks.pop_back();
                return ptr;
            }
            else {
                void* ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, bytes);

                if (err != cudaSuccess) {
                    clear(device);
                    ptr = nullptr;
                    err = cudaMalloc(&ptr, bytes);
                    if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
                    }
                    else {
                        return ptr;
                    }
                }
                else {
                    return ptr;
                }
            }
        }

        void free(void* ptr, int64_t bytes, Device device) {
            if (device.index >= m_device_count) {
                std::string error_msg = std::format("Invalid GPU index (>=): {}. Available GPUs: {}", device.index, m_device_count);
                throw std::runtime_error(error_msg);
            }
            cudaSetDevice(device.index);
            if (ptr != nullptr) {
                m_free_blocks[device.index][bytes].push_back(ptr);
            }
        }

        void clear(Device device) {
            if (device.index >= m_device_count) {
                std::string error_msg = std::format("Invalid GPU index (>=): {}. Available GPUs: {}", device.index, m_device_count);
                throw std::runtime_error(error_msg);
            }
            cudaSetDevice(device.index);
            auto& device_blocks = m_free_blocks[device.index];
            for (auto& [size, available_blocks] : device_blocks) {
                for (void* ptr : available_blocks) {
                    cudaFree(ptr);
                }
            }
            m_free_blocks[device.index].clear();
        }
    };
}
