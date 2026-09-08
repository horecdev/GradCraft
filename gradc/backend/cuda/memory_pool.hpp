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
        int64_t m_current_usage = 0;
        int64_t m_hwm = 0;
        CUDAMemPool() {
            cudaGetDeviceCount(&m_device_count);
            m_free_blocks.resize(m_device_count);
        }
    public:
        CUDAMemPool(const CUDAMemPool&) = delete;
        CUDAMemPool& operator=(const CUDAMemPool) = delete;

        static CUDAMemPool& get() {
            static CUDAMemPool inst;
            return inst;
        }

        float get_hwm_gb() {
            return static_cast<float>(m_hwm) / (1024 * 1024 * 1024);
        }

        void log_hwm() {
            std::cout << "Current HWM: " << get_hwm_gb() << " GB" << std::endl;
        }

        void* allocate(int64_t bytes, Device device) {
            if (device.index >= m_device_count) {
                std::string error_msg = std::format("Invalid GPU index (>=): {}. Available GPUs: {}", device.index, m_device_count);
                throw std::runtime_error(error_msg);
            }

            cudaSetDevice(device.index);
            std::vector<void*>& blocks = m_free_blocks[device.index][bytes];
            void* ptr = nullptr;

            if (!blocks.empty()) {
                ptr = blocks.back();
                blocks.pop_back();
            }
            else {
                cudaError_t err = cudaMallocAsync(&ptr, bytes, 0);

                if (err != cudaSuccess) {
                    clear(device);
                    ptr = nullptr;
                    err = cudaMallocAsync(&ptr, bytes, 0);
                    if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
                    }
                }
            }

            m_current_usage += bytes;
            if (m_current_usage > m_hwm) {
                m_hwm = m_current_usage;
            }
            
            return ptr;
        }

        void free(void* ptr, int64_t bytes, Device device) {
            if (device.index >= m_device_count) {
                std::string error_msg = std::format("Invalid GPU index (>=): {}. Available GPUs: {}", device.index, m_device_count);
                throw std::runtime_error(error_msg);
            }

            cudaSetDevice(device.index);
            if (ptr != nullptr) {
                m_free_blocks[device.index][bytes].push_back(ptr);
                m_current_usage -= bytes;
            }
        }

        void clear(Device device) {
            if (device.index >= m_device_count) {
                std::string error_msg = std::format("Invalid GPU index (>=): {}. Available GPUs: {}", device.index, m_device_count);
                throw std::runtime_error(error_msg);
            }
            std::cout << "Clearing mempool" << std::endl;
            cudaSetDevice(device.index);
            auto& device_blocks = m_free_blocks[device.index];
            for (auto& [size, available_blocks] : device_blocks) {
                for (void* ptr : available_blocks) {
                    cudaFreeAsync(ptr, 0);
                }
            }
            m_free_blocks[device.index].clear();
        }
    };
}
