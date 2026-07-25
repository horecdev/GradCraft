#pragma once

#include "types.hpp"
#include "../backend/cuda/cuda_utils.hpp"
#include "../backend/cpu/memory_pool.hpp"
#include "../backend/cuda/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <malloc.h>
#include <stdexcept>

namespace gradc {
    template <typename T>
    struct Storage{
        T* m_data = nullptr; // ptr to start of chunk of memory
        int64_t m_size = 0;
        Device m_device;

        Storage() : m_data(nullptr), m_size(0), m_device(Device(DeviceType::CPU)) {}

        Storage(int64_t size, T init_val = T(), Device device = Device(DeviceType::CPU), bool allocate = true, bool fill = true) : m_size(size), m_device(device) {
            int64_t bytes = size * sizeof(T);
                if (allocate) {
                    if (m_device.is_cpu()) {
                        int64_t aligned_bytes = ((bytes + 31) / 32) * 32; // must be 32 multiple
                        // by default _aligned_malloc returns void* (pointer to very first byte) so u cast it to T*
                        m_data = static_cast<T*>(CPUMemPool::get().allocate(aligned_bytes));
                        if (fill) {
                            std::fill(m_data, m_data + m_size, init_val); // m_data + m_size does pointer arithmetic (applies for sizeof)
                        }
                    }
                    
                    else if (m_device.is_cuda()) {
                        m_data = static_cast<T*>(CUDAMemPool::get().allocate(bytes, device));
                        if (fill) {
                            CUDAUtils::fill(m_data, init_val, size, device);
                        }
                    }
                }
        }

        Storage(std::initializer_list<T> data, Device device = Device(DeviceType::CPU)) : m_size(std::ssize(data)), m_device(device) {
            if (m_size < 1) {
                throw std::runtime_error("Storage constructed with empty initializer_list. Use a different constructor.");
            }

            if (m_device.is_cpu()) {
                int64_t bytes = m_size * sizeof(T);
                int64_t aligned_bytes = ((bytes + 31) / 32) * 32; 
                m_data = static_cast<T*>(CPUMemPool::get().allocate(aligned_bytes));
                
                std::memcpy(m_data, data.begin(), data.size() * sizeof(T)); 
            }

            else if (device.is_cuda()) {

                int64_t bytes = m_size * sizeof(T);
                m_data = static_cast<T*>(CUDAMemPool::get().allocate(bytes, device));

                cudaSetDevice(device.index);
                cudaError_t err = cudaMemcpy(m_data, data.begin(), bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed initializing a storage with an initializer list.");
                }
            }
        }

        T* data() const {
            return m_data;
        }

        int64_t size() const {
            return m_size;
        }

        Device device() const {
            return m_device;
        }

        ~Storage() {
            int64_t bytes = m_size * sizeof(T);
            if (m_device.is_cpu()) {
                int64_t aligned_bytes = ((bytes + 31) / 32) * 32;
                CPUMemPool::get().free(m_data, aligned_bytes); // accepts void* but any pointer can implicitly convert to void*
            }
            else if (m_device.is_cuda()) {
                CUDAMemPool::get().free(m_data, bytes, m_device);
            }
        }

        Storage(const Storage&) = delete; // since we manually free memory copying should not exist.
        Storage& operator=(const Storage&) = delete; // we dont even copy storage anywhere so its cool
        
        Storage(Storage&& other) : m_size(other.m_size), m_device(other.m_device) {
            m_data = other.m_data;
            other.m_data = nullptr;
        }

        Storage& operator=(Storage&& other) {
            if (this != &other) {
                m_data = other.m_data;
                other.m_data = nullptr;
                m_size = other.m_size;
                m_device = other.m_device;
            }
            return *this;
        } 
    };
}
