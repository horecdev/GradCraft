#pragma once

#include <corecrt_malloc.h>
#include <cstdint>
#include <new>
#include <unordered_map>

class CPUMemPool {
    private:
        std::unordered_map<int64_t, std::vector<void*>> m_free_blocks;
        int64_t m_current_usage = 0;
        int64_t m_hwm = 0;

        CPUMemPool() = default; // deleting copy/move ctors removes also default constructor

    public:
        CPUMemPool(const CPUMemPool&) = delete;
        CPUMemPool& operator=(const CPUMemPool&) = delete;
        CPUMemPool(CPUMemPool&&) = delete;
        CPUMemPool& operator=(CPUMemPool&&) = delete;

        static CPUMemPool& get() { // returned by reference so there is no copy
            static CPUMemPool inst; // static (one) global variable in this class
            return inst;
        }

        float get_hwm_gb() {
            return static_cast<float>(m_hwm) / (1024 * 1024 * 1024);
        }

        void* allocate(int64_t aligned_bytes) {
            std::vector<void*>& blocks = m_free_blocks[aligned_bytes]; // all blocks of this size that were recycled
            void* ptr = nullptr;

            if (!blocks.empty()) {
                ptr = blocks.back(); // take the last one
                blocks.pop_back();
            }
            else {
                ptr = _aligned_malloc(aligned_bytes, 32);
                // allocates starting at adress that is multiple of 32. 
                // We allocate slightly more (aligned_bytes so rounded up to upper 32 multiple) so that SIMD can read chunks of 8
                // to sum up: start at adress mul of 32 to make SIMD even start. Make memory multiple of 32 bc SIMD only reads chunks of 8. If it was 30, reads 6 + 2 past = segfualt
                if (ptr == nullptr) { // OOM
                    clear();
                    ptr = _aligned_malloc(aligned_bytes, 32);
                    if (ptr == nullptr) { // factual OOM (claring cache didnt fix)
                        throw std::bad_alloc();
                    }
                }
            }

            m_current_usage += aligned_bytes;
            if (m_current_usage > m_hwm) {
                m_hwm = m_current_usage;
            }

            return ptr;
        }

        void free(void* ptr, int64_t aligned_bytes) {
            if (ptr != nullptr) {
                m_free_blocks[aligned_bytes].push_back(ptr);
                m_current_usage -= aligned_bytes;
            }
        }

        void clear() {
            for (auto& [size, blocks] : m_free_blocks) {
                for (void* ptr : blocks) {
                    _aligned_free(ptr);
                }
            }
            m_free_blocks.clear();
        }

};