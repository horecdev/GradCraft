#pragma once

#include <cstdint>
#include <random>

namespace gradc {
    class CPUUtils {
        public:
        template <typename T>
        static void fill_arange(T* ptr, T start, T step, int64_t size) {
            for (int64_t i = 0; i < size; ++i) {
                ptr[i] = start + step * static_cast<T>(i);
            }
        }

        template <typename T>
        static void upper_triangular(T* ptr, int64_t size, T fill_val) {
            for (int64_t i = 0; i < size; ++i) {
                for (int64_t j = i + 1; j < size; ++j) {
                    ptr[i * size + j] = fill_val;
                }
            }
        }

        template <typename T>
        static void fill_normal(T* ptr, T mean, T std, int64_t size) {
            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());

            std::normal_distribution<T> dist(mean, std);

            for (int64_t i = 0; i < size; ++i) {
                ptr[i] = dist(gen);
            }
        }

        template <typename T>
        static void fill_uniform(T* ptr, T low, T high, int64_t size) {
            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());

            std::uniform_real_distribution<T> dist(low, high);

            for (int64_t i = 0; i < size; ++i) {
                ptr[i] = dist(gen);
            }
        }

        template <typename T>
        // res is 0 initialized
        // res is [B, T, distrib_dim], indices are [B, T], batch_size is [B * T]
        // res and indices must be DENSE.
        static void one_hot_encode(T* res_ptr, int64_t* indices_ptr, int64_t batch_size, int64_t distrib_dim) { // batch_size is B*T
            for (int64_t row = 0; row < batch_size; ++row) {
                int64_t one_idx = indices_ptr[row];
                int64_t batch_offset = row * distrib_dim;
                res_ptr[batch_offset + one_idx] = 1;
            }
        }
    };
}
