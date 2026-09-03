#pragma once

#include "gradc/core/types.hpp"

namespace gradc {
    class CUDAUtils {
        public:
        template <typename T>
            static void set_scalar(T* d_ptr, T value);
            template <typename T>
            static void fill(T* ptr, T val, int64_t size, Device device);
            template <typename T>
            static void fill_arange(T* ptr, T start, T step, int64_t size, Device device);
            template <typename T>
            static void upper_triangular(T* ptr, int64_t size, T fill_val, Device device);
            template <typename T>
            static void fill_normal(T* ptr, T mean, T std, int64_t size, Device device);
            template <typename T>
            static void fill_uniform(T* ptr, T low, T high, int64_t size, Device device);
    };
}
