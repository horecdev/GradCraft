#pragma once

#include "gradc/core/types.hpp"

namespace gradc {
    class CUDAUtils {
        public:
            template <typename T>
            static void fill(T* ptr, T val, int64_t size, Device device);
    };
}
