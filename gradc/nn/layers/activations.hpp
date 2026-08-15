#pragma once
#include "../base/module.hpp"

namespace gradc {
    template <typename T>
    class ReLU : public Module<T> {
        public:
            ReLU() = default;

            Tensor<T> forward(Tensor<T> x) {
                return x.relu();
            }
    };
}