#pragma once

#include <vector>

namespace gradc {

    struct TensorStateBase;

    template <typename T>
    class Tensor;

    template <typename T>
    class Node {

        public:
            virtual Tensor<T> realize() = 0;

            virtual void backward(const Tensor<T>& out_grad, bool retain_graph) = 0;

            virtual std::vector<TensorStateBase*> get_input_states() = 0;

            virtual ~Node() {}
    };
}