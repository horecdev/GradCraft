#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

namespace gradc {

    template <typename T>
    class MSELossNode : public Node<T> {
        
    };
    
    template <typename T>
    class SoftmaxCrossEntropyLossNode : public Node<T> {
        
    };
}