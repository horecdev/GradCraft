#pragma once

#include "gradc/nn/base/parameter.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Optimizer {
        protected:
            std::vector<Parameter<T>*> m_params;
        public:
            Optimizer() = default;
            virtual ~Optimizer() = default;
    };
}