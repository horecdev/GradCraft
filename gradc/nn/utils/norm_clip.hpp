#pragma once

#include "../base/parameter.hpp"
#include "../../backend/dispatcher.hpp"

namespace gradc {

    template <typename T>
    class GlobalNormClipper {
        private:
            T m_max_norm;
        public:
            GlobalNormClipper(T max_norm = static_cast<T>(1.0)) : m_max_norm(max_norm) {}

            T normalize(const std::vector<Parameter<T>*>& params) {
                if (params.empty()) {return;}

                Device device = params[0]->device();

                Tensor<T> total_sq = Tensor<T>(std::vector<int64_t>{}, T(0), device);

                for (Parameter<T>* p : params) {
                    if (!p->grad().has_value()) continue;
                    Tensor<T> grad = p->grad().value();
                    
                    Tensor<T> sq = Tensor<T>(grad.shape(), device, uninitialized);
                    dispatch(device, UnaryOp::Square, sq, grad);
                    
                    std::vector<int64_t> all_axes;
                    all_axes.reserve(std::ssize(grad.shape()));
                    for (int64_t i = 0; i < std::ssize(grad.shape()); ++i) {
                        all_axes.push_back(i);
                    }
                    
                    RedMeta red_meta = infer_red_meta(grad.shape(), all_axes, false);
                    Tensor<T> sum_sq = Tensor<T>(red_meta.result_shape, device, uninitialized);
                    dispatch(device, ReduceOp::Sum, red_meta, sum_sq, sq);

                    dispatch(device, BinaryOpInPlace::Add, total_sq, sum_sq);
                }
                
                T global_sq = total_sq.item();
                T global_norm = std::sqrt(global_sq);
                
                if (global_norm > m_max_norm) {
                    T scale_factor = m_max_norm / (global_norm + static_cast<T>(1e-6));
                    Tensor<T> scale_tensor = Tensor<T>(std::vector<int64_t>{}, scale_factor, device);

                    for (Parameter<T>* p : params) {
                        if (!p->grad().has_value()) continue;
                        Tensor<T> grad = p->grad().value();
                        
                        dispatch(device, BinaryOpInPlace::Mul, grad, scale_tensor);
                    }
                }

                return global_norm; // for logging
            }
    };
}