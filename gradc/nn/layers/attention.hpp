#pragma once
#include "../base/module.hpp"
#include "../base/initializers.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Linear : public Module<T> {
        private:
            Linear<T> m_q_proj;
            Linear<T> m_k_proj;
            Linear<T> m_v_proj;
            Linear<T> m_out_proj;
            int64_t m_num_heads;
            int64_t m_head_dim;
            Tensor<T> m_causal_mask;
        public:
    };

    
}