#pragma once
#include "../base/module.hpp"
#include "./linear.hpp"
#include "../base/initializers.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class MultiHeadAttention : public Module<T> {
        private:
            Linear<T> m_q_proj;
            Linear<T> m_k_proj;
            Linear<T> m_v_proj;
            Linear<T> m_out_proj;
            int64_t m_num_heads;
            int64_t m_head_dim;
            Tensor<T> m_causal_mask; // register this as??? how tf do you move 
            
        public:
            MultiHeadAttention(int64_t embed_dim, int64_t num_heads, bool is_causal, int64_t max_seq_len, const Initializer<T>& proj_w_init, const Initializer<T>& proj_b_init)
             : m_q_proj(Linear<T>(embed_dim, embed_dim, proj_w_init, proj_b_init)), m_k_proj(Linear<T>(embed_dim, embed_dim, proj_w_init, proj_b_init)), 
                m_v_proj(Linear<T>(embed_dim, embed_dim, proj_w_init, proj_b_init)), m_out_proj(Linear<T>(embed_dim, embed_dim, proj_w_init, proj_b_init)) 
            {
                if (embed_dim % num_heads != 0) {
                    throw std::runtime_error("embed_dim modulo num_heads must = 0 in MHA.");
                }

                m_num_heads = num_heads;
                m_head_dim = embed_dim / num_heads;

                this->register_module("q_proj", &m_q_proj);
                this->register_module("k_proj", &m_k_proj);
                this->register_module("v_proj", &m_v_proj);
                this->register_module("out_proj", &m_out_proj);

                if (is_causal) {
                    m_causal_mask = Tensor<T>::upper_triangular(max_seq_len, std::numeric_limits<T>::lowest());
                }
            }

            Tensor<T> forward(Tensor<T> x) { // x is (B, T, C)
                int64_t B = x.shape()[0];
                int64_t seq_len = x.shape()[1];

                Tensor<T> Q = m_q_proj.forward(x); // (B, T, C) @ (C, C)
                Tensor<T> K = m_k_proj.forward(x);
                Tensor<T> V = m_v_proj.forward(x);

                Q = Q.reshape({B, seq_len, m_num_heads, m_head_dim}).permute({0, 2, 1, 3});
                K = K.reshape({B, seq_len, m_num_heads, m_head_dim}).permute({0, 2, 1, 3});
                V = V.reshape({B, seq_len, m_num_heads, m_head_dim}).permute({0, 2, 1, 3});

                if (m_causal_mask.device() != x.device()) {
                    m_causal_mask = m_causal_mask.to(x.device());
                    m_causal_mask.realize();
                    m_causal_mask.make_leaf();
                }

                Tensor<T> active_mask = m_causal_mask[Slice(0, seq_len), Slice(0, seq_len)];

                Tensor<T> attn = sdpa_naive(Q, K, V); // [B, num_heads, T, head_dim]
                attn = attn.permute({0, 2, 1, 3}).reshape({B, seq_len, m_num_heads * m_head_dim}); // [B, T, C]
                return m_out_proj.forward(attn);
            }
    };

    
}