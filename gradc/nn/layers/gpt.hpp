#pragma once

#include "../base/module.hpp"
#include "./linear.hpp"
#include "./normalization.hpp"
#include "./attention.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class SwiGLUMLP: Module<T> {
        private:
            Linear<T> m_w1;
            Linear<T> m_w2;
            Linear<T> m_w3;
        public:
            SwiGLUMLP(int64_t embed_dim, int64_t hidden_dim, const Initializer<T>& init)
             : m_w1(embed_dim, hidden_dim, init, ZerosInit<T>()), m_w2(embed_dim, hidden_dim, init, ZerosInit<T>()), m_w3(hidden_dim, embed_dim, init, ZerosInit<T>()) {
                this->register_module("w1", &m_w1);
                this->register_module("w2", &m_w2);
                this->register_module("w3", &m_w3);
            }

            Tensor<T> forward(Tensor<T> x) {
                return m_w3.forward(m_w1.forward(x).silu() * m_w2.forward(x)); // GLU but projected once again
            }
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class TransformerBlock: Module<T> {
        private:
            RMSNorm<T> m_norm1;
            MultiHeadAttention<T> m_attn;
            RMSNorm<T> m_norm2;
            SwiGLUMLP<T> m_mlp; // feed forward

        public:
            TransformerBlock(int64_t embed_dim, int64_t num_heads, int64_t max_seq_len, const Initializer<T>& init, T eps = static_cast<T>(1e-5))
             : m_norm1({embed_dim}, {-1}, OnesInit<T>(), eps), m_attn(embed_dim, num_heads, true, max_seq_len, init, ZerosInit<T>()),
               m_norm2({embed_dim}, {-1}, OnesInit<T>(), eps), m_mlp(embed_dim, embed_dim * 3, init) {

                this->register_module("norm1", &m_norm1);
                this->register_module("attn", &m_attn);
                this->register_module("norm2", &m_norm2);
                this->register_module("mlp", &m_mlp);
            }

            Tensor<T> forward(Tensor<T> x) {
                x = x + m_attn.forward(m_norm1.forward(x));
                x = x + m_mlp.forward(m_norm2.forward(x));
                return x;
            }
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class GPT : Module<T> {

    };
}
