#pragma once

#include "../base/module.hpp"
#include "./linear.hpp"
#include "./other.hpp"
#include "./normalization.hpp"
#include "./attention.hpp"
#include <string>

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
        private:
            Embedding<T> m_token_embed;
            PosEncoding<T> m_pos_embed;
            std::vector<std::unique_ptr<TransformerBlock<T>>> m_blocks;
            RMSNorm<T> m_final_norm;
            Linear<T> m_lm_head;
        public:
            GPT(int64_t vocab_size, int64_t max_seq_len, int64_t embed_dim, int64_t num_heads, int64_t num_layers, const Initializer<T>& init, T eps = static_cast<T>(1e-5))
             : m_token_embed(vocab_size, {embed_dim}, NormalInit<T>(0, 0.02)), m_pos_embed(max_seq_len, {embed_dim}, init),
               m_final_norm({embed_dim}, {-1}, OnesInit<T>(), eps), m_lm_head(embed_dim, vocab_size, init, ZerosInit<T>()) {

                this->register_module("token_embed", &m_token_embed);
                this->register_module("pos_embed", &m_pos_embed);

                for (int64_t i = 0; i < num_layers; ++i) {
                    // allocate the transformer block on the heap. hold the POINTER in a vector. Register the pointer on the heap. 
                    // the adress of vector of pointers will change (reallocation) but the pointers it holds will not.
                    m_blocks.push_back(std::make_unique<TransformerBlock<T>>(embed_dim, num_heads, max_seq_len, init, eps)); 
                    this->register_module("block_" + std::to_string(i), m_blocks.back().get());
                }

                this->register_module("final_norm", &m_final_norm);
                this->register_module("lm_head", &m_lm_head);
            }

            Tensor<T> forward(Tensor<T> indices) { // (B, T)
                int64_t seq_len = indices.shape()[1];

                Tensor<T> tok_emb = m_token_embed.forward(indices);
                Tensor<T> pos_emb = m_pos_embed.forward(seq_len);

                Tensor<T> x = tok_emb + pos_emb; // (B, T, C)

                for (auto& block : m_blocks) {
                    x = block->forward(x);
                }

                x = m_final_norm(x);
                return m_lm_head.forward(x);
            }
    };
}
