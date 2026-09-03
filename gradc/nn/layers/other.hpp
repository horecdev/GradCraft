#pragma once
#include "../base/module.hpp"
#include "../base/initializers.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Embedding : public Module<T> {
        private:
            Parameter<T> m_embeds;
        public:
            // embed_dim is a vector because we support ndim embeddings
            Embedding(int64_t distrib_dim, std::vector<int64_t> embed_dim, const Initializer<T>& embed_init) {
                embed_dim.insert(embed_dim.begin(), distrib_dim);
                m_embeds = Parameter<T>(embed_init.generate(embed_dim, Device(DeviceType::CPU)));

                this->register_parameter("embeds", &m_embeds);
            }

            Tensor<T> forward(Tensor<T> indices) {
                Tensor<T> y = embed(indices, m_embeds);
                return y;
            }
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class PosEncoding : public Module<T> {
        private: 
            Parameter<T> m_pos_embeds;
            Tensor<int64_t> m_token_range;
        public:
            PosEncoding(int64_t max_seq_len, std::vector<int64_t> pos_embed_dim, const Initializer<T>& embed_init) {
                pos_embed_dim.insert(pos_embed_dim.begin(), max_seq_len);
                m_pos_embeds = embed_init.generate(pos_embed_dim, Device(DeviceType::CPU));

                m_token_range = Tensor<int64_t>::arange(0, max_seq_len, 1);

                this->register_parameter("pos_embeds", &m_pos_embeds);
            }

            Tensor<T> forward(int64_t seq_len) {
                if (m_token_range.device() != m_pos_embeds.device()) {
                    m_token_range = m_token_range.to(m_pos_embeds.device());
                    m_token_range.realize();
                    m_token_range.make_leaf();
                }
                
                return embed(m_token_range[Slice(0, seq_len)], m_pos_embeds);
            }
    };

    template <typename T>
    requires std::is_floating_point_v<T>
    class Dropout : public Module<T> {
        private:
            T m_p;
        public:
            Dropout(T p) {
                m_p = p;
            }

            Tensor<T> forward(Tensor<T> x) {
                if (this->is_training()) {
                    return x.dropout(m_p);
                }
                else {
                    return x;
                }
            }
    };
}

