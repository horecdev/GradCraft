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

