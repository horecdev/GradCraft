#pragma once

#include <random>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include"../../core/tensor.hpp"

namespace gradc {
    class DataLoader {
        private:
            std::ifstream m_in;
            int64_t m_total_tokens;
            std::vector<uint32_t> m_buffer;
            int64_t m_cursor = 0;

            std::mt19937_64 m_rng;

            const int64_t CHUNK_TOKENS = 4 * 1024 * 1024; // we will fetch 16MB of tokens at a time randomly

            void load_random_chunk() {
                std::uniform_int_distribution<int64_t> dist(0, m_total_tokens - CHUNK_TOKENS - 1);
                int64_t random_start_token = dist(m_rng);

                m_in.seekg(random_start_token * sizeof(uint32_t));

                m_in.read(reinterpret_cast<char*>(m_buffer.data()), CHUNK_TOKENS * sizeof(uint32_t));
                m_cursor = 0;
            }
        public:
            DataLoader(const std::string& path) : m_rng(42) {
                m_in = std::ifstream(path, std::ios::binary | std::ios::ate);
                if (!m_in) {throw std::runtime_error("Unable to open: " + path);}

                int64_t file_bytes = std::filesystem::file_size(path);
                m_total_tokens = file_bytes / sizeof(uint32_t);

                m_buffer.resize(CHUNK_TOKENS);

                load_random_chunk();
            }

            std::pair<Tensor<int64_t>, Tensor<int64_t>> next_batch(int64_t B, int64_t T, Device cpu_device) {
                int64_t tokens_required = B * T + 1;

                if (m_cursor + tokens_required > CHUNK_TOKENS) {
                    load_random_chunk();
                }

                Tensor<int64_t> X = Tensor<int64_t>({B, T}, cpu_device, uninitialized);
                Tensor<int64_t> Y = Tensor<int64_t>({B, T}, cpu_device, uninitialized);\

                int64_t* p_x = X._get_storage()->data();
                int64_t* p_y = Y._get_storage()->data();

                for (int64_t b = 0; b < B; ++b) {
                    for (int64_t t = 0; t < T; ++t) {
                        int64_t idx = m_cursor + (b * T + t);

                        p_x[(b * T) + t] = static_cast<int64_t>(m_buffer[idx]);
                        p_y[(b * T) + t] = static_cast<int64_t>(m_buffer[idx + 1]);
                    }
                }

                m_cursor += B * T;
                return {X, Y};
            }
    };
}
