#pragma once

#include <random>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <stdexcept>

#include "../../core/tensor.hpp"

namespace gradc {
    class DataLoader {
        private:
            std::ifstream m_in;
            int64_t m_total_tokens = 0;
            int64_t m_seq_len = 0;
            int64_t m_total_sequences = 0;
            int64_t m_seq_cursor = 0;

            std::vector<int64_t> m_sequence_indices;
            std::mt19937_64 m_rng;

            void shuffle_dataset() {
                std::shuffle(m_sequence_indices.begin(), m_sequence_indices.end(), m_rng);
                m_seq_cursor = 0;
            }

        public:
            DataLoader(const std::string& path, int64_t seq_len, uint64_t seed = 67) 
                : m_seq_len(seq_len), m_rng(seed) 
            {
                m_in = std::ifstream(path, std::ios::binary | std::ios::ate);
                if (!m_in) {
                    throw std::runtime_error("Unable to open dataset: " + path);
                }

                int64_t file_bytes = std::filesystem::file_size(path);
                m_total_tokens = file_bytes / sizeof(uint32_t);

                if (m_total_tokens <= m_seq_len + 1) {
                    throw std::runtime_error("Dataset too small for sequence length " + std::to_string(seq_len));
                }

                m_total_sequences = (m_total_tokens - 1) / m_seq_len;

                m_sequence_indices.resize(m_total_sequences);
                for (int64_t i = 0; i < m_total_sequences; ++i) {
                    m_sequence_indices[i] = i;
                }

                shuffle_dataset();
            }

            std::pair<Tensor<int64_t>, Tensor<int64_t>> next_batch(int64_t B, int64_t T, Device cpu_device) {
                if (T != m_seq_len) {
                    throw std::invalid_argument("Requested T does not match initialized seq_len.");
                }

                if (m_seq_cursor + B > m_total_sequences) {
                    shuffle_dataset();
                }

                Tensor<int64_t> X = Tensor<int64_t>({B, T}, cpu_device, uninitialized);
                Tensor<int64_t> Y = Tensor<int64_t>({B, T}, cpu_device, uninitialized);

                int64_t* p_x = X._get_storage()->data();
                int64_t* p_y = Y._get_storage()->data();

                std::vector<uint32_t> raw_buf(T + 1);

                for (int64_t b = 0; b < B; ++b) {
                    int64_t seq_idx = m_sequence_indices[m_seq_cursor + b];
                    int64_t token_offset = seq_idx * T;

                    m_in.seekg(token_offset * sizeof(uint32_t));
                    m_in.read(reinterpret_cast<char*>(raw_buf.data()), (T + 1) * sizeof(uint32_t));

                    int64_t row_offset = b * T;
                    for (int64_t t = 0; t < T; ++t) {
                        p_x[row_offset + t] = static_cast<int64_t>(raw_buf[t]);
                        p_y[row_offset + t] = static_cast<int64_t>(raw_buf[t + 1]);
                    }
                }

                m_seq_cursor += B;
                return {X, Y};
            }

            int64_t total_sequences() const { return m_total_sequences; }
    };
}