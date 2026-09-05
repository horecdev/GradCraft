#include <iostream>
#include <unordered_map>

#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <fstream>

#include <omp.h>

namespace gradc {

    struct PairHash {
        size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
            return (static_cast<size_t>(p.first) << 32) | p.second;
        }
    };

    struct WordSeq {
            std::vector<uint32_t> seq;
            uint64_t count;
    };

    struct PreTokenizer {
        public:
            static void process_normal_text(std::string_view text, std::vector<std::string_view>& pieces) {
                int64_t start = 0;
                int64_t stop = 0;
                int64_t length = text.length();

                while (stop < length) {
                    unsigned char c = static_cast<unsigned char>(text[stop]);

                    // you have to blast bitchass static_cast because isalpha etc expects unsigned char, but chars are signed by default
                    if (std::isalpha(c)) {
                        while (stop < length && std::isalpha(static_cast<unsigned char>(text[stop]))) { // checking for stop so we dont read text out of bounds
                            stop++;
                        }
                    }
                    else if (std::isdigit(c)) {
                        while (stop < length && std::isdigit(static_cast<unsigned char>(text[stop]))) {
                            stop++;
                        }
                    }
                    else if (std::ispunct(c)) {
                        while (stop < length && std::ispunct(static_cast<unsigned char>(text[stop]))) {
                            stop++;
                        }
                    }
                    else if (std::isspace(c) && length - stop > 1 && std::isalpha(static_cast<unsigned char>(text[stop + 1]))) {
                        stop++; // add the space
                        while (stop < length && std::isalpha(static_cast<unsigned char>(text[stop]))) { // add remaining chars
                            stop++;
                        }
                    }
                    else if (std::isspace(c)) {
                        while (stop < length && (std::isspace(static_cast<unsigned char>(text[stop])) || std::isalpha(static_cast<unsigned char>(text[stop])))) {
                            stop++;
                        }
                    }
                    else {
                        stop++;
                    }

                    pieces.push_back(text.substr(start, stop - start));

                    start = stop;
                }
            }

            static std::vector<std::string_view> process_chunks(std::string_view full_text) {
                std::vector<std::string_view> pieces;
                std::vector<std::string_view> specials = {"<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"};
                int64_t current = 0;
                while (current < full_text.length()) {
                    int64_t earliest_pos = std::string_view::npos;
                    std::string_view next_special = "";

                    for (std::string_view sp : specials) {
                        int64_t pos = full_text.find(sp, current);
                        if (pos < earliest_pos) {
                            earliest_pos = pos;
                            next_special = sp;
                        }
                    }

                    if (earliest_pos == std::string_view::npos) { // just process the rest of the text
                        process_normal_text(full_text.substr(current), pieces);
                    }

                    if (earliest_pos > current) { // process text up to the special token
                        process_normal_text(full_text.substr(current, earliest_pos - current), pieces);
                    }

                    pieces.push_back(next_special);

                    current = earliest_pos + next_special.length();
                }

                return pieces;
            }
    };

    class BytePairEncoding {
        private:
            std::vector<WordSeq> m_sequences;
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> m_merges;
            std::vector<std::string> m_vocab;
            int32_t m_num_tokens = 1024; // 2^15
        public:
            BytePairEncoding() {
                m_vocab.resize(m_num_tokens);
                for (int i = 0; i < 256; ++i) {
                    m_vocab[i] = std::string(1, static_cast<char>(i));
                }
                m_vocab[256] = "<|endoftext|>";
                m_vocab[257] = "<|pad|>";
                m_vocab[258] = "<|im_start|>";
                m_vocab[259] = "<|im_end|>";
            }

            void prepare_sequences(const std::vector<std::string_view>& pieces) {
                std::unordered_map<std::string_view, uint64_t> word_counts;
                for (std::string_view str : pieces) {
                    word_counts[str]++;
                }

                m_sequences.reserve(word_counts.size());

                for (const auto& [str, count] : word_counts) { 
                    // iterate over the UNIQUE ELEMS in hash map. This way every unique seq only appears ONCE
                    WordSeq ws;
                    ws.count = count; // how many times that str appeared

                    if (str == "<|endoftext|>") {ws.seq.push_back(256);}
                    else if (str == "<|pad|>") {ws.seq.push_back(257);}
                    else if (str == "<|im_start|>") {ws.seq.push_back(258);}
                    else if (str == "<|im_end|>") {ws.seq.push_back(259);}
                    else {
                        ws.seq.reserve(str.length());
                        for (char c : str) {
                            ws.seq.push_back(static_cast<uint32_t>(static_cast<unsigned char>(c)));
                        }
                    }
                    m_sequences.push_back(ws);
                }
        }

            void create_vocabulary(const std::vector<std::string_view>& pieces) { // thats the pieces PreTokenizer modifies
                std::cout << "Preparing sequences." << std::endl;
                prepare_sequences(pieces);

                std::cout << "Starting the token loop" << std::endl;

                int num_threads = omp_get_max_threads();

                for (int32_t token_id = 260; token_id < m_num_tokens; ++token_id) {

                    std::vector<std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, PairHash>> local_counts(num_threads);
                    std::unordered_map<std::pair<uint32_t, uint32_t>, uint64_t, PairHash> global_counts;

                    #pragma omp parallel
                    {
                        int tid = omp_get_thread_num();
                        auto& my_counts = local_counts[tid];

                        #pragma omp for schedule(guided) // all threads walk over the m_sequences and accumulate into their std::unordered_map
                        for (int64_t i = 0; i < std::ssize(m_sequences); ++i) {
                            const WordSeq& ws = m_sequences[i];
                            for (int64_t idx = 0; idx < std::ssize(ws.seq) - 1; ++idx) {
                                std::pair<uint32_t, uint32_t> pair = {ws.seq[idx], ws.seq[idx + 1]};
                                my_counts[pair] += ws.count;
                            }
                        }
                    }

                    for (const auto& lc : local_counts) { // accumulate into the master count for each pair
                        for (const auto& [pair, count] : lc) {
                            global_counts[pair] += count;
                        }
                    }

                    std::cout << "Global size: " + std::to_string(global_counts.size()) << std::endl;

                    int64_t max_freq = 0;
                    std::pair<uint32_t, uint32_t> max_pair;
                    for (const auto& [pair, freq] : global_counts) {
                        if (freq > max_freq) {
                            max_freq = freq;
                            max_pair = pair;
                        }
                    }

                    if (max_freq <= 1) {
                        throw std::runtime_error("Unable to create a vocab of 32768.");
                    }

                    // now we have to incorporate the pair into every sequence and put it into 

                    m_merges[max_pair] = token_id;
                    m_vocab[token_id] = m_vocab[max_pair.first] + m_vocab[max_pair.second];
                    std::cout << "Created token: " + std::to_string(token_id) + " - " + m_vocab[token_id] << std::endl;

                    #pragma omp parallel for schedule(guided) // each thread reaches for a vector and overwrites it replacing pair with new token
                        for (int64_t i = 0; i < std::ssize(m_sequences); ++i) {
                        WordSeq& ws = m_sequences[i];
                        int64_t read_idx = 0;
                        int64_t write_idx = 0;

                        while (read_idx < std::ssize(ws.seq)) {
                            if (read_idx < std::ssize(ws.seq) - 1 && ws.seq[read_idx] == max_pair.first && ws.seq[read_idx + 1] == max_pair.second) {
                                ws.seq[write_idx] = token_id;
                                read_idx += 2;
                                write_idx += 1;
                            }
                            else {
                                ws.seq[write_idx] = ws.seq[read_idx];
                                read_idx++;
                                write_idx++;
                            }
                        }
                        ws.seq.resize(write_idx);
                    }
                }
            }

            std::vector<uint32_t> encode(std::string_view text) {
                std::vector<std::string_view> pieces = PreTokenizer::process_chunks(text);

                std::vector<uint32_t> final_tokens;
                final_tokens.reserve(text.length() / 3);

                for (const std::string_view& str : pieces) {
                    std::vector<uint32_t> seq;

                    if (str == "<|endoftext|>") {seq.push_back(256);}
                    else if (str == "<|pad|>") {seq.push_back(257);}
                    else if (str == "<|im_start|>") {seq.push_back(258);}
                    else if (str == "<|im_end|>") {seq.push_back(259);}
                    else {
                        seq.reserve(str.length());
                        for (char c : str) {
                            seq.push_back(static_cast<uint32_t>(static_cast<unsigned char>(c)));
                        }
                    }

                    while (std::ssize(seq) >= 2) {
                        uint32_t best_token_id = 0xFFFFFFFF; // set the best token to be the lowest priority token. Then you walk and find higher priorities
                        std::pair<uint32_t, uint32_t> best_pair;
                        bool found_merge = false;

                        for (int64_t i = 0; i < std::ssize(seq) - 1; ++i) {
                            auto it = m_merges.find({seq[i], seq[i + 1]});
                            if (it != m_merges.end() && it->second < best_token_id) {
                                best_token_id = it->second;
                                best_pair = it->first;
                                found_merge = true;
                            }
                        }

                        if (!found_merge) {
                            break; // cant compress further
                        }

                        int64_t read_idx = 0;
                        int64_t write_idx = 0;
                        while (read_idx < std::ssize(seq)) {
                            if (read_idx < std::ssize(seq) - 1 && seq[read_idx] == best_pair.first && seq[read_idx + 1] == best_pair.second) {
                                seq[write_idx] = best_token_id;
                                read_idx += 2;
                                write_idx++;
                            }  
                            else {
                                seq[write_idx] = seq[read_idx];
                                read_idx++;
                                write_idx++;
                            }
                        }
                        seq.resize(write_idx);
                    }
                    final_tokens.insert(final_tokens.end(), seq.begin(), seq.end()); // append all seq elems
                }

                return final_tokens;
            }

            std::string decode(std::vector<uint32_t> indices) {
                std::string result = "";
                for (uint32_t idx : indices) {
                    result = result + m_vocab[idx];
                }
                return result;
            }

            void save_vocab(std::string path) {
                std::ofstream out(path, std::ios::binary);
                if (!out) {throw std::runtime_error("Failed to open file for saving: " + path);}

                uint32_t num_merges = m_merges.size();
                out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));
                for (const auto& [pair, token_id] : m_merges) {
                    out.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
                    out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
                    out.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
                }

                uint32_t num_vocab = m_vocab.size();
                out.write(reinterpret_cast<const char*>(&num_vocab), sizeof(num_vocab));

                for (const std::string& str : m_vocab) {
                    uint32_t len = str.length();
                    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
                    out.write(str.data(), len);
                }

            }

            void load_vocab(std::string path) {
                std::ifstream in(path, std::ios::binary);
                if (!in) {throw std::runtime_error("Failed to open file for loading");}

                m_merges.clear();
                m_vocab.clear();

                uint32_t num_merges;
                in.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));

                for (uint32_t i = 0; i < num_merges; ++i) {
                    uint32_t first, second, token_id;
                    in.read(reinterpret_cast<char*>(&first), sizeof(first));
                    in.read(reinterpret_cast<char*>(&second), sizeof(second));
                    in.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));

                    m_merges[{first, second}] = token_id;
                }

                uint32_t num_vocab;
                in.read(reinterpret_cast<char*>(&num_vocab), sizeof(num_vocab));
                m_vocab.resize(num_vocab);

                for (int32_t i = 0; i < num_vocab; ++i) {
                    uint32_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));

                    m_vocab[i].resize(len); // gotta preallocate string size
                    in.read(m_vocab[i].data(), len);
                }
            }
    };

    struct TokenManager {
        static void create_vocab_out_of_files(std::string vocab_save_path, std::vector<std::string> paths) {
            int64_t total_bytes = 0;
            const int64_t MAX_BYTES_PER_FILE = 50 * 1024 * 1024;

            for (const std::string& path : paths) {
                if (std::filesystem::exists(path)) {
                    int64_t file_bytes = std::filesystem::file_size(path);
                    total_bytes += std::min(file_bytes, MAX_BYTES_PER_FILE);
                }
            }


            std::vector<char> raw_data(total_bytes);

            int64_t write_offset = 0;
            for (const std::string& path : paths) {
                if (std::filesystem::exists(path)) {
                    int64_t file_bytes = std::filesystem::file_size(path);
                    int64_t bytes_to_read = std::min(file_bytes, MAX_BYTES_PER_FILE);
                    std::ifstream file(path, std::ios::binary);

                    if (file) {
                        file.read(raw_data.data() + write_offset, bytes_to_read);
                        write_offset += bytes_to_read;
                    }
                }
            }

            std::string_view full_text(raw_data.data(), raw_data.size());

            std::cout << "Processing chunks." << std::endl;
            std::vector<std::string_view> pieces = PreTokenizer::process_chunks(full_text);

            BytePairEncoding bpe;
            bpe.create_vocabulary(pieces);
            bpe.save_vocab(vocab_save_path);
        }

        static void encode_dataset(std::string output_path, std::string vocab_load_path, std::vector<std::string> paths) {
            BytePairEncoding bpe;
            bpe.load_vocab(vocab_load_path);

            std::ofstream out(output_path, std::ios::binary | std::ios::app); // can only ever append
            if (!out) {throw std::runtime_error("Failed to open output bin file: " + output_path);}

            for (const std::string& path : paths) { // process one file at a time
                if (!std::filesystem::exists(path)) {continue;}

                int64_t file_bytes = std::filesystem::file_size(path);
                std::vector<char> buffer(file_bytes);

                std::ifstream in(path, std::ios::binary);
                if (!in) {continue;}
                in.read(buffer.data(), file_bytes);

                std::string_view text(buffer.data(), file_bytes);
                std::vector<uint32_t> tokens = bpe.encode(text);

                int64_t bytes_to_write = tokens.size() * sizeof(uint32_t);
                out.write(reinterpret_cast<const char*>(tokens.data()), bytes_to_write);
            }
        }
    };
}
