#pragma 

#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <fstream>

namespace gradc {

    class PreTokenizer {
        private:
            std::vector<std::string> m_filepaths;
            std::vector<char> m_raw_data;
            std::vector<std::string_view> m_pieces;
        public:
            PreTokenizer(std::vector<std::string> filepaths) : m_filepaths(std::move(filepaths)) {}

            const std::vector<std::string_view>& get_pieces() {
                return m_pieces;
            }

            void process() {
                uint64_t total_bytes = 0;
                for (const std::string& path : m_filepaths) {
                    if (std::filesystem::exists(path)) {
                        uint64_t file_bytes = std::filesystem::file_size(path);
                        total_bytes += file_bytes;
                    }
                }

                m_raw_data.resize(total_bytes);
                m_pieces.reserve(total_bytes / 4); // hopefully one piece will merge >= 4 chars on average

                int64_t write_offset = 0;
                for (const std::string& path : m_filepaths) {
                    if (std::filesystem::exists(path)) {
                        uint64_t file_bytes = std::filesystem::file_size(path);

                        std::ifstream file(path, std::ios::binary);

                        if (file.is_open()) {
                            file.read(reinterpret_cast<char*>(m_raw_data.data() + write_offset), file_bytes);
                            
                            write_offset += file_bytes;
                        }
                    }
                }

                std::string_view text(m_raw_data.data(), m_raw_data.size());

                int64_t start = 0;
                int64_t stop = 0;
                int64_t length = text.length();

                while (stop < length) {
                    char c = static_cast<unsigned char>(m_raw_data[stop]);

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

                    m_pieces.push_back(text.substr(start, stop - start));

                    start = stop;
                }
            }

            
    };

    class BytePairEncoding {
        private:
            std::vector<std::vector<uint32_t>> m_sequences;
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> merges;
        public:
            void prepare_sequences(const std::vector<std::string_view>& pieces) {
                m_sequences.resize(pieces.size());

                for (int64_t i = 0; i < pieces.size(); ++i) {
                    const std::string_view& view = pieces[i];
                    std::vector<uint32_t> seq = m_sequences[i];

                    for (char c : view) {
                        seq.push_back(static_cast<uint32_t>(c));
                    }
                }
            }

            void merge() {
                // first: loop through every sequence. Create a full map. 
            }


    };

}
