#pragma once

#include "../layers/gpt.hpp"
#include "../utils/tokenizer.hpp"
#include <random>

namespace gradc {

    template <typename T>
    struct GPTGenerator {
        public:
            static std::vector<std::string> run_inference(GPT<T>& gpt, BytePairEncoding& bpe, int64_t context_length, const std::string& start_text, int64_t max_tokens, int64_t num_sequences, Device cpu, Device infer_device, std::optional<float> temperature) {
                std::vector<uint32_t> tokens = bpe.encode(start_text);
                int64_t current_length = std::ssize(tokens);

                if (std::ssize(tokens) >= max_tokens) {
                    throw std::runtime_error("Tokenized start_text is >= max_tokens in GPT inference.");
                }

                std::mt19937 rng;
                if (temperature.has_value()) {
                    rng = std::mt19937(67); // 67 joke
                }
                
                Tensor<int64_t> tensor_tokens = Tensor<int64_t>({num_sequences, max_tokens}, cpu, uninitialized);
                int64_t* tok_ptr = tensor_tokens._get_storage()->data();

                for (int64_t b = 0; b < num_sequences; ++b) {
                    for (int64_t t = 0; t < current_length; ++t) {
                        tok_ptr[b * max_tokens + t] = static_cast<int64_t>(tokens[t]);
                    }
                }
                // after that tensor is pre-filled for each B from 0 to std::ssize(tokens)
                // now gotta slice 1024 tokens MAX up to std::ssize(tensor_tokens) master buffer

                Tensor<int64_t> moved_tokens = tensor_tokens.to(infer_device);
                int64_t* moved_tokens_ptr = moved_tokens._get_storage()->data();

                for (int64_t step = current_length; step < max_tokens; ++step) {
                    Tensor<int64_t> context = moved_tokens[_, Slice(std::max(0LL, step - context_length), step)];

                    Tensor<T> logits = gpt.forward(context);
                    Tensor<T> last_logits = logits[_, -1, _];

                    if (!temperature.has_value()) { // argmax
                        Tensor<int64_t> next_tokens = last_logits.argmax(-1, false);
                        next_tokens.realize();
                        int64_t* next_tokens_ptr = next_tokens._get_storage()->data();

                        if (infer_device.is_cpu()) {
                            for (int64_t b = 0; b < num_sequences; ++b) {
                                moved_tokens_ptr[b * max_tokens + step] = next_tokens_ptr[b];
                            }
                        }
                        else if (infer_device.is_cuda()) {
                            cudaMemcpy2D(
                                moved_tokens_ptr + step, // destination pointer (offset to first element - "upper left corner")
                                max_tokens * sizeof(int64_t), // bytes between rows
                                next_tokens_ptr, // source ptr (upper left corner)
                                sizeof(int64_t), // width of source row
                                sizeof(int64_t), // width to copy per row
                                num_sequences, // how many rows to copy (height)
                                cudaMemcpyDeviceToDevice
                            );
                        }
                    }
                    else { // do softmax to get probs that sum to 1.0, move to CPU to run sampling, then copy memory back
                        Tensor<T> probs = (last_logits / temperature.value()).softmax(-1); 
                        Tensor<T> cpu_probs = probs.to(cpu);
                        cpu_probs.realize();

                        T* probs_ptr = cpu_probs._get_storage()->data();
                        int64_t vocab_size = cpu_probs.shape().back();

                        for (int64_t b = 0; b < num_sequences; ++b) {
                            std::discrete_distribution<int64_t> dist(probs_ptr + (b * vocab_size), probs_ptr + ((b + 1) * vocab_size));

                            int64_t next_token_id = dist(rng);

                            if (infer_device.is_cpu()) {
                                moved_tokens_ptr[b * max_tokens + step] = next_token_id;
                            } 
                            else {
                                cudaMemcpy(moved_tokens_ptr + (b * max_tokens + step), 
                                &next_token_id, 
                                sizeof(int64_t), 
                                cudaMemcpyHostToDevice);
                            }
                        }
                    }
                }

                Tensor<int64_t> final_cpu_tokens = moved_tokens.to(cpu);
                final_cpu_tokens.realize();
                int64_t* final_ptr = final_cpu_tokens._get_storage()->data();

                std::vector<std::string> decoded_sequences;
                decoded_sequences.reserve(num_sequences);

                for (int64_t b = 0; b < num_sequences; ++b) {
                    std::vector<uint32_t> current_seq;
                    current_seq.reserve(max_tokens);

                    for (int64_t t = 0; t < max_tokens; ++t) {
                        int64_t token_id = final_ptr[b * max_tokens + t];
                        
                        if (token_id == 259) { // <|im_end|>
                            break;
                        }
                        
                        current_seq.push_back(static_cast<uint32_t>(token_id));
                    }
                    decoded_sequences.push_back(bpe.decode(current_seq));
                }

                return decoded_sequences;
            }
        };     
}
