#include "gradc/gradc.hpp" // IWYU pragma: keep

using namespace gradc;

int main() {
    // GPT params
    int64_t vocab_size = 4096;
    int64_t embed_dim = 256;
    int64_t max_seq_len = 1024;
    int64_t num_heads = 8;
    int64_t num_layers = 4;
    float eps = 1e-5;

    // this init is used everywhere
    NormalInit<float> init = NormalInit<float>(0.0, 0.02);

    GPT<float> model = GPT<float>(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, init, eps);
}