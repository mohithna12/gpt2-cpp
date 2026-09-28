#pragma once
// GPT-2 config, weights, and checkpoint loading.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace gpt2 {

struct Config {
    int n_layer = 0;
    int n_head = 0;
    int n_embd = 0;
    int n_ctx = 0;
    int vocab_size = 0;
    int head_dim() const { return n_embd / n_head; }
};

// Linear weights use Hugging Face's Conv1D layout: W is [in, out], y = x @ W + b.
struct LayerWeights {
    Tensor ln_1_w, ln_1_b;
    Tensor attn_w, attn_b;        // c_attn: [C, 3C], [3C]
    Tensor attn_proj_w, attn_proj_b;  // c_proj: [C, C], [C]
    Tensor ln_2_w, ln_2_b;
    Tensor fc_w, fc_b;            // c_fc:   [C, 4C], [4C]
    Tensor fc_proj_w, fc_proj_b;  // c_proj: [4C, C], [C]
};

struct Weights {
    Tensor wte;  // [vocab, C]; also the (tied) LM head
    Tensor wpe;  // [n_ctx, C]
    std::vector<LayerWeights> layers;
    Tensor ln_f_w, ln_f_b;
};

class Model {
public:
    // Throws std::runtime_error with a readable message on any format problem.
    static Model load(const std::string& path);

    const Config& config() const { return config_; }
    const Weights& weights() const { return weights_; }
    // All tensors by their checkpoint name, in file order.
    const std::vector<std::pair<std::string, Tensor>>& tensors() const { return tensors_; }
    size_t num_params() const;

private:
    Config config_;
    Weights weights_;
    std::vector<std::pair<std::string, Tensor>> tensors_;
    // Whole file, 64-byte aligned; tensors point into it.
    std::shared_ptr<float> buffer_;
    size_t buffer_bytes_ = 0;
};

}  // namespace gpt2
