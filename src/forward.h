#pragma once
// GPT-2 forward pass with a KV cache: each call processes only new tokens, attending to the
// keys and values the cache holds for earlier positions.

#include <cstdint>
#include <vector>

#include "model.h"

namespace gpt2 {

// Keys and values for every position processed so far, per layer: [n_layer, n_ctx, n_embd]
// each (~75 MB for GPT-2 small).
struct KVCache {
    explicit KVCache(const Config& c);

    int64_t pos = 0;  // positions stored; the next token goes here
    void clear() { pos = 0; }

    int64_t n_layer, n_ctx, n_embd;
    std::vector<float> k, v;
    float* k_layer(int64_t l) { return k.data() + l * n_ctx * n_embd; }
    float* v_layer(int64_t l) { return v.data() + l * n_ctx * n_embd; }
};

enum class Logits {
    All,   // [tokens.size(), vocab_size]
    Last,  // [1, vocab_size]: only the last new position (skips most of the LM head)
};

// Runs `tokens` at positions cache.pos... and appends their keys/values to the cache.
// Throws std::invalid_argument if tokens is empty, has a bad id, would overflow n_ctx, or the
// cache was made for a different config. On a throw the cache is unchanged.
std::vector<float> forward(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                           Logits which = Logits::All);

// Whole sequence from scratch (fresh cache), logits for every position.
std::vector<float> forward(const Model& m, const std::vector<int>& tokens);

}  // namespace gpt2
