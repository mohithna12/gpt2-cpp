#pragma once
// GPT-2 forward pass. Phase 2: the whole sequence is recomputed on every call (no KV cache).

#include <vector>

#include "model.h"

namespace gpt2 {

// Logits for every position, [tokens.size(), vocab_size] row-major.
// Throws std::invalid_argument if tokens is empty, longer than n_ctx, or has a bad id.
std::vector<float> forward(const Model& m, const std::vector<int>& tokens);

}  // namespace gpt2
