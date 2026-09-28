#pragma once
// Transformer building blocks on raw float32 row-major buffers. Phase 2: plain loops, kept
// simple and obviously correct; later phases optimize these behind the same signatures.

#include <cstdint>

namespace gpt2::ops {

// out[t] = (x[t] - mean) / sqrt(var + eps) * w + b, per row of length C. out may alias x.
void layernorm(float* out, const float* x, const float* w, const float* b, int64_t T, int64_t C,
               float eps = 1e-5f);

// out[T, N] = x[T, K] @ W[K, N] + bias[N]. W uses HF's Conv1D [in, out] layout.
// bias may be null. out must not alias x.
void matmul(float* out, const float* x, const float* W, const float* bias, int64_t T, int64_t K,
            int64_t N);

// out[T, N] = x[T, K] @ W[N, K]^T: the tied LM head, where W is wte [vocab, C].
void matmul_transposed(float* out, const float* x, const float* W, int64_t T, int64_t K,
                       int64_t N);

// In place GELU, tanh approximation (HF "gelu_new"), over n elements.
void gelu(float* x, int64_t n);

// out[t] += x[t] over n elements.
void add_inplace(float* out, const float* x, int64_t n);

// Causal multi-head self-attention for T new queries at absolute positions pos0..pos0+T-1.
// Queries come from qkv [T, 3C] (the q third; heads are C / n_head column slices). Keys and
// values come from the cache, k/v [pos0 + T, C], which must already hold the new positions.
// out is [T, C] with heads concatenated.
void attention(float* out, const float* qkv, const float* k, const float* v, int64_t T,
               int64_t pos0, int64_t C, int64_t n_head);

}  // namespace gpt2::ops
