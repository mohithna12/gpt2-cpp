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

// Causal multi-head self-attention. qkv is [T, 3C] (q | k | v, each split into n_head
// heads of C / n_head columns); out is [T, C] with heads concatenated.
void attention(float* out, const float* qkv, int64_t T, int64_t C, int64_t n_head);

}  // namespace gpt2::ops
