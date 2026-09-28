#include "ops.h"

#include <cmath>
#include <vector>

namespace gpt2::ops {

void layernorm(float* out, const float* x, const float* w, const float* b, int64_t T, int64_t C,
               float eps) {
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * C;
        float* o = out + t * C;
        float mean = 0;
        for (int64_t i = 0; i < C; ++i) mean += xr[i];
        mean /= C;
        float var = 0;
        for (int64_t i = 0; i < C; ++i) var += (xr[i] - mean) * (xr[i] - mean);
        var /= C;  // biased, like torch
        const float rstd = 1.0f / std::sqrt(var + eps);
        for (int64_t i = 0; i < C; ++i) o[i] = (xr[i] - mean) * rstd * w[i] + b[i];
    }
}

void matmul(float* out, const float* x, const float* W, const float* bias, int64_t T, int64_t K,
            int64_t N) {
    // i-k-j order so the inner loop walks rows of W contiguously.
    for (int64_t t = 0; t < T; ++t) {
        float* o = out + t * N;
        for (int64_t j = 0; j < N; ++j) o[j] = bias ? bias[j] : 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            const float xk = x[t * K + k];
            const float* wr = W + k * N;
            for (int64_t j = 0; j < N; ++j) o[j] += xk * wr[j];
        }
    }
}

void matmul_transposed(float* out, const float* x, const float* W, int64_t T, int64_t K,
                       int64_t N) {
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * K;
        for (int64_t j = 0; j < N; ++j) {
            const float* wr = W + j * K;
            float s = 0;
            for (int64_t k = 0; k < K; ++k) s += xr[k] * wr[k];
            out[t * N + j] = s;
        }
    }
}

void gelu(float* x, int64_t n) {
    constexpr float c = 0.7978845608028654f;  // sqrt(2 / pi)
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

void add_inplace(float* out, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] += x[i];
}

void attention(float* out, const float* qkv, const float* k, const float* v, int64_t T,
               int64_t pos0, int64_t C, int64_t n_head) {
    const int64_t hd = C / n_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> scores(pos0 + T);
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            const float* q = qkv + t * 3 * C + h * hd;
            const int64_t n = pos0 + t + 1;  // causal: positions 0..pos0+t
            // Scores, then a stable softmax.
            float mx = -INFINITY;
            for (int64_t s = 0; s < n; ++s) {
                const float* ks = k + s * C + h * hd;
                float dot = 0;
                for (int64_t i = 0; i < hd; ++i) dot += q[i] * ks[i];
                scores[s] = dot * scale;
                if (scores[s] > mx) mx = scores[s];
            }
            float sum = 0;
            for (int64_t s = 0; s < n; ++s) {
                scores[s] = std::exp(scores[s] - mx);
                sum += scores[s];
            }
            float* o = out + t * C + h * hd;
            for (int64_t i = 0; i < hd; ++i) o[i] = 0;
            for (int64_t s = 0; s < n; ++s) {
                const float p = scores[s] / sum;
                const float* vs = v + s * C + h * hd;
                for (int64_t i = 0; i < hd; ++i) o[i] += p * vs[i];
            }
        }
    }
}

}  // namespace gpt2::ops
