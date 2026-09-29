#include "ops.h"

#include <cmath>
#include <vector>

#if defined(__ARM_NEON) && !defined(GPT2_NO_SIMD)
#include <arm_neon.h>
#define GPT2_NEON 1
#endif

namespace gpt2::ops {

namespace {

#if GPT2_NEON
// MT x NJ tile of dot products: out[i * ldo + j] = x[i, :] . W[j, :], rows of length K.
// Every element is reduced the same way whatever the tile shape (4-lane FMA partial sums, one
// horizontal add, then a scalar FMA tail), so a row's result doesn't depend on how many other
// rows are in the batch; the KV-cache test relies on that.
template <int MT, int NJ>
inline void dot_tile(float* out, int64_t ldo, const float* x, const float* W, int64_t K) {
    float32x4_t acc[MT][NJ];
    for (int i = 0; i < MT; ++i)
        for (int j = 0; j < NJ; ++j) acc[i][j] = vdupq_n_f32(0.0f);
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t xv[MT], wv[NJ];
        for (int i = 0; i < MT; ++i) xv[i] = vld1q_f32(x + i * K + k);
        for (int j = 0; j < NJ; ++j) wv[j] = vld1q_f32(W + j * K + k);
        for (int i = 0; i < MT; ++i)
            for (int j = 0; j < NJ; ++j) acc[i][j] = vfmaq_f32(acc[i][j], xv[i], wv[j]);
    }
    for (int i = 0; i < MT; ++i) {
        for (int j = 0; j < NJ; ++j) {
            float s = vaddvq_f32(acc[i][j]);
            for (int64_t kk = k; kk < K; ++kk) s = std::fma(x[i * K + kk], W[j * K + kk], s);
            out[i * ldo + j] = s;
        }
    }
}

// Rows [t, t+4) of out[T, N] = x @ W + bias, for columns [j, j+16): 16 accumulators, one
// sequential FMA chain per element over k, starting from the bias.
inline void matmul_tile_4x16(float* out, const float* x, const float* W, const float* bias,
                             int64_t K, int64_t N, int64_t t, int64_t j) {
    float32x4_t acc[4][4];
    for (int c = 0; c < 4; ++c) {
        const float32x4_t b0 = bias ? vld1q_f32(bias + j + 4 * c) : vdupq_n_f32(0.0f);
        for (int r = 0; r < 4; ++r) acc[r][c] = b0;
    }
    const float* x0 = x + t * K;
    for (int64_t k = 0; k < K; ++k) {
        const float* wk = W + k * N + j;
        const float32x4_t w0 = vld1q_f32(wk), w1 = vld1q_f32(wk + 4), w2 = vld1q_f32(wk + 8),
                          w3 = vld1q_f32(wk + 12);
        for (int r = 0; r < 4; ++r) {
            const float32x4_t xr = vdupq_n_f32(x0[r * K + k]);
            acc[r][0] = vfmaq_f32(acc[r][0], xr, w0);
            acc[r][1] = vfmaq_f32(acc[r][1], xr, w1);
            acc[r][2] = vfmaq_f32(acc[r][2], xr, w2);
            acc[r][3] = vfmaq_f32(acc[r][3], xr, w3);
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) vst1q_f32(out + (t + r) * N + j + 4 * c, acc[r][c]);
}

// One row of out = x @ W + bias over columns [j0, N) (the T=1 decode path). Streams W rows
// contiguously, four k at a time, so the output row is loaded and stored once per 4 k instead
// of once per k. Per element this is the same arithmetic as the tile: a sequential FMA chain
// over k from the bias.
inline void matmul_row(float* o, const float* xr, const float* W, const float* bias, int64_t K,
                       int64_t N, int64_t j0) {
    for (int64_t j = j0; j < N; ++j) o[j] = bias ? bias[j] : 0.0f;
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        const float32x4_t x0 = vdupq_n_f32(xr[k]), x1 = vdupq_n_f32(xr[k + 1]),
                          x2 = vdupq_n_f32(xr[k + 2]), x3 = vdupq_n_f32(xr[k + 3]);
        const float *w0 = W + k * N, *w1 = w0 + N, *w2 = w1 + N, *w3 = w2 + N;
        int64_t j = j0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t a = vld1q_f32(o + j);
            a = vfmaq_f32(a, x0, vld1q_f32(w0 + j));
            a = vfmaq_f32(a, x1, vld1q_f32(w1 + j));
            a = vfmaq_f32(a, x2, vld1q_f32(w2 + j));
            a = vfmaq_f32(a, x3, vld1q_f32(w3 + j));
            vst1q_f32(o + j, a);
        }
        for (; j < N; ++j) {
            float a = o[j];
            a = std::fma(xr[k], w0[j], a);
            a = std::fma(xr[k + 1], w1[j], a);
            a = std::fma(xr[k + 2], w2[j], a);
            o[j] = std::fma(xr[k + 3], w3[j], a);
        }
    }
    for (; k < K; ++k) {
        const float* wk = W + k * N;
        for (int64_t j = j0; j < N; ++j) o[j] = std::fma(xr[k], wk[j], o[j]);
    }
}
#endif

}  // namespace

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
#if GPT2_NEON
    // 16-column stripes of W (K x 16 floats) outside, 4-row tiles of x inside, so a stripe
    // stays in cache while every row tile passes over it. Leftover rows and columns go through
    // matmul_row, which computes each element identically.
    const int64_t T4 = T / 4 * 4, N16 = N / 16 * 16;
    for (int64_t j = 0; j < N16; j += 16)
        for (int64_t t = 0; t < T4; t += 4) matmul_tile_4x16(out, x, W, bias, K, N, t, j);
    for (int64_t t = 0; t < T4; ++t)
        if (N16 < N) matmul_row(out + t * N, x + t * K, W, bias, K, N, N16);
    for (int64_t t = T4; t < T; ++t) matmul_row(out + t * N, x + t * K, W, bias, K, N, 0);
#else
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
#endif
}

void matmul_transposed(float* out, const float* x, const float* W, int64_t T, int64_t K,
                       int64_t N) {
#if GPT2_NEON
    // Tiles of 4 x-rows by 4 W-rows: each W row is loaded once per 4 positions, and the 4 W rows
    // (4K floats) stay in L1 while every x row passes over them.
    int64_t j = 0;
    for (; j + 4 <= N; j += 4) {
        int64_t t = 0;
        for (; t + 4 <= T; t += 4) dot_tile<4, 4>(out + t * N + j, N, x + t * K, W + j * K, K);
        for (; t < T; ++t) dot_tile<1, 4>(out + t * N + j, N, x + t * K, W + j * K, K);
    }
    for (; j < N; ++j)
        for (int64_t t = 0; t < T; ++t) dot_tile<1, 1>(out + t * N + j, N, x + t * K, W + j * K, K);
#else
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * K;
        for (int64_t j = 0; j < N; ++j) {
            const float* wr = W + j * K;
            float s = 0;
            for (int64_t k = 0; k < K; ++k) s += xr[k] * wr[k];
            out[t * N + j] = s;
        }
    }
#endif
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
#if GPT2_NEON
                float dot;
                dot_tile<1, 1>(&dot, 1, q, ks, hd);
#else
                float dot = 0;
                for (int64_t i = 0; i < hd; ++i) dot += q[i] * ks[i];
#endif
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
#if GPT2_NEON
                const float32x4_t pv = vdupq_n_f32(p);
                int64_t i = 0;
                for (; i + 4 <= hd; i += 4) vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), pv, vld1q_f32(vs + i)));
                for (; i < hd; ++i) o[i] = std::fma(p, vs[i], o[i]);
#else
                for (int64_t i = 0; i < hd; ++i) o[i] += p * vs[i];
#endif
            }
        }
    }
}

}  // namespace gpt2::ops
