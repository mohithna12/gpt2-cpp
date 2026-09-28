#include "forward.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "ops.h"

namespace gpt2 {

KVCache::KVCache(const Config& c)
    : n_layer(c.n_layer),
      n_ctx(c.n_ctx),
      n_embd(c.n_embd),
      k(static_cast<size_t>(c.n_layer) * c.n_ctx * c.n_embd),
      v(k.size()) {}

std::vector<float> forward(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                           Logits which) {
    const Config& c = m.config();
    const Weights& w = m.weights();
    const int64_t T = static_cast<int64_t>(tokens.size());
    const int64_t C = c.n_embd, V = c.vocab_size, pos0 = cache.pos;
    if (cache.n_layer != c.n_layer || cache.n_ctx != c.n_ctx || cache.n_embd != C)
        throw std::invalid_argument("forward: KV cache was made for a different config");
    if (T == 0) throw std::invalid_argument("forward: empty token sequence");
    if (pos0 + T > c.n_ctx)
        throw std::invalid_argument("forward: " + std::to_string(pos0) + " cached + " +
                                    std::to_string(T) + " new tokens exceeds n_ctx " +
                                    std::to_string(c.n_ctx));
    for (int id : tokens)
        if (id < 0 || id >= V)
            throw std::invalid_argument("forward: token id " + std::to_string(id) + " out of range");

    // Residual stream: token + position embeddings.
    std::vector<float> x(T * C);
    for (int64_t t = 0; t < T; ++t) {
        const float* te = w.wte.row(tokens[t]);
        const float* pe = w.wpe.row(pos0 + t);
        for (int64_t i = 0; i < C; ++i) x[t * C + i] = te[i] + pe[i];
    }

    std::vector<float> ln(T * C), qkv(T * 3 * C), att(T * C), proj(T * C), fc(T * 4 * C);
    for (int64_t l = 0; l < c.n_layer; ++l) {
        const LayerWeights& L = w.layers[l];
        // x = x + attn_proj(attention(ln_1(x))), with this call's keys/values appended to the
        // cache first so the new tokens attend to themselves too.
        ops::layernorm(ln.data(), x.data(), L.ln_1_w.data, L.ln_1_b.data, T, C);
        ops::matmul(qkv.data(), ln.data(), L.attn_w.data, L.attn_b.data, T, C, 3 * C);
        float* kc = cache.k_layer(l);
        float* vc = cache.v_layer(l);
        for (int64_t t = 0; t < T; ++t) {
            std::memcpy(kc + (pos0 + t) * C, qkv.data() + t * 3 * C + C, C * sizeof(float));
            std::memcpy(vc + (pos0 + t) * C, qkv.data() + t * 3 * C + 2 * C, C * sizeof(float));
        }
        ops::attention(att.data(), qkv.data(), kc, vc, T, pos0, C, c.n_head);
        ops::matmul(proj.data(), att.data(), L.attn_proj_w.data, L.attn_proj_b.data, T, C, C);
        ops::add_inplace(x.data(), proj.data(), T * C);

        // x = x + fc_proj(gelu(fc(ln_2(x))))
        ops::layernorm(ln.data(), x.data(), L.ln_2_w.data, L.ln_2_b.data, T, C);
        ops::matmul(fc.data(), ln.data(), L.fc_w.data, L.fc_b.data, T, C, 4 * C);
        ops::gelu(fc.data(), T * 4 * C);
        ops::matmul(proj.data(), fc.data(), L.fc_proj_w.data, L.fc_proj_b.data, T, 4 * C, C);
        ops::add_inplace(x.data(), proj.data(), T * C);
    }
    cache.pos = pos0 + T;

    // Final LayerNorm and the tied LM head, for every position or only the last.
    const int64_t first = which == Logits::Last ? T - 1 : 0, n = T - first;
    ops::layernorm(ln.data(), x.data() + first * C, w.ln_f_w.data, w.ln_f_b.data, n, C);
    std::vector<float> logits(n * V);
    ops::matmul_transposed(logits.data(), ln.data(), w.wte.data, n, C, V);
    return logits;
}

std::vector<float> forward(const Model& m, const std::vector<int>& tokens) {
    KVCache cache(m.config());
    return forward(m, cache, tokens, Logits::All);
}

}  // namespace gpt2
