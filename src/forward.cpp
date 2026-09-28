#include "forward.h"

#include <stdexcept>
#include <string>

#include "ops.h"

namespace gpt2 {

std::vector<float> forward(const Model& m, const std::vector<int>& tokens) {
    const Config& c = m.config();
    const Weights& w = m.weights();
    const int64_t T = static_cast<int64_t>(tokens.size());
    const int64_t C = c.n_embd, V = c.vocab_size;
    if (T == 0) throw std::invalid_argument("forward: empty token sequence");
    if (T > c.n_ctx)
        throw std::invalid_argument("forward: " + std::to_string(T) + " tokens exceeds n_ctx " +
                                    std::to_string(c.n_ctx));

    // Residual stream: token + position embeddings.
    std::vector<float> x(T * C);
    for (int64_t t = 0; t < T; ++t) {
        const int id = tokens[t];
        if (id < 0 || id >= V)
            throw std::invalid_argument("forward: token id " + std::to_string(id) + " out of range");
        const float* te = w.wte.row(id);
        const float* pe = w.wpe.row(t);
        for (int64_t i = 0; i < C; ++i) x[t * C + i] = te[i] + pe[i];
    }

    std::vector<float> ln(T * C), qkv(T * 3 * C), att(T * C), proj(T * C), fc(T * 4 * C);
    for (const LayerWeights& L : w.layers) {
        // x = x + attn_proj(attention(ln_1(x)))
        ops::layernorm(ln.data(), x.data(), L.ln_1_w.data, L.ln_1_b.data, T, C);
        ops::matmul(qkv.data(), ln.data(), L.attn_w.data, L.attn_b.data, T, C, 3 * C);
        ops::attention(att.data(), qkv.data(), T, C, c.n_head);
        ops::matmul(proj.data(), att.data(), L.attn_proj_w.data, L.attn_proj_b.data, T, C, C);
        ops::add_inplace(x.data(), proj.data(), T * C);

        // x = x + fc_proj(gelu(fc(ln_2(x))))
        ops::layernorm(ln.data(), x.data(), L.ln_2_w.data, L.ln_2_b.data, T, C);
        ops::matmul(fc.data(), ln.data(), L.fc_w.data, L.fc_b.data, T, C, 4 * C);
        ops::gelu(fc.data(), T * 4 * C);
        ops::matmul(proj.data(), fc.data(), L.fc_proj_w.data, L.fc_proj_b.data, T, 4 * C, C);
        ops::add_inplace(x.data(), proj.data(), T * C);
    }

    ops::layernorm(ln.data(), x.data(), w.ln_f_w.data, w.ln_f_b.data, T, C);
    std::vector<float> logits(T * V);
    ops::matmul_transposed(logits.data(), ln.data(), w.wte.data, T, C, V);  // tied LM head
    return logits;
}

}  // namespace gpt2
