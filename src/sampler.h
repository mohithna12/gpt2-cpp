#pragma once
// Next-token selection: greedy, or sampling with temperature, top-k and top-p, following HF's
// semantics (applied in that order; top-k keeps boundary ties; top-p keeps the smallest
// high-probability set whose mass exceeds p, and always at least one token).

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace gpt2 {

struct SamplerConfig {
    float temperature = 0.0f;  // <= 0: greedy (argmax, lowest id on ties)
    int top_k = 50;            // <= 0: disabled. 50 is HF's default when sampling.
    float top_p = 1.0f;        // >= 1: disabled
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& c);

    // Picks the next token from one row of logits. Deterministic for a given seed: the RNG is
    // std::mt19937_64, whose output the C++ standard fixes, and draws don't use
    // implementation-defined distributions.
    int sample(const float* logits, int vocab_size);

    // The filtered, renormalized distribution sample() draws from: (token id, probability)
    // pairs, most likely first (ties by lower id). Exposed for tests.
    std::vector<std::pair<int, double>> distribution(const float* logits, int vocab_size) const;

private:
    SamplerConfig c_;
    std::mt19937_64 rng_;
};

}  // namespace gpt2
