#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpt2 {

Sampler::Sampler(const SamplerConfig& c) : c_(c), rng_(c.seed) {
    if (!(c.top_p > 0.0f)) throw std::invalid_argument("top_p must be > 0");
}

std::vector<std::pair<int, double>> Sampler::distribution(const float* logits, int V) const {
    if (c_.temperature <= 0.0f)  // greedy: all mass on the argmax
        return {{static_cast<int>(std::max_element(logits, logits + V) - logits), 1.0}};
    // Temperature in float32, as HF does (scores / temperature).
    std::vector<std::pair<float, int>> s(V);
    for (int i = 0; i < V; ++i) s[i] = {logits[i] / c_.temperature, i};
    const auto more_likely = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };

    // Top-k: keep every token scoring >= the k-th largest score (boundary ties included).
    size_t n = s.size();
    if (c_.top_k > 0 && c_.top_k < V) {
        std::nth_element(s.begin(), s.begin() + (c_.top_k - 1), s.end(), more_likely);
        const float kth = s[c_.top_k - 1].first;
        n = static_cast<size_t>(
            std::partition(s.begin(), s.end(), [&](const auto& e) { return e.first >= kth; }) -
            s.begin());
    }
    s.resize(n);

    // Softmax over the survivors (masked tokens have probability 0), in double.
    float mx = s[0].first;
    for (const auto& e : s) mx = std::max(mx, e.first);
    double sum = 0;
    for (const auto& e : s) sum += std::exp(double(e.first) - mx);

    // Only the most likely tokens can survive top-p. When the top kPartial already hold enough
    // mass that the cut must fall among them (with a margin for rounding), sort just those
    // instead of all survivors: same set, since selection and sort use the same total order.
    // Sorting the full 50k-token vocabulary every step cost ~30% of decode throughput.
    constexpr size_t kPartial = 1024;
    size_t m = n;
    if (c_.top_p < 1.0f && n > kPartial) {
        std::nth_element(s.begin(), s.begin() + kPartial, s.end(), more_likely);
        double top = 0;
        for (size_t i = 0; i < kPartial; ++i) top += std::exp(double(s[i].first) - mx);
        if (1.0 - top / sum <= (1.0 - double(c_.top_p)) - 1e-9) m = kPartial;
    }
    std::sort(s.begin(), s.begin() + m, more_likely);
    std::vector<std::pair<int, double>> out(m);
    for (size_t i = 0; i < m; ++i) out[i] = {s[i].second, std::exp(double(s[i].first) - mx) / sum};

    // Top-p: HF drops tokens whose cumulative probability, summed from the least likely up, is
    // <= 1 - p. Equivalently, keep the most likely prefix up to and including the token where
    // the mass above it first reaches p. At least one token always stays.
    if (c_.top_p < 1.0f) {
        const double cut = 1.0 - double(c_.top_p);
        double below = 1.0;  // mass of this token and everything less likely
        size_t keep = 0;
        while (keep < out.size() && (keep == 0 || below > cut)) below -= out[keep++].second;
        out.resize(keep);
        double kept = 0;
        for (const auto& e : out) kept += e.second;
        for (auto& e : out) e.second /= kept;
    }
    return out;
}

int Sampler::sample(const float* logits, int V) {
    if (c_.temperature <= 0.0f)
        return static_cast<int>(std::max_element(logits, logits + V) - logits);
    const std::vector<std::pair<int, double>> dist = distribution(logits, V);
    // Uniform in [0, 1) from the top 53 bits, without std::uniform_real_distribution, whose
    // algorithm is implementation-defined.
    const double u = double(rng_() >> 11) * 0x1.0p-53;
    double acc = 0;
    for (const auto& [id, p] : dist) {
        acc += p;
        if (u < acc) return id;
    }
    return dist.back().first;  // u within rounding of 1
}

}  // namespace gpt2
