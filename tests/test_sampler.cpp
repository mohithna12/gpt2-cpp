// Sampler tests.
//
//   test_sampler unit
//       Greedy and tie rules, top-k boundary ties, a worked top-p example, seed
//       reproducibility, and empirical draw frequencies against the target distribution.
//   test_sampler hf <sampler_ref.bin>
//       The filtered distribution must equal what HF's warpers produce on real GPT-2 logits
//       (scripts/make_sampler_ref.py): same token set, probabilities within 1e-6. The token
//       sets may differ only at the cut itself, and each such token is printed:
//         - exact score ties straddling the cut (torch.sort is unstable; we break ties by id);
//         - top-p: tokens whose cumulative mass is within 1e-6 of 1 - p (HF sums in float32).
//       Exits 77 (skipped) if the file is missing.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sampler.h"

static int failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);            \
            std::fprintf(stderr, "\n");                   \
            ++failures;                                   \
        }                                                 \
    } while (0)

using Dist = std::vector<std::pair<int, double>>;

static Dist dist(const std::vector<float>& logits, float t, int k, float p) {
    return gpt2::Sampler({t, k, p, 0}).distribution(logits.data(), int(logits.size()));
}

static std::vector<int> ids_of(const Dist& d) {
    std::vector<int> ids;
    for (const auto& e : d) ids.push_back(e.first);
    return ids;
}

static void test_unit() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 3.0f, 0.5f};

    // Greedy: argmax, lowest id on ties (like torch.argmax); temperature 0 and top_k 1 agree.
    gpt2::Sampler greedy({0.0f, 50, 1.0f, 0});
    CHECK(greedy.sample(logits.data(), 5) == 1, "greedy should pick id 1 (first of the tie)");
    gpt2::Sampler k1({1.0f, 1, 1.0f, 123});
    for (int i = 0; i < 100; ++i) {
        const int id = k1.sample(logits.data(), 5);
        CHECK(id == 1 || id == 3, "top_k=1 drew %d, outside the tied maxima {1, 3}", id);
    }

    // No filtering: exactly softmax(logits / T), most likely first, ties by lower id.
    const Dist d = dist(logits, 0.5f, 0, 1.0f);
    CHECK((ids_of(d) == std::vector<int>{1, 3, 2, 0, 4}), "order of the unfiltered distribution");
    double z = 0;
    for (float l : logits) z += std::exp(double(l / 0.5f));
    for (const auto& [id, pr] : d)
        CHECK(std::fabs(pr - std::exp(double(logits[id] / 0.5f)) / z) < 1e-12, "softmax prob of %d",
              id);

    // Top-k keeps boundary ties: k=1 over {3, 3} keeps both, like HF's `scores < kth` mask.
    CHECK((ids_of(dist(logits, 1.0f, 1, 1.0f)) == std::vector<int>{1, 3}), "top-k boundary ties");
    CHECK((ids_of(dist(logits, 1.0f, 3, 1.0f)) == std::vector<int>{1, 3, 2}), "top-k = 3");

    // Top-p worked example: probs 0.5, 0.3, 0.15, 0.05. With p = 0.75, HF drops tokens whose
    // ascending cumulative mass is <= 0.25: 0.05 and 0.05+0.15=0.2 go, 0.5 stays -> {0, 1}.
    const std::vector<float> lp = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
    CHECK((ids_of(dist(lp, 1.0f, 0, 0.75f)) == std::vector<int>{0, 1}), "top-p 0.75");
    CHECK((ids_of(dist(lp, 1.0f, 0, 0.81f)) == std::vector<int>{0, 1, 2}), "top-p 0.81");
    CHECK((ids_of(dist(lp, 1.0f, 0, 0.01f)) == std::vector<int>{0}), "top-p keeps >= 1 token");
    const Dist dp = dist(lp, 1.0f, 0, 0.75f);
    CHECK(std::fabs(dp[0].second - 0.5 / 0.8) < 1e-6 && std::fabs(dp[1].second - 0.3 / 0.8) < 1e-6,
          "top-p renormalizes");

    // Seeds: same seed, same sequence; different seed, different sequence.
    const auto draws = [&](uint64_t seed) {
        gpt2::Sampler s({1.0f, 0, 1.0f, seed});
        std::vector<int> out;
        for (int i = 0; i < 64; ++i) out.push_back(s.sample(logits.data(), 5));
        return out;
    };
    CHECK(draws(7) == draws(7), "same seed must reproduce the same draws");
    CHECK(draws(7) != draws(8), "different seeds should give different draws");

    // Empirical frequencies within 5 standard deviations of the target probabilities.
    const int n = 200000;
    gpt2::Sampler s({0.8f, 0, 0.95f, 42});
    const Dist target = dist(logits, 0.8f, 0, 0.95f);
    std::map<int, int> count;
    for (int i = 0; i < n; ++i) ++count[s.sample(logits.data(), 5)];
    for (const auto& [id, pr] : target) {
        const double sd = std::sqrt(n * pr * (1 - pr));
        CHECK(std::fabs(count[id] - n * pr) <= 5 * sd, "token %d drawn %d times, expected %.0f", id,
              count[id], n * pr);
    }
    int outside = 0;
    for (const auto& [id, c] : count) {
        bool in = false;
        for (const auto& e : target) in |= e.first == id;
        if (!in) outside += c;
    }
    CHECK(outside == 0, "%d draws outside the top-p set", outside);

    bool threw = false;
    try {
        gpt2::Sampler bad({1.0f, 0, 0.0f, 0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "top_p = 0 accepted");
}

static int test_hf(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("SKIP: need %s (scripts/make_sampler_ref.py)\n", path.c_str());
        return 77;
    }
    const auto rd = [&](void* p, size_t n) {
        f.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error(path + ": truncated");
    };
    const auto u32 = [&] { uint32_t v; rd(&v, 4); return v; };
    char magic[4];
    rd(magic, 4);
    if (std::memcmp(magic, "SMPL", 4) != 0 || u32() != 1) throw std::runtime_error("bad header");
    const uint32_t n_rows = u32(), V = u32();
    std::vector<std::vector<float>> rows(n_rows, std::vector<float>(V));
    for (auto& r : rows) rd(r.data(), V * sizeof(float));

    const uint32_t n_cases = u32();
    double worst = 0;
    size_t total_boundary = 0;
    for (uint32_t c = 0; c < n_cases; ++c) {
        uint32_t row, m;
        float t, p;
        int32_t k;
        rd(&row, 4), rd(&t, 4), rd(&k, 4), rd(&p, 4), m = u32();
        std::vector<int32_t> ids(m);
        std::vector<double> probs(m);
        rd(ids.data(), m * 4), rd(probs.data(), m * 8);

        std::map<int, double> got, want;
        for (const auto& [id, pr] : dist(rows.at(row), t, k, p)) got[id] = pr;
        for (uint32_t i = 0; i < m; ++i) want[ids[i]] = probs[i];

        // Tokens in exactly one of the two sets must sit at the cut. `pre` is our distribution
        // before top-p, most likely first; `below` is a token's mass plus all less likely.
        const Dist pre = dist(rows.at(row), t, k, 1.0f);
        std::map<int, std::pair<double, double>> at;  // id -> (probability, below)
        double below = 1.0;
        for (const auto& [id, pr] : pre) at[id] = {pr, below}, below -= pr;
        size_t boundary = 0, bad = 0;
        const auto check_side = [&](const std::map<int, double>& a, const std::map<int, double>& b) {
            for (const auto& [id, pr] : a) {
                if (b.count(id)) continue;
                const auto [prob, blw] = at.at(id);
                bool tie = false;  // an exactly tied token on the other side of the cut
                for (const auto& [id2, pr2] : b)
                    if (!a.count(id2) && at.at(id2).first == prob) tie = true;
                const bool near_cut = p < 1.0f && std::fabs(blw - (1.0 - double(p))) <= 1e-6;
                (tie || near_cut ? boundary : bad)++;
                std::printf("    token %d only on one side: prob %.3g, mass at/below %.9f (1-p = %.9f)%s\n",
                            id, prob, blw, 1.0 - double(p),
                            tie ? ": exact tie across the cut" : near_cut ? ": at the cut" : "");
            }
        };
        check_side(got, want);
        check_side(want, got);
        double diff = 0;
        for (const auto& [id, pr] : want)
            if (got.count(id)) diff = std::fmax(diff, std::fabs(got[id] - pr));
        std::printf("row %u T=%.2f k=%3d p=%.3f: HF keeps %5u tokens, ours %5zu  max|dp|=%.2g%s\n",
                    row, t, k, p, m, got.size(), diff,
                    boundary ? "  (boundary differences allowed)" : "");
        CHECK(bad == 0, "row %u T=%g k=%d p=%g: %zu tokens differ from HF away from the cut", row,
              t, k, p, bad);
        CHECK(diff <= 1e-6, "row %u T=%g k=%d p=%g: probability diff %.3g", row, t, k, p, diff);
        total_boundary += boundary;
        worst = std::fmax(worst, diff);
    }
    std::printf("%u cases: worst max|dp| %.2g; %zu tokens differ, all at the cut\n", n_cases,
                worst, total_boundary);
    return 0;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (!((mode == "unit" && argc == 2) || (mode == "hf" && argc == 3))) {
        std::fprintf(stderr, "usage: %s unit | hf <sampler_ref.bin>\n", argv[0]);
        return 2;
    }
    try {
        if (mode == "unit") {
            test_unit();
        } else if (test_hf(argv[2]) == 77) {
            return 77;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_sampler %s: all checks passed\n", mode.c_str());
    return 0;
}
