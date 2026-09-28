// Forward-pass tests.
//
//   test_forward tiny <tiny.bin>   invariants on the random tiny checkpoint (no download):
//                                  causality (a prefix's logits don't depend on later tokens),
//                                  determinism, finite outputs, input validation.
//   test_forward hf <models dir>   logits for every position vs HF float32 (models/logits_ref.bin
//                                  from scripts/make_logits_ref.py): max abs diff <= 2e-3 and
//                                  the argmax must match. Exits 77 (skipped) if files are missing.
//
// Why 2e-3 and not 1e-4: GPT-2 logits reach ~300, and HF's own float32 logits are up to ~6.5e-4
// from the float64 truth, so no independent float32 implementation can match them to 1e-4.
// Calibrated on scripts/make_logits_ref.py's prompts: correct code's worst is 7.9e-4; exact-erf
// GELU gives 7.5e-2..0.72 (fails all prompts); LayerNorm eps 1e-6 fails 5 of 6 prompts.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "forward.h"
#include "model.h"

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

static constexpr double kTol = 2e-3;  // max abs logit difference vs HF float32

static void test_tiny(const std::string& path) {
    const gpt2::Model m = gpt2::Model::load(path);
    const int64_t V = m.config().vocab_size;
    std::vector<int> tokens;
    for (int i = 0; i < m.config().n_ctx; ++i) tokens.push_back((i * 37 + 11) % V);

    const std::vector<float> full = gpt2::forward(m, tokens);
    CHECK(full.size() == tokens.size() * V, "logits size");
    bool finite = true;
    for (float v : full) finite = finite && std::isfinite(v);
    CHECK(finite, "non-finite logits");
    CHECK(gpt2::forward(m, tokens) == full, "forward is not deterministic");

    // Causality: the logits of every prefix must equal the corresponding rows of the full run.
    for (size_t k : {size_t(1), size_t(2), size_t(7), tokens.size() - 1}) {
        const std::vector<int> prefix(tokens.begin(), tokens.begin() + k);
        const std::vector<float> p = gpt2::forward(m, prefix);
        CHECK(std::memcmp(p.data(), full.data(), p.size() * sizeof(float)) == 0,
              "prefix of length %zu gives different logits", k);
    }

    const auto rejects = [&](const char* label, const std::vector<int>& t) {
        bool threw = false;
        try {
            gpt2::forward(m, t);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw, "forward accepted %s", label);
    };
    rejects("an empty sequence", {});
    rejects("a negative id", {1, -1});
    rejects("an id >= vocab", {1, static_cast<int>(V)});
    rejects("a sequence longer than n_ctx", std::vector<int>(m.config().n_ctx + 1, 0));
}

static int test_hf(const std::string& dir) {
    std::ifstream f(dir + "/logits_ref.bin", std::ios::binary);
    if (!f || !std::ifstream(dir + "/gpt2.bin")) {
        std::printf("SKIP: need %s/gpt2.bin and logits_ref.bin (scripts/export_gpt2.py, "
                    "scripts/make_logits_ref.py)\n", dir.c_str());
        return 77;
    }
    const gpt2::Model m = gpt2::Model::load(dir + "/gpt2.bin");
    const auto u32 = [&] {
        uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 4);  // little-endian host
        if (!f) throw std::runtime_error("logits_ref.bin: truncated");
        return v;
    };
    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "LGTS", 4) != 0 || u32() != 1)
        throw std::runtime_error("logits_ref.bin: bad header");
    const uint32_t n_prompts = u32(), V = u32();
    CHECK(V == static_cast<uint32_t>(m.config().vocab_size), "vocab size mismatch");

    double worst = 0;
    for (uint32_t p = 0; p < n_prompts; ++p) {
        const uint32_t T = u32();
        std::vector<int> ids(T);
        std::vector<float> want(static_cast<size_t>(T) * V);
        f.read(reinterpret_cast<char*>(ids.data()), T * sizeof(int));
        f.read(reinterpret_cast<char*>(want.data()), want.size() * sizeof(float));
        if (!f) throw std::runtime_error("logits_ref.bin: truncated");

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> got = gpt2::forward(m, ids);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        double max_diff = 0;
        size_t argmax_mismatch = 0;
        for (uint32_t t = 0; t < T; ++t) {
            const float* g = got.data() + static_cast<size_t>(t) * V;
            const float* w = want.data() + static_cast<size_t>(t) * V;
            size_t ag = 0, aw = 0;
            for (uint32_t j = 0; j < V; ++j) {
                max_diff = std::fmax(max_diff, std::fabs(double(g[j]) - w[j]));
                if (g[j] > g[ag]) ag = j;
                if (w[j] > w[aw]) aw = j;
            }
            argmax_mismatch += ag != aw;
        }
        std::printf("prompt %u: T=%3u  max|diff|=%.3g (%3.0f%% of tolerance)  "
                    "argmax mismatches=%zu  (%.0f ms)\n",
                    p, T, max_diff, 100 * max_diff / kTol, argmax_mismatch, ms);
        CHECK(max_diff <= kTol, "prompt %u: max abs diff %.3g > %.0e", p, max_diff, kTol);
        CHECK(argmax_mismatch == 0, "prompt %u: argmax differs at %zu positions", p,
              argmax_mismatch);
        worst = std::fmax(worst, max_diff);
    }
    std::printf("worst max|diff| over %u prompts: %.3g (%.0f%% of tolerance %.0e)\n", n_prompts,
                worst, 100 * worst / kTol, kTol);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s tiny <tiny.bin> | hf <models dir>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    try {
        if (mode == "tiny") {
            test_tiny(argv[2]);
        } else if (mode == "hf") {
            if (test_hf(argv[2]) == 77) return 77;
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_forward %s: all checks passed\n", mode.c_str());
    return 0;
}
