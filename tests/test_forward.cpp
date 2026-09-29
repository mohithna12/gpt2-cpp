// Forward-pass tests.
//
//   test_forward tiny <tiny.bin>
//       Invariants on the random tiny checkpoint (no download): causality (a prefix's logits
//       don't depend on later tokens), determinism, finite outputs, input validation, and the
//       KV cache: token-by-token and chunked decoding reproduce the full pass bit for bit.
//   test_forward hf <gpt2.bin> <ref.bin>
//       Differential test vs HF float32, using a reference from scripts/make_logits_ref.py
//       (format v3). Compares log-probabilities (log-softmax of the logits), max abs diff
//       <= 2e-3: over the full vocabulary at the stored rows, and for HF's top-20 tokens at
//       every position. The argmax must match HF at every position where HF's top1-top2 logit
//       gap is >= 2e-3. At closer near-ties a flip is allowed (and printed) if our pick is one
//       of HF's top-20, whose log-probs are checked anyway. Exits 77 (skipped) if either file is
//       missing. ctest runs it twice: forward_hf (6 prompts, every row) and
//       diff_hf (tests/diff_prompts.jsonl: 94 prompts up to n_ctx, sampled rows; "slow").
//   test_forward decode <gpt2.bin> <ref.bin>
//       Same comparison, but tokens are fed one at a time through the KV cache (the generation
//       path). ctest runs it on the forward_hf prompts as forward_hf_decode.
//
// Why log-probs at 2e-3 (the project first targeted raw logits within 1e-4): GPT-2 logits reach
// ~300, and float32 logits, HF's own included, carry rare whole-row shifts of up to ~7e-3 vs a
// float64 run. Softmax cancels those shifts; HF float32 log-probs stay within ~1e-3 of float64
// at every position, flat with length. See README "Correctness" for the calibration.

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

static constexpr double kTol = 2e-3;  // max abs log-prob difference vs HF float32

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

    // KV cache: feeding the sequence in pieces must reproduce the full pass bit for bit.
    const auto same_rows = [&](const std::vector<float>& got, size_t first_row) {
        return std::memcmp(got.data(), full.data() + first_row * V, got.size() * sizeof(float)) == 0;
    };
    for (const std::vector<size_t>& chunks :
         {std::vector<size_t>(tokens.size(), 1), std::vector<size_t>{3, 1, 5, 20, 2, 33}}) {
        gpt2::KVCache cache(m.config());
        size_t at = 0;
        for (size_t n : chunks) {
            const std::vector<int> piece(tokens.begin() + at, tokens.begin() + at + n);
            CHECK(same_rows(gpt2::forward(m, cache, piece), at),
                  "chunk [%zu, %zu) differs from the full pass", at, at + n);
            at += n;
        }
        CHECK(at == tokens.size() && cache.pos == static_cast<int64_t>(at), "chunks/cache pos");
    }
    {
        gpt2::KVCache cache(m.config());
        const std::vector<int> head(tokens.begin(), tokens.begin() + 10);
        const std::vector<float> last = gpt2::forward(m, cache, head, gpt2::Logits::Last);
        CHECK(last.size() == size_t(V) && same_rows(last, 9), "Logits::Last != last row of All");

        // Overflowing n_ctx throws and leaves the cache usable and unchanged.
        bool threw = false;
        try {
            gpt2::forward(m, cache, std::vector<int>(m.config().n_ctx, 0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw && cache.pos == 10, "overflow: threw=%d pos=%lld", threw, (long long)cache.pos);
        CHECK(same_rows(gpt2::forward(m, cache, {tokens[10]}), 10), "cache unusable after a throw");

        cache.clear();
        CHECK(same_rows(gpt2::forward(m, cache, head), 0), "clear() does not reset the cache");

        gpt2::Config other = m.config();
        other.n_ctx /= 2;
        gpt2::KVCache wrong(other);
        threw = false;
        try {
            gpt2::forward(m, wrong, head);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw, "forward accepted a cache made for a different config");
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

struct Reader {
    std::ifstream f;
    std::string path;
    void read(void* dst, size_t n) {
        f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error(path + ": truncated");
    }
    uint32_t u32() {
        uint32_t v;
        read(&v, 4);  // little-endian host
        return v;
    }
    template <typename T>
    std::vector<T> vec(size_t n) {
        std::vector<T> v(n);
        read(v.data(), n * sizeof(T));
        return v;
    }
};

// Log-softmax of one row, in double.
static void log_softmax(const float* x, size_t n, std::vector<double>& out) {
    out.resize(n);
    double mx = x[0];
    for (size_t j = 1; j < n; ++j) mx = std::fmax(mx, x[j]);
    double sum = 0;
    for (size_t j = 0; j < n; ++j) sum += std::exp(double(x[j]) - mx);
    const double lse = mx + std::log(sum);
    for (size_t j = 0; j < n; ++j) out[j] = double(x[j]) - lse;
}

static int test_hf(const std::string& model_path, const std::string& ref_path, bool decode) {
    Reader r{std::ifstream(ref_path, std::ios::binary), ref_path};
    if (!r.f || !std::ifstream(model_path)) {
        std::printf("SKIP: need %s and %s (scripts/export_gpt2.py, scripts/make_logits_ref.py)\n",
                    model_path.c_str(), ref_path.c_str());
        return 77;
    }
    const gpt2::Model m = gpt2::Model::load(model_path);
    char magic[4];
    r.read(magic, 4);
    if (std::memcmp(magic, "LGTS", 4) != 0 || r.u32() != 3)
        throw std::runtime_error(ref_path + ": bad header (regenerate with make_logits_ref.py)");
    const uint32_t n_prompts = r.u32(), V = r.u32(), K = r.u32();
    CHECK(V == static_cast<uint32_t>(m.config().vocab_size), "vocab size mismatch");

    double worst = 0, total_ms = 0;
    std::string worst_name;
    size_t total_tokens = 0, total_rows = 0, total_mismatch = 0, total_flips = 0;
    std::vector<double> lp;
    for (uint32_t p = 0; p < n_prompts; ++p) {
        std::string name(r.u32(), '\0');
        r.read(name.data(), name.size());
        const uint32_t T = r.u32();
        const std::vector<int> ids = r.vec<int>(T);
        const std::vector<int> top1 = r.vec<int>(T);
        const std::vector<float> gap = r.vec<float>(T);
        const std::vector<int> topk_ids = r.vec<int>(size_t(T) * K);
        const std::vector<float> topk_lp = r.vec<float>(size_t(T) * K);
        const std::vector<uint32_t> rows = r.vec<uint32_t>(r.u32());
        const std::vector<float> want = r.vec<float>(rows.size() * size_t(V));

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> got;
        if (decode) {  // one token at a time through the KV cache
            gpt2::KVCache cache(m.config());
            got.reserve(size_t(T) * V);
            for (int id : ids) {
                const std::vector<float> row = gpt2::forward(m, cache, {id});
                got.insert(got.end(), row.begin(), row.end());
            }
        } else {
            got = gpt2::forward(m, ids);
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // Every position: argmax, and log-probs of HF's top-K tokens.
        size_t mismatch = 0, near_tie_flips = 0, row_k = 0;
        double max_diff = 0;
        for (uint32_t t = 0; t < T; ++t) {
            const float* g = got.data() + size_t(t) * V;
            uint32_t ag = 0;
            for (uint32_t j = 1; j < V; ++j)
                if (g[j] > g[ag]) ag = j;
            if (static_cast<int>(ag) != top1[t]) {
                // Below the tolerance, which of two near-tied tokens wins is float32 noise.
                bool in_topk = false;
                for (uint32_t k = 0; k < K; ++k) in_topk |= topk_ids[size_t(t) * K + k] == int(ag);
                const bool allowed = gap[t] < kTol && in_topk;
                (allowed ? near_tie_flips : mismatch)++;
                std::printf("  %s pos %u: argmax %u, HF %d (HF top1-top2 gap %.3g)%s\n",
                            name.c_str(), t, ag, top1[t], gap[t],
                            allowed ? ": near-tie, allowed" : "");
            }
            log_softmax(g, V, lp);
            for (uint32_t k = 0; k < K; ++k) {
                const size_t i = size_t(t) * K + k;
                max_diff = std::fmax(max_diff, std::fabs(lp[topk_ids[i]] - topk_lp[i]));
            }
            // Full vocabulary at the stored rows (sorted, so walk them alongside t).
            if (row_k < rows.size() && rows[row_k] == t) {
                const float* w = want.data() + row_k * V;
                for (uint32_t j = 0; j < V; ++j)
                    max_diff = std::fmax(max_diff, std::fabs(lp[j] - w[j]));
                ++row_k;
            }
        }
        CHECK(row_k == rows.size(), "%s: stored rows not sorted or out of range", name.c_str());
        std::printf("%-34s T=%4u rows=%3zu  max|dlogp|=%.3g (%3.0f%% of tol)  argmax mism=%zu  "
                    "%.0f ms\n",
                    name.c_str(), T, rows.size(), max_diff, 100 * max_diff / kTol, mismatch, ms);
        CHECK(max_diff <= kTol, "%s: max abs log-prob diff %.3g > %.0e", name.c_str(), max_diff,
              kTol);
        CHECK(mismatch == 0, "%s: argmax differs at %zu positions", name.c_str(), mismatch);
        if (max_diff > worst) worst = max_diff, worst_name = name;
        total_ms += ms;
        total_tokens += T;
        total_rows += rows.size();
        total_mismatch += mismatch;
        total_flips += near_tie_flips;
    }
    std::printf("%u prompts, %zu tokens (argmax + top-%u log-probs checked at all), %zu full rows: "
                "worst max|dlogp| %.3g (%.0f%% of tolerance %.0e) in %s; %zu argmax mismatches, "
                "%zu near-tie flips allowed; forward %.1f s\n",
                n_prompts, total_tokens, K, total_rows, worst, 100 * worst / kTol, kTol,
                worst_name.c_str(), total_mismatch, total_flips, total_ms / 1000);
    return 0;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (!((mode == "tiny" && argc == 3) || ((mode == "hf" || mode == "decode") && argc == 4))) {
        std::fprintf(stderr, "usage: %s tiny <tiny.bin> | {hf|decode} <gpt2.bin> <ref.bin>\n",
                     argv[0]);
        return 2;
    }
    try {
        if (mode == "tiny") {
            test_tiny(argv[2]);
        } else if (test_hf(argv[2], argv[3], mode == "decode") == 77) {
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
    std::printf("test_forward %s: all checks passed\n", mode.c_str());
    return 0;
}
