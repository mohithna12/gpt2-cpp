// Checks the tokenizer against ids produced by HF's GPT-2 tokenizer
// (tests/tokenizer_cases.tsv, from scripts/make_tokenizer_fixtures.py), plus byte-exact
// round trips on invalid UTF-8 and rejection of malformed vocab/merges files.
//
//   test_tokenizer <models dir> <tokenizer_cases.tsv>
//
// Needs models/vocab.json and models/merges.txt (scripts/export_gpt2.py); exits 77 (skipped)
// if they are missing.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.h"

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

static std::string unhex(const std::string& h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2)
        out += static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16));
    return out;
}

static std::string ids_str(const std::vector<int>& ids) {
    std::string s;
    for (int id : ids) s += (s.empty() ? "" : " ") + std::to_string(id);
    return s;
}

static void test_matches_hf(const gpt2::Tokenizer& tok, const std::string& cases_path) {
    std::ifstream f(cases_path);
    CHECK(bool(f), "cannot open %s", cases_path.c_str());
    std::string line;
    int n = 0, bad = 0;
    while (std::getline(f, line)) {
        const size_t tab = line.find('\t');
        const std::string text = unhex(line.substr(0, tab));
        std::vector<int> want;
        std::istringstream is(line.substr(tab + 1));
        for (int id; is >> id;) want.push_back(id);

        const std::vector<int> got = tok.encode(text);
        ++n;
        if (got != want && ++bad <= 5)
            CHECK(false, "case %d: text %s\n  got  %s\n  want %s", n, line.substr(0, tab).c_str(),
                  ids_str(got).c_str(), ids_str(want).c_str());
        CHECK(tok.decode(want) == text, "case %d: decode does not round-trip", n);
    }
    CHECK(n > 0, "no cases in %s", cases_path.c_str());
    CHECK(bad == 0, "%d of %d cases differ from HF", bad, n);
}

// Arbitrary bytes, including invalid UTF-8: pieces must tile the input and decode(encode(x))
// must return x exactly.
static void test_arbitrary_bytes(const gpt2::Tokenizer& tok) {
    std::mt19937 rng(0);
    const std::vector<std::string> fixed = {"\xff", "\xc3", "\xc3(", "\xe2\x82", "\xed\xa0\x80",
                                            "\xc0\xaf", "\xf4\x90\x80\x80", "a\x80 b\xfe\n"};
    for (int k = 0; k < 2000 + static_cast<int>(fixed.size()); ++k) {
        std::string s;
        if (k < static_cast<int>(fixed.size())) {
            s = fixed[k];
        } else {
            s.resize(rng() % 32);
            for (char& c : s) c = static_cast<char>(rng() % 4 ? rng() % 256 : " a'\n"[rng() % 4]);
        }
        std::string joined;
        for (std::string_view p : gpt2::pretokenize(s)) {
            CHECK(!p.empty(), "empty piece");
            joined += p;
        }
        CHECK(joined == s, "pieces do not tile the input (case %d)", k);
        CHECK(tok.decode(tok.encode(s)) == s, "byte round trip failed (case %d)", k);
    }
}

static void test_errors(const gpt2::Tokenizer& tok, const std::string& models) {
    bool threw = false;
    try {
        tok.decode({tok.vocab_size()});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "decode accepted an out-of-range id");

    const auto rejects = [&](const std::string& label, const std::string& vocab,
                             const std::string& merges) {
        const std::string v = "bad_vocab_" + label + ".json", m = "bad_merges_" + label + ".txt";
        std::ofstream(v) << vocab;
        std::ofstream(m) << merges;
        bool threw = false;
        try {
            gpt2::Tokenizer::load(v, m);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "malformed input '%s' was accepted", label.c_str());
        std::remove(v.c_str());
        std::remove(m.c_str());
    };
    std::ifstream vf(models + "/vocab.json"), mf(models + "/merges.txt");
    const std::string vocab{std::istreambuf_iterator<char>(vf), {}};
    const std::string merges{std::istreambuf_iterator<char>(mf), {}};
    rejects("truncated_vocab", vocab.substr(0, vocab.size() / 2), merges);
    rejects("not_json", "[1, 2]", merges);
    rejects("small_vocab", "{\"a\": 0}", merges);
    rejects("bad_merge_line", vocab, "#version: 0.2\nabc\n");
    rejects("unknown_merge", vocab, "#version: 0.2\n\xc4\xa0 zzzzzzzz\n");
    threw = false;
    try {
        gpt2::Tokenizer::load(models + "/does_not_exist.json", models + "/merges.txt");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "missing vocab file was accepted");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <models dir> <tokenizer_cases.tsv>\n", argv[0]);
        return 2;
    }
    const std::string models = argv[1];
    if (!std::ifstream(models + "/vocab.json") || !std::ifstream(models + "/merges.txt")) {
        std::printf("SKIP: %s/vocab.json or merges.txt missing; run scripts/export_gpt2.py\n",
                    models.c_str());
        return 77;
    }
    try {
        const gpt2::Tokenizer tok =
            gpt2::Tokenizer::load(models + "/vocab.json", models + "/merges.txt");
        CHECK(tok.vocab_size() == 50257, "vocab size %d", tok.vocab_size());
        test_matches_hf(tok, argv[2]);
        test_arbitrary_bytes(tok);
        test_errors(tok, models);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_tokenizer: all checks passed\n");
    return 0;
}
