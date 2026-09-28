// Loads a checkpoint and compares every tensor against reference stats written by
// scripts/make_tiny_checkpoint.py (name, count, sum, first, last). Also checks that
// corrupted files are rejected with an error instead of crashing.
//
//   test_loader <tiny.bin>          (expects <tiny.bin>.ref next to it)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

static bool close(double a, double b, double rel = 1e-5, double abs = 1e-4) {
    return std::fabs(a - b) <= abs + rel * std::fabs(b);
}

static void test_matches_reference(const std::string& path) {
    const gpt2::Model m = gpt2::Model::load(path);
    const auto& c = m.config();
    CHECK(c.n_layer == 2 && c.n_head == 4 && c.n_embd == 32 && c.n_ctx == 64 && c.vocab_size == 101,
          "unexpected config");
    CHECK(m.weights().layers.size() == 2, "layer count");
    CHECK(m.weights().layers[1].fc_w.has_shape({32, 128}), "fc_w shape %s",
          m.weights().layers[1].fc_w.shape_str().c_str());

    std::ifstream ref(path + ".ref");
    CHECK(bool(ref), "missing %s.ref", path.c_str());
    size_t idx = 0;
    std::string line;
    while (std::getline(ref, line)) {
        std::istringstream is(line);
        std::string name;
        int64_t count;
        double sum, first, last;
        is >> name >> count >> sum >> first >> last;
        CHECK(idx < m.tensors().size(), "more reference rows than tensors");
        if (idx >= m.tensors().size()) return;
        const auto& [tname, t] = m.tensors()[idx++];
        CHECK(tname == name, "order: got %s, want %s", tname.c_str(), name.c_str());
        CHECK(t.numel() == count, "%s: numel", name.c_str());
        CHECK(reinterpret_cast<uintptr_t>(t.data) % 64 == 0, "%s: not 64-byte aligned", name.c_str());
        double s = 0;
        for (int64_t i = 0; i < t.numel(); ++i) s += t.data[i];
        CHECK(close(s, sum), "%s: sum %.9g want %.9g", name.c_str(), s, sum);
        CHECK(t.data[0] == static_cast<float>(first), "%s: first element", name.c_str());
        CHECK(t.data[t.numel() - 1] == static_cast<float>(last), "%s: last element", name.c_str());
    }
    CHECK(idx == m.tensors().size(), "reference has %zu rows, model has %zu tensors", idx,
          m.tensors().size());
}

static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

static void expect_rejected(const std::string& label, const std::vector<char>& bytes) {
    const std::string p = "corrupt_" + label + ".bin";
    std::ofstream(p, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    bool threw = false;
    try {
        gpt2::Model::load(p);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "corrupt file '%s' was accepted", label.c_str());
    std::remove(p.c_str());
}

static void test_rejects_corruption(const std::string& path) {
    const std::vector<char> good = slurp(path);

    auto bad_magic = good;
    bad_magic[0] = 'X';
    expect_rejected("magic", bad_magic);

    auto bad_version = good;
    bad_version[4] = 9;
    expect_rejected("version", bad_version);

    auto wrong_embd = good;
    wrong_embd[16] = 33;  // n_embd 32 -> 33: shapes no longer match
    expect_rejected("n_embd", wrong_embd);

    expect_rejected("truncated_header", std::vector<char>(good.begin(), good.begin() + 40));
    expect_rejected("truncated_data", std::vector<char>(good.begin(), good.end() - 100));

    bool threw = false;
    try {
        gpt2::Model::load("does_not_exist.bin");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "missing file was accepted");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tiny.bin>\n", argv[0]);
        return 2;
    }
    try {
        test_matches_reference(argv[1]);
        test_rejects_corruption(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_loader: all checks passed\n");
    return 0;
}
