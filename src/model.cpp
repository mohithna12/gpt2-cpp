#include "model.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpt2 {

std::string Tensor::shape_str() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
    os << "]";
    return os.str();
}

namespace {

constexpr uint32_t kVersion = 1;
constexpr size_t kAlign = 64;

// 64-byte aligned allocation, released with std::free. posix_memalign rather than
// std::aligned_alloc: the latter only exists on macOS 10.15+, the former everywhere POSIX.
void* aligned_malloc(size_t bytes) {
    void* p = nullptr;
    return posix_memalign(&p, kAlign, bytes) == 0 ? p : nullptr;
}

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error(path + ": " + msg);
}

// Bounds-checked little-endian reader over the metadata region.
struct Cursor {
    const unsigned char* p;
    size_t size;
    size_t pos = 0;
    const std::string& path;

    void need(size_t n) {
        if (pos + n > size) fail(path, "truncated header");
    }
    uint32_t u32() {
        need(4);
        uint32_t v = uint32_t(p[pos]) | uint32_t(p[pos + 1]) << 8 | uint32_t(p[pos + 2]) << 16 |
                     uint32_t(p[pos + 3]) << 24;
        pos += 4;
        return v;
    }
    uint64_t u64() {
        uint64_t lo = u32();
        uint64_t hi = u32();
        return lo | hi << 32;
    }
    std::string str(size_t n) {
        need(n);
        std::string s(reinterpret_cast<const char*>(p + pos), n);
        pos += n;
        return s;
    }
};

struct Expected {
    std::string name;
    std::vector<int64_t> shape;
};

// Must match expected_shapes() in scripts/gpt2bin.py.
std::vector<Expected> expected_tensors(const Config& c) {
    const int64_t C = c.n_embd;
    std::vector<Expected> e = {{"wte.weight", {c.vocab_size, C}}, {"wpe.weight", {c.n_ctx, C}}};
    for (int i = 0; i < c.n_layer; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        e.push_back({p + "ln_1.weight", {C}});
        e.push_back({p + "ln_1.bias", {C}});
        e.push_back({p + "attn.c_attn.weight", {C, 3 * C}});
        e.push_back({p + "attn.c_attn.bias", {3 * C}});
        e.push_back({p + "attn.c_proj.weight", {C, C}});
        e.push_back({p + "attn.c_proj.bias", {C}});
        e.push_back({p + "ln_2.weight", {C}});
        e.push_back({p + "ln_2.bias", {C}});
        e.push_back({p + "mlp.c_fc.weight", {C, 4 * C}});
        e.push_back({p + "mlp.c_fc.bias", {4 * C}});
        e.push_back({p + "mlp.c_proj.weight", {4 * C, C}});
        e.push_back({p + "mlp.c_proj.bias", {C}});
    }
    e.push_back({"ln_f.weight", {C}});
    e.push_back({"ln_f.bias", {C}});
    return e;
}

}  // namespace

Model Model::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail(path, "cannot open file");
    const size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0);

    Model m;
    // Tensors stay 64-byte aligned because the file places each one at a 64-byte offset.
    // Size is rounded up to a whole number of cache lines.
    m.buffer_bytes_ = (file_size + kAlign - 1) / kAlign * kAlign;
    void* raw = aligned_malloc(m.buffer_bytes_ ? m.buffer_bytes_ : kAlign);
    if (!raw) fail(path, "out of memory");
    m.buffer_ = std::shared_ptr<float>(static_cast<float*>(raw), [](float* q) { std::free(q); });
    auto* bytes = reinterpret_cast<unsigned char*>(raw);
    if (!f.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(file_size)))
        fail(path, "read failed");

    Cursor cur{bytes, file_size, 0, path};
    if (cur.str(4) != "GPT2") fail(path, "bad magic (not a gpt2.bin file)");
    const uint32_t version = cur.u32();
    if (version != kVersion) fail(path, "unsupported version " + std::to_string(version));

    Config& c = m.config_;
    c.n_layer = static_cast<int>(cur.u32());
    c.n_head = static_cast<int>(cur.u32());
    c.n_embd = static_cast<int>(cur.u32());
    c.n_ctx = static_cast<int>(cur.u32());
    c.vocab_size = static_cast<int>(cur.u32());
    if (c.n_layer <= 0 || c.n_head <= 0 || c.n_embd <= 0 || c.n_ctx <= 0 || c.vocab_size <= 0)
        fail(path, "config has a non-positive dimension");
    if (c.n_embd % c.n_head != 0) fail(path, "n_embd is not divisible by n_head");

    const uint32_t n_tensors = cur.u32();
    const auto expected = expected_tensors(c);
    if (n_tensors != expected.size())
        fail(path, "expected " + std::to_string(expected.size()) + " tensors, file has " +
                       std::to_string(n_tensors));

    for (const Expected& exp : expected) {
        const std::string name = cur.str(cur.u32());
        if (name != exp.name) fail(path, "expected tensor '" + exp.name + "', found '" + name + "'");
        Tensor t;
        const uint32_t ndim = cur.u32();
        for (uint32_t d = 0; d < ndim; ++d) t.shape.push_back(cur.u32());
        if (t.shape != exp.shape) fail(path, name + ": shape " + t.shape_str() + " is wrong");
        const uint64_t off = cur.u64();
        const uint64_t nbytes = static_cast<uint64_t>(t.numel()) * sizeof(float);
        if (off % kAlign != 0) fail(path, name + ": data is not 64-byte aligned");
        if (off < cur.pos || off + nbytes > file_size) fail(path, name + ": data out of bounds");
        t.data = reinterpret_cast<float*>(bytes + off);
        m.tensors_.emplace_back(name, t);
    }

    // Wire named tensors into the structured view.
    size_t i = 0;
    auto next = [&]() { return m.tensors_[i++].second; };
    Weights& w = m.weights_;
    w.wte = next();
    w.wpe = next();
    w.layers.resize(c.n_layer);
    for (LayerWeights& L : w.layers) {
        L.ln_1_w = next(); L.ln_1_b = next();
        L.attn_w = next(); L.attn_b = next();
        L.attn_proj_w = next(); L.attn_proj_b = next();
        L.ln_2_w = next(); L.ln_2_b = next();
        L.fc_w = next(); L.fc_b = next();
        L.fc_proj_w = next(); L.fc_proj_b = next();
    }
    w.ln_f_w = next();
    w.ln_f_b = next();
    return m;
}

size_t Model::num_params() const {
    size_t n = 0;
    for (const auto& kv : tensors_) n += static_cast<size_t>(kv.second.numel());
    return n;
}

}  // namespace gpt2
