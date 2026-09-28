// gpt2 CLI.
//   gpt2 inspect <model.bin>          print the config and tensor table
//   gpt2 encode <models dir> <text>   print token ids ("-" reads the text from stdin)
//   gpt2 decode <models dir> <id>...  print the text for token ids
//   gpt2 generate <models dir> <prompt> [n]   greedy continuation, n tokens (default 20)
//   gpt2 bench <models dir> [T]       time prefill of T tokens (default 64), then decoding
//                                     32 more through the KV cache

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "forward.h"
#include "model.h"
#include "tokenizer.h"

static int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s inspect <model.bin>\n"
                 "       %s encode <models dir> <text | ->\n"
                 "       %s decode <models dir> <id>...\n"
                 "       %s generate <models dir> <prompt> [n]\n"
                 "       %s bench <models dir> [T]\n",
                 argv0, argv0, argv0, argv0, argv0);
    return 2;
}

static gpt2::Tokenizer load_tokenizer(const std::string& dir) {
    return gpt2::Tokenizer::load(dir + "/vocab.json", dir + "/merges.txt");
}

static int parse_count(const char* s, const char* what) {
    const std::string a = s;
    size_t used = 0;
    int n = -1;
    try {
        n = std::stoi(a, &used);
    } catch (const std::logic_error&) {
    }
    if (n < 1 || used != a.size()) throw std::runtime_error(std::string("bad ") + what + ": " + a);
    return n;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static int argmax(const std::vector<float>& row) {
    return static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
}

int main(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);
    const std::string cmd = argv[1];
    try {
        if (cmd == "inspect") {
            const gpt2::Model m = gpt2::Model::load(argv[2]);
            const gpt2::Config& c = m.config();
            std::printf("n_layer=%d n_head=%d n_embd=%d n_ctx=%d vocab_size=%d\n", c.n_layer,
                        c.n_head, c.n_embd, c.n_ctx, c.vocab_size);
            for (const auto& [name, t] : m.tensors())
                std::printf("  %-28s %s\n", name.c_str(), t.shape_str().c_str());
            std::printf("params: %zu (%.1fM)\n", m.num_params(), m.num_params() / 1e6);
        } else if (cmd == "encode" && argc == 4) {
            std::string text = argv[3];
            if (text == "-")
                text.assign(std::istreambuf_iterator<char>(std::cin), {});
            const std::vector<int> ids = load_tokenizer(argv[2]).encode(text);
            for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
            std::printf("\n");
        } else if (cmd == "decode" && argc >= 4) {
            std::vector<int> ids;
            for (int i = 3; i < argc; ++i) {
                const std::string a = argv[i];
                size_t used = 0;
                int id = -1;
                try {
                    id = std::stoi(a, &used);
                } catch (const std::logic_error&) {
                }
                if (id < 0 || used != a.size()) throw std::runtime_error("not a token id: " + a);
                ids.push_back(id);
            }
            const std::string text = load_tokenizer(argv[2]).decode(ids);
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::printf("\n");
        } else if (cmd == "generate" && (argc == 4 || argc == 5)) {
            const std::string dir = argv[2];
            const int n = argc == 5 ? parse_count(argv[4], "token count") : 20;
            const gpt2::Tokenizer tok = load_tokenizer(dir);
            const gpt2::Model m = gpt2::Model::load(dir + "/gpt2.bin");
            std::vector<int> ids = tok.encode(argv[3]);
            if (ids.empty()) ids.push_back(gpt2::Tokenizer::kEndOfText);
            const size_t n_prompt = ids.size();
            std::fwrite(argv[3], 1, std::strlen(argv[3]), stdout);
            gpt2::KVCache cache(m.config());
            // Prefill the prompt once, then feed back one token at a time through the cache.
            auto t0 = std::chrono::steady_clock::now();
            int next = argmax(gpt2::forward(m, cache, ids, gpt2::Logits::Last));
            const double prefill_ms = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) {
                ids.push_back(next);
                const std::string piece = tok.decode({next});
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
                if (i + 1 == n || cache.pos == m.config().n_ctx) break;
                next = argmax(gpt2::forward(m, cache, {next}, gpt2::Logits::Last));
            }
            const double decode_ms = ms_since(t0);
            const size_t made = ids.size() - n_prompt;
            std::printf("\n");
            std::fprintf(stderr,
                         "[prefill %zu tokens %.0f ms; %zu generated tokens %.0f ms, %.2f tok/s]\n",
                         n_prompt, prefill_ms, made, decode_ms,
                         made > 1 ? (made - 1) * 1000.0 / decode_ms : 0.0);
        } else if (cmd == "bench" && (argc == 3 || argc == 4)) {
            const std::string dir = argv[2];
            const gpt2::Model m = gpt2::Model::load(dir + "/gpt2.bin");
            const int T = argc == 4 ? parse_count(argv[3], "sequence length") : 64;
            constexpr int kDecode = 32;
            if (T + kDecode > m.config().n_ctx)
                throw std::runtime_error("bench: T + " + std::to_string(kDecode) + " exceeds n_ctx");
            std::vector<int> ids(T);
            for (int t = 0; t < T; ++t) ids[t] = (t * 7919 + 13) % m.config().vocab_size;
            // Best of 3: prefill with logits for every position (comparable with earlier
            // phases), then kDecode single-token steps through the cache.
            double prefill = 1e300, decode = 1e300;
            for (int rep = 0; rep < 3; ++rep) {
                gpt2::KVCache cache(m.config());
                auto t0 = std::chrono::steady_clock::now();
                int next = argmax(gpt2::forward(m, cache, ids, gpt2::Logits::All));
                prefill = std::min(prefill, ms_since(t0));
                t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kDecode; ++i)
                    next = argmax(gpt2::forward(m, cache, {next}, gpt2::Logits::Last));
                decode = std::min(decode, ms_since(t0));
            }
            std::printf("prefill T=%d (all logits): %.1f ms (%.1f tok/s)\n", T, prefill,
                        T * 1000.0 / prefill);
            std::printf("decode %d tokens after T=%d: %.1f ms/token (%.1f tok/s)\n", kDecode, T,
                        decode / kDecode, kDecode * 1000.0 / decode);
        } else {
            return usage(argv[0]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
