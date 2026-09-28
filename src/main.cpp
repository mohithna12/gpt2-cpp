// gpt2 CLI.
//   gpt2 inspect <model.bin>          print the config and tensor table
//   gpt2 encode <models dir> <text>   print token ids ("-" reads the text from stdin)
//   gpt2 decode <models dir> <id>...  print the text for token ids
//   gpt2 generate <models dir> <prompt> [n]   greedy continuation, n tokens (default 20)
//   gpt2 bench <models dir> [T]       time one forward pass over T tokens (default 64)

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

// Index of the largest logit in the last row.
static int argmax_last(const std::vector<float>& logits, size_t T, size_t V) {
    const float* row = logits.data() + (T - 1) * V;
    return static_cast<int>(std::max_element(row, row + V) - row);
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
            const size_t V = m.config().vocab_size, n_prompt = ids.size();
            std::fwrite(argv[3], 1, std::strlen(argv[3]), stdout);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n && static_cast<int>(ids.size()) < m.config().n_ctx; ++i) {
                // No KV cache yet: the whole sequence is recomputed for every new token.
                ids.push_back(argmax_last(gpt2::forward(m, ids), ids.size(), V));
                const std::string piece = tok.decode({ids.back()});
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
            }
            const double ms = ms_since(t0);
            const size_t made = ids.size() - n_prompt;
            std::printf("\n");
            std::fprintf(stderr, "[%zu prompt + %zu generated tokens, %.0f ms, %.2f tok/s]\n",
                         n_prompt, made, ms, made * 1000.0 / ms);
        } else if (cmd == "bench" && (argc == 3 || argc == 4)) {
            const std::string dir = argv[2];
            const gpt2::Model m = gpt2::Model::load(dir + "/gpt2.bin");
            const int T = argc == 4 ? parse_count(argv[3], "sequence length") : 64;
            std::vector<int> ids(T);
            for (int t = 0; t < T; ++t) ids[t] = (t * 7919 + 13) % m.config().vocab_size;
            double best = 1e300;
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                const std::vector<float> logits = gpt2::forward(m, ids);
                best = std::min(best, ms_since(t0));
            }
            std::printf("forward T=%d: best of 3 = %.1f ms (%.1f tok/s)\n", T, best,
                        T * 1000.0 / best);
        } else {
            return usage(argv[0]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
