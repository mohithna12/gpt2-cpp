// gpt2 CLI.
//   gpt2 inspect <model.bin>          print the config and tensor table
//   gpt2 encode <models dir> <text>   print token ids ("-" reads the text from stdin)
//   gpt2 decode <models dir> <id>...  print the text for token ids

#include <cstdio>
#include <exception>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "tokenizer.h"

static int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s inspect <model.bin>\n"
                 "       %s encode <models dir> <text | ->\n"
                 "       %s decode <models dir> <id>...\n",
                 argv0, argv0, argv0);
    return 2;
}

static gpt2::Tokenizer load_tokenizer(const std::string& dir) {
    return gpt2::Tokenizer::load(dir + "/vocab.json", dir + "/merges.txt");
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
        } else {
            return usage(argv[0]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
