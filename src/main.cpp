// gpt2 CLI. Phase 1: `gpt2 inspect <model.bin>` prints the config and tensor table.

#include <cstdio>
#include <exception>
#include <string>

#include "model.h"

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "inspect") {
        std::fprintf(stderr, "usage: %s inspect <model.bin>\n", argv[0]);
        return 2;
    }
    try {
        const gpt2::Model m = gpt2::Model::load(argv[2]);
        const gpt2::Config& c = m.config();
        std::printf("n_layer=%d n_head=%d n_embd=%d n_ctx=%d vocab_size=%d\n", c.n_layer, c.n_head,
                    c.n_embd, c.n_ctx, c.vocab_size);
        for (const auto& [name, t] : m.tensors())
            std::printf("  %-28s %s\n", name.c_str(), t.shape_str().c_str());
        std::printf("params: %zu (%.1fM)\n", m.num_params(), m.num_params() / 1e6);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
