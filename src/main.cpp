// gpt2 CLI.
//   gpt2 inspect <model.bin>                  print the config and tensor table
//   gpt2 encode <models dir> <text | ->       print token ids ("-" reads stdin)
//   gpt2 decode <models dir> <id>...          print the text for token ids
//   gpt2 generate <models dir> <prompt | -> [options]   continue the prompt
//   gpt2 bench <models dir> [options]         time prefill and cached decoding
// Run without arguments for the options.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <sys/resource.h>

#include "forward.h"
#include "model.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer.h"

static int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s inspect <model.bin>\n"
                 "       %s encode <models dir> <text | ->\n"
                 "       %s decode <models dir> <id>...\n"
                 "       %s generate <models dir> <prompt | -> [options]\n"
                 "           -n N               tokens to generate (default 64)\n"
                 "           --temperature T    0 = greedy (default); > 0 samples\n"
                 "           --top-k K          keep the K most likely tokens (default 50; 0 = off)\n"
                 "           --top-p P          nucleus sampling mass (default 1 = off)\n"
                 "           --seed S           RNG seed (default: random, printed to stderr)\n"
                 "           --no-stop          keep going after <|endoftext|>\n"
                 "           --threads N        worker threads (default: GPT2_THREADS or P-cores)\n"
                 "       %s bench <models dir> [options]\n"
                 "           --prompt T         prompt length (default 64)\n"
                 "           --gen N            decode steps (default 32)\n"
                 "           --reps R           repetitions, best is reported (default 3)\n"
                 "           --threads N        worker threads\n",
                 argv0, argv0, argv0, argv0, argv0);
    return 2;
}

static gpt2::Tokenizer load_tokenizer(const std::string& dir) {
    return gpt2::Tokenizer::load(dir + "/vocab.json", dir + "/merges.txt");
}

template <typename T>
static T parse_number(const std::string& a, const char* what, T min) {
    size_t used = 0;
    T v{};
    try {
        if constexpr (std::is_floating_point_v<T>) {
            v = static_cast<T>(std::stod(a, &used));
        } else if constexpr (std::is_signed_v<T>) {
            const long long x = std::stoll(a, &used);
            v = static_cast<T>(x);
            if (x != static_cast<long long>(v)) used = 0;  // out of range for T
        } else {
            if (!a.empty() && a[0] == '-') throw std::invalid_argument("negative");
            v = static_cast<T>(std::stoull(a, &used));
        }
    } catch (const std::logic_error&) {
        used = 0;
    }
    if (used == 0 || used != a.size() || v < min)
        throw std::runtime_error(std::string("bad ") + what + ": " + a);
    return v;
}

// Walks "--flag value" options after the positional arguments.
struct Options {
    int argc;
    char** argv;
    int i;
    bool next(std::string& flag) {
        if (i >= argc) return false;
        flag = argv[i++];
        return true;
    }
    std::string value(const std::string& flag) {
        if (i >= argc) throw std::runtime_error(flag + " needs a value");
        return argv[i++];
    }
};

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static double peak_rss_mb() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return ru.ru_maxrss / 1e6;  // bytes
#else
    return ru.ru_maxrss / 1e3;  // kilobytes
#endif
}

static int generate(const std::string& dir, std::string prompt, Options opt) {
    int n = 64;
    bool stop_at_eot = true, seed_given = false;
    gpt2::SamplerConfig sc;
    for (std::string f; opt.next(f);) {
        if (f == "-n") n = parse_number<int>(opt.value(f), "-n", 1);
        else if (f == "--temperature") sc.temperature = parse_number<float>(opt.value(f), "temperature", 0);
        else if (f == "--top-k") sc.top_k = parse_number<int>(opt.value(f), "top-k", 0);
        else if (f == "--top-p") sc.top_p = parse_number<float>(opt.value(f), "top-p", 0);
        else if (f == "--seed") sc.seed = parse_number<uint64_t>(opt.value(f), "seed", 0), seed_given = true;
        else if (f == "--no-stop") stop_at_eot = false;
        else if (f == "--threads") gpt2::set_num_threads(parse_number<int>(opt.value(f), "threads", 1));
        else throw std::runtime_error("unknown option " + f);
    }
    if (!seed_given) sc.seed = std::random_device{}();
    if (prompt == "-") prompt.assign(std::istreambuf_iterator<char>(std::cin), {});

    const gpt2::Tokenizer tok = load_tokenizer(dir);
    const gpt2::Model m = gpt2::Model::load(dir + "/gpt2.bin");
    gpt2::Sampler sampler(sc);
    std::vector<int> ids = tok.encode(prompt);
    if (ids.empty()) ids.push_back(gpt2::Tokenizer::kEndOfText);
    const int V = m.config().vocab_size;
    const size_t n_prompt = ids.size();
    std::fwrite(prompt.data(), 1, prompt.size(), stdout);

    // Prefill the prompt once, then feed back one token at a time through the cache.
    gpt2::KVCache cache(m.config());
    auto t0 = std::chrono::steady_clock::now();
    int next = sampler.sample(gpt2::forward(m, cache, ids, gpt2::Logits::Last).data(), V);
    const double prefill_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        if (stop_at_eot && next == gpt2::Tokenizer::kEndOfText) break;
        ids.push_back(next);
        const std::string piece = tok.decode({next});
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        if (i + 1 == n || cache.pos == m.config().n_ctx) break;
        next = sampler.sample(gpt2::forward(m, cache, {next}, gpt2::Logits::Last).data(), V);
    }
    const double decode_ms = ms_since(t0);
    const size_t made = ids.size() - n_prompt;
    std::printf("\n");
    if (sc.temperature > 0)
        std::fprintf(stderr, "[sampling: temperature %g, top-k %d, top-p %g, seed %llu]\n",
                     sc.temperature, sc.top_k, sc.top_p, (unsigned long long)sc.seed);
    else
        std::fprintf(stderr, "[greedy]\n");
    std::fprintf(stderr,
                 "[%d threads; prefill %zu tokens %.0f ms; %zu generated tokens %.0f ms, %.2f tok/s]\n",
                 gpt2::pool().size(), n_prompt, prefill_ms, made, decode_ms,
                 made > 1 ? (made - 1) * 1000.0 / decode_ms : 0.0);
    return 0;
}

static int bench(const std::string& dir, Options opt) {
    int T = 64, gen = 32, reps = 3;
    for (std::string f; opt.next(f);) {
        if (f == "--prompt") T = parse_number<int>(opt.value(f), "prompt length", 1);
        else if (f == "--gen") gen = parse_number<int>(opt.value(f), "gen", 1);
        else if (f == "--reps") reps = parse_number<int>(opt.value(f), "reps", 1);
        else if (f == "--threads") gpt2::set_num_threads(parse_number<int>(opt.value(f), "threads", 1));
        else throw std::runtime_error("unknown option " + f);
    }
    const gpt2::Model m = gpt2::Model::load(dir + "/gpt2.bin");
    if (T + gen > m.config().n_ctx) throw std::runtime_error("bench: prompt + gen exceeds n_ctx");
    std::vector<int> ids(T);
    for (int t = 0; t < T; ++t) ids[t] = (t * 7919 + 13) % m.config().vocab_size;
    const auto argmax = [](const std::vector<float>& row) {
        return static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
    };

    // Best of `reps`: prefill with all logits (comparable with earlier phases), prefill with
    // only the last row's logits (time to first token), then `gen` cached decode steps.
    double prefill_all = 1e300, prefill_last = 1e300, decode = 1e300;
    for (int rep = 0; rep < reps; ++rep) {
        {
            gpt2::KVCache cache(m.config());
            const auto t0 = std::chrono::steady_clock::now();
            gpt2::forward(m, cache, ids, gpt2::Logits::All);
            prefill_all = std::min(prefill_all, ms_since(t0));
        }
        gpt2::KVCache cache(m.config());
        auto t0 = std::chrono::steady_clock::now();
        int next = argmax(gpt2::forward(m, cache, ids, gpt2::Logits::Last));
        prefill_last = std::min(prefill_last, ms_since(t0));
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < gen; ++i)
            next = argmax(gpt2::forward(m, cache, {next}, gpt2::Logits::Last));
        decode = std::min(decode, ms_since(t0));
    }
    std::printf("threads: %d, prompt: %d tokens, decode: %d tokens, best of %d\n",
                gpt2::pool().size(), T, gen, reps);
    std::printf("prefill T=%d (all logits): %.1f ms (%.1f tok/s)\n", T, prefill_all,
                T * 1000.0 / prefill_all);
    std::printf("time to first token (last logits only): %.1f ms\n", prefill_last);
    std::printf("decode %d tokens after T=%d: %.2f ms/token (%.1f tok/s)\n", gen, T, decode / gen,
                gen * 1000.0 / decode);
    std::printf("peak RSS: %.0f MB\n", peak_rss_mb());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);
    const std::string cmd = argv[1];
    try {
        if (cmd == "inspect" && argc == 3) {
            const gpt2::Model m = gpt2::Model::load(argv[2]);
            const gpt2::Config& c = m.config();
            std::printf("n_layer=%d n_head=%d n_embd=%d n_ctx=%d vocab_size=%d\n", c.n_layer,
                        c.n_head, c.n_embd, c.n_ctx, c.vocab_size);
            for (const auto& [name, t] : m.tensors())
                std::printf("  %-28s %s\n", name.c_str(), t.shape_str().c_str());
            std::printf("params: %zu (%.1fM)\n", m.num_params(), m.num_params() / 1e6);
        } else if (cmd == "encode" && argc == 4) {
            std::string text = argv[3];
            if (text == "-") text.assign(std::istreambuf_iterator<char>(std::cin), {});
            const std::vector<int> ids = load_tokenizer(argv[2]).encode(text);
            for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
            std::printf("\n");
        } else if (cmd == "decode" && argc >= 4) {
            std::vector<int> ids;
            for (int i = 3; i < argc; ++i) ids.push_back(parse_number<int>(argv[i], "token id", 0));
            const std::string text = load_tokenizer(argv[2]).decode(ids);
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::printf("\n");
        } else if (cmd == "generate" && argc >= 4) {
            return generate(argv[2], argv[3], Options{argc, argv, 4});
        } else if (cmd == "bench") {
            return bench(argv[2], Options{argc, argv, 3});
        } else {
            return usage(argv[0]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
