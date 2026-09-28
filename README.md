# gpt2-cpp

GPT-2 inference in C++17 with no ML-framework dependencies.

## Status

- [x] Phase 1a: weight export, checkpoint format, `Tensor`, loader, `gpt2 inspect`
- [x] Phase 1b: byte-level BPE tokenizer (must match HF token ids)
- [x] Phase 2: naive forward pass
- [x] Phase 3: differential tests against HF (94 prompts up to n_ctx; log-probs within 2e-3, argmax equal)
- [x] Phase 4: KV cache
- [ ] Phase 5: blocked / SIMD GEMM
- [ ] Phase 6: multithreading
- [ ] Phase 7: CLI sampling, benchmarks, results table

## Quick start

```bash
# Python env for the exporter and test fixtures (CMake prefers .venv/bin/python)
python3 -m venv .venv && .venv/bin/pip install numpy torch transformers
export PATH="$PWD/.venv/bin:$PATH"

# Build and run tests (tests use a tiny random checkpoint, no download needed)
cmake -S . -B build
cmake --build build -j
cd build && ctest --output-on-failure

# Export real weights (run once), then verify them
python scripts/export_gpt2.py --model gpt2 --out models
./build/gpt2 inspect models/gpt2.bin     # expect params: 124439808
python scripts/check_real_weights.py     # every tensor bit-identical to HF

./build/gpt2 encode models "Hello world"  # 15496 995
./build/gpt2 decode models 15496 995      # Hello world

python scripts/make_logits_ref.py         # HF reference for forward_hf (6 prompts, fast)
python scripts/make_logits_ref.py --prompts tests/diff_prompts.jsonl --rows sampled \
    --out models/diff_ref.bin             # HF reference for diff_hf (94 prompts, ~9 min now)
./build/gpt2 generate models "The meaning of life is" 20   # greedy
./build/gpt2 bench models 64              # time one forward pass
```

The `tokenizer`, `forward_hf` and `diff_hf` tests need the generated files in `models/`; they
are skipped if those are missing. `diff_hf` is labelled `slow`: `ctest -LE slow` skips it (do
that under ASan, where it would take hours at current speed).

Debug with sanitizers: `cmake -S . -B build-asan -DGPT2_SANITIZE=ON`.

## Layout

```
scripts/gpt2bin.py              checkpoint format spec + Python reader/writer
scripts/export_gpt2.py          HF -> models/gpt2.bin, vocab.json, merges.txt
scripts/make_tiny_checkpoint.py random 2-layer model + reference stats for tests
scripts/check_real_weights.py   exported gpt2.bin vs HF state_dict, exact comparison
scripts/gen_unicode_tables.py   probe HF's pre-tokenizer -> src/unicode_tables.h
scripts/make_tokenizer_fixtures.py  HF token ids -> tests/tokenizer_cases.tsv
scripts/make_diff_prompts.py    builds tests/diff_prompts.jsonl (94 prompts, token ids)
scripts/make_logits_ref.py      HF log-probs -> models/logits_ref.bin, models/diff_ref.bin
src/tensor.h                    non-owning float32 tensor view
src/model.{h,cpp}               Config, Weights, Model::load
src/tokenizer.{h,cpp}           byte-level BPE: pre-tokenizer, merges, encode/decode
src/unicode_tables.h            generated letter/number/space code point ranges
src/ops.{h,cpp}                 layernorm, matmul, GELU, causal attention (plain loops)
src/forward.{h,cpp}             forward pass with KV cache: new tokens -> logits
src/main.cpp                    CLI: inspect, encode, decode, generate, bench
tests/test_loader.cpp           loader vs reference stats, corrupt-file rejection
tests/test_tokenizer.cpp        5098 HF-generated cases, invalid UTF-8 round trips, bad files
tests/test_forward.cpp          tiny-model invariants (causality, KV cache = full pass bit for
                                bit); vs HF, both full prefill and token-by-token decode
tests/diff_prompts.jsonl        differential-test prompt set (committed)
```

## Correctness

The differential test compares **log-probabilities** (log-softmax of the logits) with HF float32:
max |diff| ≤ 2e-3, and the argmax must equal HF's at every position. Two ctest entries use it:

- `forward_hf`: 6 prompts, every position, full vocabulary. `forward_hf_decode` runs the same
  prompts one token at a time through the KV cache.
- `diff_hf`: 94 prompts, 10,070 tokens (`tests/diff_prompts.jsonl`): short text, code,
  multilingual/emoji, repetition, random token ids, prose, and long contexts of 512, 768 and
  1024 tokens, including a full-context prompt with `<|endoftext|>` in the middle. It checks
  argmax and HF's top-20 log-probs at every position, and the full vocabulary at 961 sampled
  positions.

**Why log-probs within 2e-3 and not raw logits within 1e-4 (the original target):** GPT-2 logits
reach ~300. Float32 logits, HF's own included, carry rare whole-row shifts vs a float64 run: up
to 6.8e-3 for HF on a 512-token prompt, and 8e-3 between C++ and HF on 4 of 200 rows of a prose
prompt. Softmax cancels a row shift, so sampling and perplexity never see it. Log-probs are
stable: HF float32 stays within ~9e-4 of float64 at every position, flat with length.

| Measurement (all positions, full vocabulary) | Worst |
|---|---|
| C++ vs HF float32, log-probs, worst prompts incl. 1024 tokens | 7.4e-4 |
| HF float32 vs float64, log-probs | 9.7e-4 |
| `diff_hf` as checked (top-20 everywhere + sampled full rows) | 2.8e-4 |
| Argmax mismatches, 10,070 positions | 0 |

Calibration with injected bugs (a bug counts as caught if any prompt fails):

| Injected bug | forward_hf (6 prompts) | diff_hf (94 prompts) |
|---|---|---|
| Exact-erf GELU instead of tanh | 6/6 fail, 12–99× tolerance | |
| LayerNorm eps 1e-4 | 5/6 fail, up to 35× | |
| Attention scale ×1.001 | 5/6 fail, up to 25× | |
| LayerNorm eps 1e-6 | 3/6 fail, up to 3.5× | 42/94 fail, up to 14× |

Greedy generation matches HF's `generate(do_sample=False)` token for token.

## Tokenizer

Matches HF's GPT-2 tokenizer id for id: literal `<|endoftext|>` becomes 50256, no BOS/EOS is
added. The pre-tokenizer regex is implemented as a hand-written scanner. Its character classes
(`\p{L}`, `\p{N}`, `\s`) come from `src/unicode_tables.h`, which is generated by probing HF's
own pre-tokenizer with every code point. That table differs from Python's `unicodedata` on
9,396 code points (e.g. U+001C–U+001F aren't whitespace there; newer letters are).
Invalid UTF-8, which HF can't receive, is tokenized byte by byte and decodes back exactly.

## Benchmarks

Apple M2 Pro, Release build, single thread.

| What | Input | Time | Notes |
|---|---|---|---|
| `gpt2 encode` (naive BPE, no cache) | 1.86 MB, 768,200 tokens | 0.70 s end to end | includes ~0.02 s vocab load; ids identical to HF |
| HF fast tokenizer `encode` (reference) | same | 1.11 s | Python call, for scale |
| Forward, naive loops (phase 2) | T=1 | 56.7 ms | 17.6 tok/s |
| Forward, naive loops (phase 2) | T=64 | 3186 ms | 20.1 tok/s, ~5 GFLOP/s |
| Forward, naive loops (phase 2) | T=256 | 12869 ms | 19.9 tok/s |
| Greedy generate, no KV cache (phase 2) | 5 + 20 tokens | 14.6 s | 1.37 tok/s |
| Greedy generate, no KV cache (phase 3 code) | 64 + 20 tokens | 74.9 s | 0.27 tok/s |
| Greedy generate, KV cache (phase 4) | 5 + 20 tokens | 0.23 s prefill + 1.26 s | 15.0 tok/s |
| Greedy generate, KV cache (phase 4) | 64 + 20 tokens | 0.74 s prefill + 1.16 s | 16.4 tok/s |
| Decode step, KV cache (phase 4) | after T=1 / 64 / 256 / 900 | 56.5 / 59.5 / 61.6 / 72.7 ms | 17.7 → 13.8 tok/s |
| Prefill, all logits (phase 4) | T=64 / 256 | 3233 / 13257 ms | unchanged from phase 2 |

Prefill and decode numbers are `gpt2 bench models T`, best of 3. Generate numbers are from
`gpt2 generate`. Prefill with all logits is dominated by the LM head: `matmul_transposed` is a
dot-product reduction the compiler won't vectorize without fast-math. With only the last row's
logits, a 64-token prefill takes 0.74 s instead of 3.2 s. That's the first target for phase 5.
