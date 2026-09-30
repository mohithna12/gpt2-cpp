# gpt2-cpp

GPT-2 inference in C++17 with no ML-framework dependencies.

## Results

GPT-2 small (124M), Apple M2 Pro, 6 threads, measured interleaved with HF PyTorch
(`scripts/bench_hf.py`) on the same prompt ids and the same definitions:

| | This engine | HF PyTorch 2.14 (CPU) |
|---|---|---|
| Decode, after 64 / 512 tokens | **8.8 / 10.0 ms/token** | 41.5 / 30.6 ms/token (3.0–4.7× slower) |
| Greedy generation | **~100–120 tok/s** | |
| Prefill 64 / 512 tokens, all logits | 68 / 563 ms | **49 / 222 ms** (1.4–2.5× faster) |
| Time to first token, 64 / 512 tokens | 55 / 430 ms | **48 / 187 ms** |
| Peak memory | 600–730 MB | |

On one thread: decode 28.0 vs 36 ms/token (C++ 1.0–1.3× faster), prefill 364 vs 61 ms at T=64
(HF 6–7× faster). PyTorch on Apple Silicon uses Accelerate BLAS, which runs on the AMX matrix
coprocessor; plain NEON can't match it on large matmuls. HF's decode barely scales with threads
because its per-token dispatch overhead dominates.

Correctness, every commit: log-probs within 2e-3 of HF (worst 8.8e-4) and the same argmax at
all 10,070 positions of 94 prompts up to the full 1024-token context (one allowed near-tie).
Tokenizer ids match HF on 5,098 cases. Greedy output matches HF `generate` token for token. The
sampler reproduces HF's temperature/top-k/top-p distributions, and a seed gives identical text
for any thread count.

## Status

- [x] Phase 1a: weight export, checkpoint format, `Tensor`, loader, `gpt2 inspect`
- [x] Phase 1b: byte-level BPE tokenizer (must match HF token ids)
- [x] Phase 2: naive forward pass
- [x] Phase 3: differential tests against HF (94 prompts up to n_ctx; log-probs within 2e-3, argmax equal)
- [x] Phase 4: KV cache
- [x] Phase 5: blocked / SIMD GEMM (NEON)
- [x] Phase 6: multithreading (thread pool, 6 performance cores by default)
- [x] Phase 7: CLI sampling (temperature, top-k, top-p, seed), `bench`, results table

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
    --out models/diff_ref.bin             # HF reference for diff_hf (94 prompts)
python scripts/make_sampler_ref.py        # HF sampling distributions for sampler_hf

./build/gpt2 generate models "The meaning of life is" -n 20          # greedy
./build/gpt2 generate models "Once upon a time" -n 100 --temperature 0.8 --top-p 0.9 --seed 7
./build/gpt2 bench models --prompt 64 --gen 32                      # prefill, TTFT, decode, RSS
python scripts/bench_hf.py --prompt 64 --gen 32 --threads 6         # same numbers for PyTorch
```

`generate` is greedy by default. `--temperature T` > 0 samples with HF's defaults (top-k 50,
top-p 1), applied in HF's order: temperature, top-k, top-p. Without `--seed` a random seed is
used and printed. Generation stops at `<|endoftext|>` unless `--no-stop` is given. `-` reads the
prompt from stdin.

The `tokenizer`, `forward_hf`, `forward_threads`, `diff_hf` and `sampler_hf` tests need the
generated files in `models/`; they are skipped if those are missing. `diff_hf` is labelled
`slow` (~15 s in Release, ~8 min under ASan): `ctest -LE slow` skips it.

Debug with sanitizers: `cmake -S . -B build-asan -DGPT2_SANITIZE=ON`. On ARM the kernels use
NEON; `-DGPT2_NO_SIMD=ON` builds the portable scalar path instead (also tested). Race checks:
`cmake -S . -B build-tsan -DGPT2_TSAN=ON`. Thread count: `GPT2_THREADS=n` (default: the number
of performance cores, 6 on an M2 Pro). Results are bit-identical for any thread count.

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
scripts/make_sampler_ref.py     HF warper distributions -> models/sampler_ref.bin
scripts/bench_hf.py             PyTorch timings with the same definitions as `gpt2 bench`
src/tensor.h                    non-owning float32 tensor view
src/model.{h,cpp}               Config, Weights, Model::load
src/tokenizer.{h,cpp}           byte-level BPE: pre-tokenizer, merges, encode/decode
src/unicode_tables.h            generated letter/number/space code point ranges
src/ops.{h,cpp}                 layernorm, matmul, GELU, attention (NEON + scalar fallback)
src/forward.{h,cpp}             forward pass with KV cache: new tokens -> logits
src/threadpool.{h,cpp}          persistent pool, dynamic task handout, spin-then-sleep
src/sampler.{h,cpp}             greedy / temperature / top-k / top-p, HF semantics, seeded
src/main.cpp                    CLI: inspect, encode, decode, generate, bench
tests/test_loader.cpp           loader vs reference stats, corrupt-file rejection
tests/test_tokenizer.cpp        5098 HF-generated cases, invalid UTF-8 round trips, bad files
tests/test_forward.cpp          tiny-model invariants (causality, KV cache = full pass bit for
                                bit); 1..10 threads bit-identical; vs HF, prefill and decode
tests/diff_prompts.jsonl        differential-test prompt set (committed)
tests/test_sampler.cpp          sampler rules, seeds, draw frequencies; vs HF's warpers
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

The argmax must match wherever HF's top-1/top-2 gap is at least 2e-3. At closer near-ties (11 of
10,070 positions) a flip is allowed, and printed, if our pick is one of HF's top-20, whose
log-probs are checked anyway. The one flip today is a 4.7e-5 float64 gap at logit −95.6, where
the NEON build is within 8e-5 of float64 on both tokens but lands on the other side.

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
| Argmax mismatches, 10,070 positions | 0 (1 allowed near-tie flip since phase 5) |

Since phase 5 the worst `diff_hf` difference is 8.8e-4 (44%), up from 2.8e-4 with the scalar
code. That's the NEON kernels being *more* accurate: against float64 they're 2–5× closer than
the scalar code (e.g. 7.6e-4 → 2.8e-4), while the scalar LM head had happened to share HF's
rounding. Since HF float32 is itself ~1e-3 from float64, a margin around 40–50% is the expected
floor for an accurate implementation.

Calibration with injected bugs (a bug counts as caught if any prompt fails):

| Injected bug | forward_hf (6 prompts) | diff_hf (94 prompts) |
|---|---|---|
| Exact-erf GELU instead of tanh | 6/6 fail, 12–99× tolerance | |
| LayerNorm eps 1e-4 | 5/6 fail, up to 35× | |
| Attention scale ×1.001 | 5/6 fail, up to 25× | |
| LayerNorm eps 1e-6 | 3/6 fail, up to 3.5× | 42/94 fail, up to 14× |
| *Re-run on the phase 5 NEON kernels:* | | |
| Exact-erf GELU | | 94/94 fail, up to 204× |
| LayerNorm eps 1e-6 | | 44/94 fail, up to 14× |
| Attention scale ×1.001 | | 92/94 fail, up to 95× |

Greedy generation matches HF's `generate(do_sample=False)` token for token.

## Tokenizer

Matches HF's GPT-2 tokenizer id for id: literal `<|endoftext|>` becomes 50256, no BOS/EOS is
added. The pre-tokenizer regex is implemented as a hand-written scanner. Its character classes
(`\p{L}`, `\p{N}`, `\s`) come from `src/unicode_tables.h`, which is generated by probing HF's
own pre-tokenizer with every code point. That table differs from Python's `unicodedata` on
9,396 code points (e.g. U+001C–U+001F aren't whitespace there; newer letters are).
Invalid UTF-8, which HF can't receive, is tokenized byte by byte and decodes back exactly.

## Benchmarks

Apple M2 Pro (6 performance + 4 efficiency cores), Release build. `gpt2 bench models
--prompt T` prefills T tokens with all logits, then does 32 cached decode steps; best of 3 runs
of best-of-3. `gpt2 generate` gives greedy generation. This machine is also in normal use, and
run-to-run noise reaches 20–35%. Compare within a table, where rows were measured together;
across phases, prefer interleaved A/B runs of both builds (as in the phase 6 steps).

Phase 7: C++ vs HF PyTorch 2.14 (CPU, `torch.set_num_threads`), interleaved, best of 2 rounds
of best-of-3:

| | C++ 1 thread | HF 1 thread | C++ 6 threads | HF 6 threads |
|---|---|---|---|---|
| Prefill T=64, all logits | 364 ms | 61 ms | 68 ms | 49 ms |
| Prefill T=512, all logits | 3012 ms | 437 ms | 563 ms | 222 ms |
| Time to first token, T=64 / 512 | 279 / 2384 ms | 57 / 361 ms | 55 / 430 ms | 48 / 187 ms |
| Decode after T=64 | 28.0 ms/token | 35.9 | 8.8 | 41.5 |
| Decode after T=512 | 28.5 ms/token | 28.9 | 10.0 | 30.6 |

Sampling cost (`generate -n 100`, 6 threads, 3 runs): greedy 102–111 tok/s; temperature +
top-k 50 is the same (100–111). Top-p without top-k first sorted the whole 50k vocabulary per
token: 70–78 tok/s. Sorting only the top 1024 when they must contain the cut gives 95–99 tok/s,
with the same text for the same seed (A/B, 3 pairs).

Phase 6, current code (all rows measured together):

| | 1 thread | 6 threads (default) | 10 threads |
|---|---|---|---|
| Prefill T=1 / 64 / 256 / 900 | 17 / 267 / 1016 / 4250 ms | 8 / 58 / 212 / 874 ms | 13 / 61 / 200 / 789 ms |
| Decode step after T=1 / 64 / 256 / 900 | 17.3 / 18.0 / 19.1 / 26.6 ms | 8.2 / 8.4 / 8.9 / 10.7 ms | 13.0 / 12.6 / 12.5 / 15.7 ms |
| Greedy generate, 5 / 64-token prompt | | 108–121 / 101–116 tok/s | |

6 threads: prefill 4.6–4.9× the single-thread speed, decode 2.1× (2.5× at T=900). Decode is
bound by memory bandwidth and synchronization (~75 parallel jobs per token). 10 threads help
long prefills slightly but slow decode, since the efficiency cores lag.

Phase 6 steps (A/B = both builds run interleaved in the same session):

| Step | Result |
|---|---|
| Pool + 64-column chunks for every matmul | 6 threads: prefill T=256 1495 → 302 ms. But 1 thread decode 2.6× slower than phase 5: narrow chunks cut W rows into strided 256-byte pieces |
| Decode (T<4) gets one contiguous column range per thread | 6-thread decode 14.0 → 11.2 ms; 1-thread short-context decode back to phase-5 speed (A/B) |
| Attention body out of the task lambda (locals, raw scores pointer) | A/B at 1 thread, T=900: removed a consistent ~15% decode slowdown vs phase 5 |
| Spin ~100 µs before sleeping (workers and caller) | A/B at 6 threads: decode faster in 8 of 8 pairs (~10%, e.g. 11.7 → 9.0 ms); prefill unchanged |

Earlier phases (single thread; measured in their own sessions, so compare within a column):

| What | Phase 2 (naive) | Phase 4 (KV cache) | Phase 5 (NEON) |
|---|---|---|---|
| Prefill T=1, all logits | 56.7 ms | 53.1 ms | 19.6 ms |
| Prefill T=64, all logits | 3186 ms | 3233 ms | 363 ms (8.9×) |
| Prefill T=256, all logits | 12869 ms | 13257 ms | 1428 ms (9.3×) |
| Prefill T=900, all logits | — | 48841 ms | 5954 ms (8.2×) |
| Decode step after T=1 | — | 56.5 ms | 19.4–23.9 ms |
| Decode step after T=64 | — | 59.5 ms | 20.8 ms |
| Decode step after T=256 | — | 61.6 ms | 21.4 ms |
| Decode step after T=900 | — | 72.7 ms | 29.8–35.3 ms |
| Greedy generate, 5-token prompt | 1.32 tok/s | 15.0 tok/s | 41–46 tok/s |
| Greedy generate, 64-token prompt | 0.27 tok/s | 16.4 tok/s | 40–42 tok/s |

Phase 5 steps, each measured separately (prefill T=64 / decode after T=1):

| Step | Prefill T=64 | Decode | Notes |
|---|---|---|---|
| Before (phase 4) | 3233 ms | 56.5 ms | LM head was ~75% of prefill, ~70% of decode |
| 5a: NEON LM head, 4×4 dot tiles | 694 ms | 19.9 ms | |
| 5b: matmul 4×16 tiles + row kernel | 377 ms | 28.1 ms | decode regressed: row kernel reloaded the output every k |
| 5b': row kernel, 64-col register blocks | 370 ms | 91.3 ms | worse: strided W access defeats the prefetcher |
| 5b'': row kernel streaming W, 4 k per pass | 372 ms | 20.6 ms | kept |
| 5c: NEON attention (q·k, p·v) | 363 ms | 19.4 ms | T=900: prefill 7087 → 5959 ms, decode 34.0 → 29.8 ms |

Tokenizer: `gpt2 encode` handles 1.86 MB (768,200 tokens) in 0.70 s end to end, including
~0.02 s vocab load (naive BPE, no cache); ids identical to HF, whose fast tokenizer takes 1.11 s
on the same input.
