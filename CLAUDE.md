# gpt2-cpp

GPT-2 inference in C++17, no ML-framework dependencies. See README.md for layout and the
checkpoint format (spec lives in `scripts/gpt2bin.py`).

## Machine

Apple M2 Pro, macOS 15, **arm64: SIMD target is NEON** (FEAT_DotProd, FP16, BF16, I8MM
available; no AVX). 10 cores (6 performance + 4 efficiency). Apple clang 16.

## Commands

Python venv: `.venv/` (numpy, torch CPU, transformers, cmake). CMake is installed in the
venv, and CMake picks `.venv/bin/python` automatically for test fixtures.

```bash
export PATH="$PWD/.venv/bin:$PATH"

cmake -S . -B build && cmake --build build -j          # Release (default)
(cd build && ctest --output-on-failure)

cmake -S . -B build-asan -DGPT2_SANITIZE=ON && cmake --build build-asan -j   # ASan+UBSan
(cd build-asan && ctest --output-on-failure)             # diff_hf ~8 min under ASan

python scripts/export_gpt2.py --model gpt2 --out models   # gpt2.bin, vocab.json, merges.txt
./build/gpt2 inspect models/gpt2.bin                      # params: 124439808
python scripts/check_real_weights.py                      # gpt2.bin vs HF, bit-exact
python scripts/make_logits_ref.py                         # models/logits_ref.bin (forward_hf)
python scripts/make_logits_ref.py --prompts tests/diff_prompts.jsonl --rows sampled \
    --out models/diff_ref.bin                             # diff_hf reference
./build/gpt2 encode models "text"   |   ./build/gpt2 decode models 15496 995
./build/gpt2 generate models "prompt" 20                  # greedy
./build/gpt2 bench models 64                              # prefill T=64 + 32 decode steps
```

`loader` and `forward_tiny` use a tiny random checkpoint (`scripts/make_tiny_checkpoint.py`).
`tokenizer` needs `models/vocab.json` + `merges.txt`; `forward_hf` needs `models/gpt2.bin` +
`logits_ref.bin`; `diff_hf` needs `models/diff_ref.bin`. All are skipped (exit 77) without them.
`diff_hf` (label `slow`) takes ~65 s in Release since phase 5.
`models/`, `build*/` and `.venv/` are gitignored.

## Conventions

- C++17, no external C++ dependencies. Build must stay clean under `-Wall -Wextra -Wpedantic`.
  Fix warnings at the root; don't suppress them.
- Linear weights use HF Conv1D layout `[in, out]`: `y = x @ W + b`.
- LM head is tied to `wte` (logits = h @ wteᵀ); it isn't stored separately.
- Every tensor is 64-byte aligned: the file places tensors at 64-byte offsets, and the loader
  allocates with `posix_memalign` (not `std::aligned_alloc`, which needs macOS 10.15+).
- Tensor order/shapes are defined twice and must match: `expected_shapes()` in
  `scripts/gpt2bin.py` and `expected_tensors()` in `src/model.cpp`.
- KV-cache decoding must equal the full pass bit for bit (`forward_tiny`). So every kernel
  computes each output element in the same order whatever the tile shape or T: `matmul` as one
  sequential FMA chain over k from the bias (4x16 tile, streaming row kernel, and scalar tails
  all agree), and dot products (LM head, attention) as 4-lane FMA partials + `vaddvq_f32` +
  a scalar `std::fma` tail. Keep that property in phase 6 (split work by rows/columns, never
  split one element's k-reduction across threads) or ask before relaxing the test.
- SIMD: NEON intrinsics under `GPT2_NEON` in `src/ops.cpp`; the scalar path stays as the
  portable fallback (`-DGPT2_NO_SIMD=ON` builds and tests it).
- Tokenizer ground truth is HF's Rust tokenizer, not Python `unicodedata`.
  `src/unicode_tables.h` and `tests/tokenizer_cases.tsv` are generated and committed; regenerate
  with `scripts/gen_unicode_tables.py` / `scripts/make_tokenizer_fixtures.py`, never hand-edit.

## The rule for optimizations

Every optimization must (1) keep the differential test passing and (2) be benchmarked before
and after (`./build/gpt2 bench models [T]`). Record the numbers in the README.

The differential test (`forward_hf` + `diff_hf`, `tests/test_forward.cpp`): log-probs within
2e-3 of HF float32, and argmax equal wherever HF's top1-top2 gap is >= 2e-3. At closer
near-ties (11 of 10,070 positions) a flip is allowed if our pick is in HF's top-20 (user-approved,
phase 5; the first case was a 4.7e-5 float64 gap at logit -95.6). `diff_hf` covers 94 prompts up to
n_ctx. Run it after any change to `ops`/`forward`, since `ctest -LE slow` skips it.

History, approved by the user on 2026-09-28: 1e-4 on raw logits was below float32 noise (HF's
own float32 logits are up to 6.8e-3 from float64 on long prompts, as rare whole-row shifts).
It went to 2e-3 on raw logits, then to 2e-3 on log-probs, which are stable (HF float32 within
~1e-3 of float64 everywhere). Worst today: 7.4e-4 at all positions. Don't loosen it without
asking. If it gets tight, report the numbers. The calibration table is in the README.

Expect the HF margin to sit near 40-50%, not near 10%: HF float32 is itself ~1e-3 from float64
in log-probs, so a MORE accurate kernel moves away from HF. Phase 5's NEON kernels are 2-5x
closer to float64 than the scalar code, yet 3.6x further from HF. When the margin moves, compare
against a float64 HF run before calling it a regression.

## Roadmap

- [x] 1a: weight export, checkpoint format, `Tensor`, loader, `gpt2 inspect`
- [x] 1b: byte-level BPE tokenizer (must match HF token ids)
- [x] 2: naive forward pass (`src/ops.cpp`, `src/forward.cpp`)
- [x] 3: differential tests vs HF (94 prompts to n_ctx, log-probs 2e-3, argmax)
- [x] 4: KV cache (`KVCache`, `forward(m, cache, tokens, Logits::All|Last)`)
- [x] 5: blocked / SIMD GEMM: NEON LM head, 4x16 matmul tiles, row kernel, attention
- [ ] **6: multithreading: CURRENT.** Decode is now bound by memory bandwidth (~500 MB of weights
  per token on one core), so threads should help decode as well as prefill.
- [ ] 7: CLI sampling, benchmarks, results table
