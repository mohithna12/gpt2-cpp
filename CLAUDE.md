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
(cd build-asan && ctest --output-on-failure)

python scripts/export_gpt2.py --model gpt2 --out models   # gpt2.bin, vocab.json, merges.txt
./build/gpt2 inspect models/gpt2.bin                      # params: 124439808
python scripts/check_real_weights.py                      # gpt2.bin vs HF, bit-exact
python scripts/make_logits_ref.py                         # HF logits for forward_hf test
./build/gpt2 encode models "text"   |   ./build/gpt2 decode models 15496 995
./build/gpt2 generate models "prompt" 20                  # greedy
./build/gpt2 bench models 64                              # forward timing
```

`loader` and `forward_tiny` use a tiny random checkpoint (`scripts/make_tiny_checkpoint.py`).
`tokenizer` needs `models/vocab.json` + `merges.txt`; `forward_hf` needs `models/gpt2.bin` +
`logits_ref.bin`. Both are skipped (exit 77) without them. `forward_hf` takes ~75 s under ASan.
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
- Tokenizer ground truth is HF's Rust tokenizer, not Python `unicodedata`.
  `src/unicode_tables.h` and `tests/tokenizer_cases.tsv` are generated and committed; regenerate
  with `scripts/gen_unicode_tables.py` / `scripts/make_tokenizer_fixtures.py`, never hand-edit.

## The rule for optimizations

Every optimization must (1) keep the differential test passing (`forward_hf`: max |logit − HF
float32| ≤ 2e-3 and argmax equal at every position) and (2) be benchmarked before and after
(`./build/gpt2 bench models [T]`). Record the numbers in the README.

The tolerance was 1e-4 originally; it was changed to 2e-3 on 2026-09-28 with the user's
approval because 1e-4 is below float32 noise: HF's own float32 logits are up to 6.5e-4 from the
float64 truth. Don't loosen it again without asking. If it gets tight, report the numbers.
Calibration: correct code 7.9e-4 worst; exact-erf GELU 7.5e-2+; LayerNorm eps 1e-6 fails 5/6.

## Roadmap

- [x] 1a: weight export, checkpoint format, `Tensor`, loader, `gpt2 inspect`
- [x] 1b: byte-level BPE tokenizer (must match HF token ids)
- [x] 2: naive forward pass (`src/ops.cpp`, `src/forward.cpp`)
- [ ] **3: differential tests against HF logits (2e-3 abs + argmax, ~100 prompts): CURRENT**
- [ ] 4: KV cache
- [ ] 5: blocked / SIMD GEMM (NEON on this machine)
- [ ] 6: multithreading
- [ ] 7: CLI sampling, benchmarks, results table
