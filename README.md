# gpt2-cpp

GPT-2 inference in C++17 with no ML-framework dependencies.

## Status

- [x] Phase 1a: weight export, checkpoint format, `Tensor`, loader, `gpt2 inspect`
- [ ] Phase 1b: byte-level BPE tokenizer (must match HF token ids)
- [ ] Phase 2: naive forward pass
- [ ] Phase 3: differential tests against HF logits (1e-4, ~100 prompts)
- [ ] Phase 4: KV cache
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
```

Debug with sanitizers: `cmake -S . -B build-asan -DGPT2_SANITIZE=ON`.

## Layout

```
scripts/gpt2bin.py              checkpoint format spec + Python reader/writer
scripts/export_gpt2.py          HF -> models/gpt2.bin, vocab.json, merges.txt
scripts/make_tiny_checkpoint.py random 2-layer model + reference stats for tests
scripts/check_real_weights.py   exported gpt2.bin vs HF state_dict, exact comparison
src/tensor.h                    non-owning float32 tensor view
src/model.{h,cpp}               Config, Weights, Model::load
src/main.cpp                    CLI
tests/test_loader.cpp           loader vs reference stats, corrupt-file rejection
```

## Checkpoint format

Little-endian. Header: `"GPT2"`, version, n_layer, n_head, n_embd, n_ctx, vocab_size,
n_tensors, then one metadata record per tensor (name, dims, u64 data offset). Tensor data
is float32 and each tensor starts on a 64-byte boundary, which suits SIMD loads later.
Tensors appear in a fixed order and the loader checks every name and shape against the config.

Linear weights keep HF's Conv1D layout `[in, out]`, so `y = x @ W + b`. The LM head is tied
to `wte`, so it isn't stored separately.
