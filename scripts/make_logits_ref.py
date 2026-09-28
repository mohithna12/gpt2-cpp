#!/usr/bin/env python3
"""Write HF GPT-2 logits for a few prompts, every position, for the C++ forward-pass test.

    python scripts/make_logits_ref.py --model gpt2 --out models/logits_ref.bin

Format (little-endian): b"LGTS", u32 version=1, u32 n_prompts, u32 vocab_size, then per
prompt: u32 T, i32 ids[T], f32 logits[T, vocab_size].
"""
import argparse
import struct

import numpy as np

PROMPTS = [
    "<|endoftext|>",
    "Hello, my name is",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "In 1905, Albert Einstein published four papers that changed physics: on the photoelectric "
    "effect, Brownian motion, special relativity, and the equivalence of mass and energy. "
    "Each of them would have been enough to make his reputation. Taken together, they turned a "
    "26-year-old patent clerk in Bern into one of the most important scientists of the century, "
    "although it took years before the physics community accepted all of his conclusions.",
    "日本語のテキストと emoji 🙂 and numbers 3.14159 mixed together!!",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="models/logits_ref.bin")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    vocab = model.config.vocab_size

    with open(args.out, "wb") as f:
        f.write(b"LGTS" + struct.pack("<3I", 1, len(PROMPTS), vocab))
        for text in PROMPTS:
            ids = tok.encode(text)
            with torch.no_grad():
                logits = model(torch.tensor([ids])).logits[0].float().numpy()
            assert logits.shape == (len(ids), vocab)
            f.write(struct.pack("<I", len(ids)))
            f.write(np.asarray(ids, dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(logits, dtype="<f4").tobytes())
            print(f"{len(ids):4d} tokens  {text[:50]!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
