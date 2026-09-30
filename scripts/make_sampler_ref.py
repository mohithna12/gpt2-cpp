#!/usr/bin/env python3
"""Write HF's filtered sampling distributions for the C++ sampler test.

Runs HF's own warpers (TemperatureLogitsWarper -> TopKLogitsWarper -> TopPLogitsWarper, the
order generate() uses) on real GPT-2 next-token logits, for a spread of settings.

    python scripts/make_sampler_ref.py --out models/sampler_ref.bin

Format (little-endian): b"SMPL", u32 version=1, u32 n_rows, u32 vocab_size,
f32 logits[n_rows, vocab_size], u32 n_cases, then per case:
    u32 row, f32 temperature, i32 top_k (0 = off), f32 top_p (1 = off),
    u32 m, i32 ids[m], f64 probs[m]     tokens with nonzero probability, ascending id
"""
import argparse
import struct

import numpy as np

PROMPTS = ["The", "The capital of France is", "def fibonacci(n):\n   ", "I love"]
CONFIGS = [  # temperature, top_k, top_p
    (1.0, 0, 1.0), (0.7, 50, 1.0), (1.0, 50, 0.9), (1.3, 0, 0.95), (0.5, 10, 0.5),
    (1.0, 1, 1.0), (1.0, 0, 0.1), (2.0, 200, 0.99), (0.9, 40, 0.8), (1.0, 0, 0.999),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/sampler_ref.bin")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformers.generation.logits_process import (TemperatureLogitsWarper,
                                                        TopKLogitsWarper, TopPLogitsWarper)

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    rows = []
    with torch.no_grad():
        for p in PROMPTS:
            rows.append(model(torch.tensor([tok.encode(p)])).logits[0, -1].float())
    V = rows[0].numel()

    with open(args.out, "wb") as f:
        f.write(b"SMPL" + struct.pack("<3I", 1, len(rows), V))
        for r in rows:
            f.write(r.numpy().astype("<f4").tobytes())
        f.write(struct.pack("<I", len(rows) * len(CONFIGS)))
        for ri, r in enumerate(rows):
            for t, k, p in CONFIGS:
                scores = r[None].clone()
                scores = TemperatureLogitsWarper(t)(None, scores)
                if k > 0:
                    scores = TopKLogitsWarper(top_k=k)(None, scores)
                if p < 1.0:
                    scores = TopPLogitsWarper(top_p=p)(None, scores)
                probs = torch.softmax(scores.double(), dim=-1)[0].numpy()
                ids = np.nonzero(probs > 0)[0]
                f.write(struct.pack("<IfifI", ri, t, k, p, len(ids)))
                f.write(ids.astype("<i4").tobytes() + probs[ids].astype("<f8").tobytes())
    print(f"wrote {args.out}: {len(rows)} logit rows x {len(CONFIGS)} settings")


if __name__ == "__main__":
    main()
