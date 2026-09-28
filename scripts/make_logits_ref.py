#!/usr/bin/env python3
"""Write HF GPT-2 float32 reference log-probabilities for the C++ forward-pass tests.

The tests compare log-softmax, not raw logits: float32 logits carry rare per-row shifts of up to
~7e-3 (in HF itself, vs a float64 run) that softmax cancels, while log-probs from HF float32 stay
within ~1e-3 of float64 at every position.

    # 6 built-in prompts, every position (forward_hf, fast)
    python scripts/make_logits_ref.py --out models/logits_ref.bin
    # the Phase 3 prompt set, sampled positions (diff_hf, slow)
    python scripts/make_logits_ref.py --prompts tests/diff_prompts.jsonl --rows sampled \\
        --out models/diff_ref.bin

Format v3 (little-endian): b"LGTS", u32 version=3, u32 n_prompts, u32 vocab_size, u32 K, then
per prompt:
    u32 name_len, name bytes
    u32 T, i32 ids[T]
    i32 top1[T], f32 gap[T]            argmax and (top1 - top2) logit gap at EVERY position
    i32 topk_ids[T, K], f32 topk_lp[T, K]   HF's K most likely tokens and their log-probs,
                                            at EVERY position
    u32 n_rows, u32 rows[n_rows]       positions whose full log-prob row is stored
    f32 logprobs[n_rows, vocab_size]
Log-probs are computed in float64 from HF's float32 logits, then stored as float32.
"""
import argparse
import json
import struct

import numpy as np

K = 20  # top-K tokens stored at every position

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


def sampled_rows(ids, eot):
    """First 2, last 2, 8 evenly spaced, and the position right after each <|endoftext|>."""
    T = len(ids)
    if T <= 12:
        return list(range(T))
    rows = {0, 1, T - 2, T - 1}
    rows |= {round(i * (T - 1) / 9) for i in range(1, 9)}
    rows |= {t + 1 for t in range(T - 1) if ids[t] == eot}
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompts", help="jsonl of {name, ids}; default: the 6 built-in prompts")
    ap.add_argument("--rows", choices=["all", "sampled"], default="all")
    ap.add_argument("--out", default="models/logits_ref.bin")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    vocab = model.config.vocab_size

    if args.prompts:
        prompts = [(p["name"], p["ids"]) for p in map(json.loads, open(args.prompts))]
    else:
        prompts = [(f"builtin/{i}", tok.encode(t)) for i, t in enumerate(PROMPTS)]

    total_rows = 0
    with open(args.out, "wb") as f:
        f.write(b"LGTS" + struct.pack("<4I", 3, len(prompts), vocab, K))
        for name, ids in prompts:
            with torch.no_grad():
                logits = model(torch.tensor([ids])).logits[0].float().numpy()
            assert logits.shape == (len(ids), vocab)
            top2 = np.sort(np.partition(logits, -2, axis=1)[:, -2:], axis=1)
            lp = logits.astype(np.float64)
            lp -= lp.max(axis=1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
            topk = np.argsort(-lp, axis=1, kind="stable")[:, :K]
            rows = list(range(len(ids))) if args.rows == "all" else sampled_rows(ids, tok.eos_token_id)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<i4").tobytes())
            f.write(logits.argmax(axis=1).astype("<i4").tobytes())
            f.write((top2[:, 1] - top2[:, 0]).astype("<f4").tobytes())
            f.write(topk.astype("<i4").tobytes())
            f.write(np.take_along_axis(lp, topk, axis=1).astype("<f4").tobytes())
            f.write(struct.pack("<I", len(rows)) + np.asarray(rows, dtype="<u4").tobytes())
            f.write(np.ascontiguousarray(lp[rows], dtype="<f4").tobytes())
            total_rows += len(rows)
    print(f"wrote {args.out}: {len(prompts)} prompts, {sum(len(i) for _, i in prompts)} tokens, "
          f"{total_rows} full log-prob rows, top-{K} at every position")


if __name__ == "__main__":
    main()
