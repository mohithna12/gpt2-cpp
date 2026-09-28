#!/usr/bin/env python3
"""Export Hugging Face GPT-2 weights and tokenizer files for the C++ engine.

Run this on your own machine (needs internet the first time):

    pip install torch transformers numpy
    python scripts/export_gpt2.py --model gpt2 --out models/

Writes:
    models/gpt2.bin          weights in the format described in gpt2bin.py
    models/vocab.json        token string -> id (for the tokenizer, phase 1b)
    models/merges.txt        BPE merge ranks (for the tokenizer, phase 1b)
"""
import argparse
import os

import gpt2bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    os.makedirs(args.out, exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    cfg = model.config
    config = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                  n_ctx=cfg.n_positions, vocab_size=cfg.vocab_size)

    tensors = {}
    for k, v in model.state_dict().items():
        if not k.startswith("transformer."):
            continue  # lm_head.weight is tied to wte.weight; we reuse wte in C++
        name = k[len("transformer."):]
        if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
            continue  # causal-mask buffers, not weights
        tensors[name] = v.detach().float().cpu().numpy()

    path = os.path.join(args.out, f"{args.model}.bin")
    gpt2bin.write(path, config, tensors)
    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)  config={config}")

    GPT2Tokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"wrote tokenizer files to {args.out}/")


if __name__ == "__main__":
    main()
