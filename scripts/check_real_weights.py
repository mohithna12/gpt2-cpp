#!/usr/bin/env python3
"""Check an exported checkpoint against the Hugging Face weights, tensor by tensor.

Reads the file with gpt2bin.read() (the reference reader for the format the C++ loader
parses) and requires every tensor to be bit-identical to the HF state_dict.

    python scripts/check_real_weights.py --model gpt2 --bin models/gpt2.bin
"""
import argparse
import sys

import numpy as np

import gpt2bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--bin", default="models/gpt2.bin")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(args.model)
    cfg = model.config
    want_cfg = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                    n_ctx=cfg.n_positions, vocab_size=cfg.vocab_size)

    ref = {}
    for k, v in model.state_dict().items():
        if not k.startswith("transformer."):
            continue  # lm_head.weight is tied to wte.weight
        name = k[len("transformer."):]
        if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
            continue  # causal-mask buffers, not weights
        ref[name] = v.detach().float().cpu().numpy()

    got_cfg, got = gpt2bin.read(args.bin)
    errors = []
    if got_cfg != want_cfg:
        errors.append(f"config {got_cfg} != {want_cfg}")
    if set(got) != set(ref):
        errors.append(f"tensor names differ: only in file={sorted(set(got) - set(ref))[:5]} "
                      f"only in HF={sorted(set(ref) - set(got))[:5]}")

    n_params = 0
    for name in sorted(set(got) & set(ref)):
        a, b = got[name], ref[name]
        if a.shape != b.shape:
            errors.append(f"{name}: shape {a.shape} != {b.shape}")
        elif not np.array_equal(a, b):
            diff = np.abs(a.astype(np.float64) - b).max()
            errors.append(f"{name}: values differ (max abs diff {diff:.3g})")
        n_params += a.size

    if not np.array_equal(model.lm_head.weight.detach().numpy(), got["wte.weight"]):
        errors.append("lm_head.weight is not tied to wte.weight")

    if errors:
        print("FAIL")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print(f"OK: {len(got)} tensors, {n_params} params, all bit-identical to HF '{args.model}' "
          f"(lm_head tied to wte)")


if __name__ == "__main__":
    main()
