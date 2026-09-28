#!/usr/bin/env python3
"""Write a tiny random GPT-2-shaped checkpoint plus per-tensor reference stats.

Used by the C++ test so the loader can be checked without downloading real weights.

    python scripts/make_tiny_checkpoint.py build/tiny.bin
"""
import sys

import numpy as np

import gpt2bin


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny.bin"
    config = dict(n_layer=2, n_head=4, n_embd=32, n_ctx=64, vocab_size=101)
    rng = np.random.default_rng(0)
    shapes = gpt2bin.expected_shapes(config["n_layer"], config["n_embd"], config["n_ctx"], config["vocab_size"])
    tensors = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in shapes}
    gpt2bin.write(out, config, tensors)

    # Round-trip through the Python reader to make sure the writer is self-consistent.
    cfg2, back = gpt2bin.read(out)
    assert cfg2 == config
    for k, v in tensors.items():
        assert np.array_equal(back[k], v), k

    # Reference stats for the C++ test: name, element count, sum, first and last element.
    with open(out + ".ref", "w") as f:
        for name, _ in shapes:
            a = tensors[name].ravel()
            f.write(f"{name} {a.size} {a.astype(np.float64).sum():.9g} {a[0]:.9g} {a[-1]:.9g}\n")
    print(f"wrote {out} and {out}.ref")


if __name__ == "__main__":
    main()
