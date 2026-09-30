#!/usr/bin/env python3
"""Time HF PyTorch GPT-2 on CPU the same way `gpt2 bench` times the C++ engine.

Same prompt ids ((t * 7919 + 13) % vocab), and the same three numbers, best of --reps:
prefill with all logits, time to first token (last position's logits only), and --gen greedy
decode steps through the KV cache (past_key_values).

    python scripts/bench_hf.py --prompt 64 --gen 32 --threads 6
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    import torch
    from transformers import GPT2LMHeadModel

    torch.set_num_threads(args.threads)
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[(t * 7919 + 13) % V for t in range(args.prompt)]])

    prefill_all = ttft = decode = float("inf")
    with torch.inference_mode():
        model(ids)  # warm-up
        for _ in range(args.reps):
            t0 = time.perf_counter()
            model(ids)
            prefill_all = min(prefill_all, time.perf_counter() - t0)

            t0 = time.perf_counter()
            out = model(ids, use_cache=True, logits_to_keep=1)
            ttft = min(ttft, time.perf_counter() - t0)

            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            t0 = time.perf_counter()
            for _ in range(args.gen):
                out = model(nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            decode = min(decode, time.perf_counter() - t0)

    print(f"HF PyTorch {torch.__version__}, threads: {args.threads}, prompt: {args.prompt} "
          f"tokens, decode: {args.gen} tokens, best of {args.reps}")
    print(f"prefill T={args.prompt} (all logits): {prefill_all * 1e3:.1f} ms "
          f"({args.prompt / prefill_all:.1f} tok/s)")
    print(f"time to first token (last logits only): {ttft * 1e3:.1f} ms")
    print(f"decode {args.gen} tokens after T={args.prompt}: {decode / args.gen * 1e3:.2f} ms/token "
          f"({args.gen / decode:.1f} tok/s)")


if __name__ == "__main__":
    main()
