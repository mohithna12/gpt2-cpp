#!/usr/bin/env python3
"""Write tests/tokenizer_cases.tsv: texts with the token ids HF's GPT-2 tokenizer gives them.

Each line is `<utf-8 text as hex>\\t<space-separated ids>`. The C++ test must reproduce every
line exactly. Cases: hand-picked edge cases, real multi-line text, and seeded random strings
mixing every character class the pre-tokenizer regex distinguishes.

    python scripts/make_tokenizer_fixtures.py [--out tests/tokenizer_cases.tsv]
"""
import argparse
import glob
import os
import random

HANDPICKED = [
    "", " ", "  ", "\n", "\n\n", " \n", "\n ", "\t\t x", "Hello world", "Hello  world",
    "Hello world ", "Hello world  ", " Hello", "  Hello", "\n\nHello\n\nworld\n",
    "I'm here, you're there; they've gone and we'll see. It's Bob's; he'd go.",
    "'s 't 're 've 'm 'll 'd", "'S 'T 'RE", "don't DON'T", "''s", "'", "' s", "x'", "rock'n'roll",
    "123", "1234567890", " 42", "3.14159", "1,000,000", "v2.0.1", "x86_64", "a1b2c3",
    "!!!", " !!!", "?!.", "...", "---", "(foo)", "[1, 2]", "{\"a\": 1}", "a+b=c", "#include <x>",
    "<|endoftext|>", "a<|endoftext|>b", " <|endoftext|> ", "<|endoftext|><|endoftext|>",
    "<|endoftext", "<|endoftext|", "<|ENDOFTEXT|>",
    "café", "naïve résumé", "Ünïcödé", "é", " ́", "Straße", "ﬁ ligature",
    "日本語のテキスト", "中文字符", "한국어 텍스트", "Привет, мир!", "Γειά σου κόσμε",
    "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก", "٣٤٥ ۱۲۳ ½ Ⅻ ①",
    "🙂", "🙂🙂🙂", " 🙂", "👩‍👩‍👧‍👦", "🇺🇸", "a🙂b", "x​y", "﻿BOM",
    " ", "a b", "a　b", "a b", "a\u0085b", "a\u001cb\u001fc", "\x00\x01\x7f",
    "\r\n", "line1\r\nline2\r\n", "\t", "tab\tseparated\tvalues", "trailing   ",
    "aaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "antidisestablishmentarianism", "pneumonoultramicroscopicsilicovolcanoconiosis",
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x * 2  # double\n",
    "\U0001F600\U00010000\U0010FFFD", "�", " private use",
]

POOL = (
    list("abcXYZ0129 \n\t'!?.,;:-_()[]{}\"#@$%^&*+=/\\|<>~`")
    + ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "<|endoftext|>", "  ", "\r\n"]
    + list("éüñßøÆœ€£¥©®°±µ¶·¿×÷")
    + list("日本語中文한국Привет Γειά مرحبا שלום नमस्ते สวัสดี")
    + list("٣۱½Ⅻ①²³¼⁴₅")
    + [" ", "　", " ", " ", "\u0085", " ", "​", "\u001c", "\u000b"]
    + ["́", "̈", "‍", "️", "🙂", "👍", "🇺", "🇸", "\U0001F600"]
)


def random_cases(n, seed=0):
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 24)):
            if rng.random() < 0.08:
                cp = rng.randrange(0x110000)
                while 0xD800 <= cp < 0xE000:
                    cp = rng.randrange(0x110000)
                parts.append(chr(cp))
            else:
                parts.append(rng.choice(POOL))
        cases.append("".join(parts))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/tokenizer_cases.tsv")
    ap.add_argument("--random", type=int, default=5000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    root = os.path.join(os.path.dirname(__file__), "..")
    texts = [open(p, encoding="utf-8").read()
             for p in sorted(glob.glob(os.path.join(root, "*.md")))
             + sorted(glob.glob(os.path.join(root, "scripts", "gpt2bin.py")))]
    cases = HANDPICKED + texts + random_cases(args.random)

    with open(args.out, "w") as f:
        for text in cases:
            ids = tok.encode(text)
            assert tok.decode(ids) == text, repr(text)
            f.write(text.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
