#!/usr/bin/env python3
"""Build tests/diff_prompts.jsonl: the prompt set for the differential test against HF.

Each line is {"name": ..., "ids": [...]}; ids are stored (not text) so every machine runs
exactly the same token sequences, including ones truncated at a token boundary and random ids
that no text would produce. The file is committed; rerun this only to change the set.

    python scripts/make_diff_prompts.py [--out tests/diff_prompts.jsonl]

Categories: short natural language, code, multilingual/unicode, repetition and degenerate
inputs, random token ids, medium prose, and long contexts up to n_ctx = 1024.
"""
import argparse
import inspect
import json
import random

SHORT = [
    "Hello, my name is",
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "The meaning of life is",
    "In the beginning God created the heaven and the earth.",
    "To be, or not to be, that is the question:",
    "The mitochondria is the powerhouse of the",
    "Breaking news: scientists have discovered",
    "My favorite recipe for chocolate chip cookies calls for",
    "The stock market fell sharply on Monday after",
    "Q: What is the boiling point of water?\nA:",
    "Translate English to French: cheese =>",
    "1, 2, 3, 4, 5,",
    "The quick brown fox jumps over the lazy dog.",
    "Dear Sir or Madam, I am writing to complain about",
    "import numpy as np\nimport",
    "It was the best of times, it was the worst of times,",
    "The following is a list of the largest cities in the world:\n1.",
    "Water is composed of two hydrogen atoms and",
    "I'm sorry, Dave. I'm afraid I can't",
    "The theory of general relativity predicts that",
    "Step 1: Preheat the oven to",
    "According to the latest census, the population of",
    "She opened the door and saw",
    "The Industrial Revolution began in",
    "E = mc^2 means that",
    "Roses are red, violets are blue,",
    "The best way to learn a new programming language is",
    "In 1492, Columbus sailed",
    "Photosynthesis converts sunlight into",
    "The defendant pleaded not guilty to charges of",
    "Our quarterly earnings exceeded expectations because",
    "The recipe needs 2 cups of flour, 1 cup of sugar, and",
    "\"Where are you going?\" she asked. \"To the",
    "Machine learning is a subfield of artificial intelligence that",
    "The patient presented with a fever of 39.5°C and",
    "Welcome to the jungle, we've got",
    "SELECT name, age FROM users WHERE",
    "<html>\n<head>\n<title>",
    "The Eiffel Tower is 330 metres tall and",
]

MULTILINGUAL = [
    "日本語のテキストと emoji 🙂 and numbers 3.14159 mixed together!!",
    "Привет, мир! Как дела? Сегодня хорошая погода.",
    "Der schnelle braune Fuchs springt über den faulen Hund. Straße, Größe, Übung.",
    "El rápido zorro marrón salta sobre el perro perezoso. ¿Dónde está la biblioteca?",
    "中华人民共和国成立于1949年。北京是首都。",
    "مرحبا بالعالم. كيف حالك اليوم؟",
    "नमस्ते दुनिया। आज मौसम अच्छा है।",
    "🙂👍🇺🇸👩‍👩‍👧‍👦 emoji soup 🎉🎉🎉 🚀",
    "Ελληνικά: Γειά σου κόσμε. Ωραία μέρα σήμερα.",
    "Mixed: café naïve résumé coöperate ﬁ Ⅻ ½ ① — “quotes” ‘single’ …",
]

REPEATED_PARAGRAPH = (
    "The committee met on Tuesday to review the budget. After a long discussion, they agreed "
    "to postpone the vote until the next meeting, when more members would be present. "
)


def pydoc_prose():
    """Long English prose from Python's bundled documentation, in a fixed topic order."""
    from pydoc_data.topics import topics

    return [topics[k] for k in sorted(topics) if len(topics[k]) > 4000]


def code_snippets():
    import json as json_mod
    import textwrap
    import heapq
    import functools
    import shlex
    import difflib
    import fractions
    import string
    import statistics
    import ipaddress
    import base64

    funcs = [textwrap.dedent, json_mod.dumps, heapq.nsmallest, heapq.merge, shlex.shlex.read_token,
             functools.lru_cache, shlex.split, difflib.get_close_matches, fractions.Fraction.limit_denominator,
             string.capwords, statistics.median, ipaddress.ip_network, base64.b64encode]
    return [(f.__qualname__, inspect.getsource(f)) for f in funcs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/diff_prompts.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    n_ctx, eot = 1024, tok.eos_token_id
    prompts = []

    def add(name, ids):
        assert 1 <= len(ids) <= n_ctx, (name, len(ids))
        prompts.append({"name": name, "ids": list(map(int, ids))})

    for i, s in enumerate(SHORT):
        add(f"short/{i:02d}", tok.encode(s))
    for i, s in enumerate(MULTILINGUAL):
        add(f"unicode/{i:02d}", tok.encode(s))
    for name, src in code_snippets():
        add(f"code/{name}", tok.encode(src)[:160])

    add("edge/single_eot", [eot])
    add("edge/single_token", tok.encode("Hello"))
    add("edge/eot_boundaries", tok.encode("First document.<|endoftext|>Second document starts"))
    add("repeat/the", tok.encode("the " * 150))
    add("repeat/paragraph_x4", tok.encode(REPEATED_PARAGRAPH * 4))
    add("repeat/digits", tok.encode("0123456789" * 30))
    add("repeat/newlines", tok.encode("\n" * 64 + "end"))
    add("repeat/spaces", tok.encode("a" + " " * 200 + "b"))
    add("repeat/punctuation", tok.encode("!?.,;:" * 40))

    rng = random.Random(0)
    for i, n in enumerate([8, 32, 64, 128, 200, 256]):
        add(f"random/{i}_len{n}", [rng.randrange(tok.vocab_size) for _ in range(n)])

    prose = pydoc_prose()
    for i, text in enumerate(prose[:12]):
        add(f"prose/{i:02d}_len200", tok.encode(text)[:200])
    long_text = tok.encode("\n\n".join(prose[12:]))
    add("long/len512", long_text[:512])
    add("long/len768", long_text[1000:1768])
    add("long/len1024_full_ctx", long_text[2000:3024])
    # Full context mixing a document boundary in the middle, as in GPT-2 training data.
    add("long/len1024_eot_mid", long_text[4000:4511] + [eot] + long_text[5000:5512])

    with open(args.out, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    total = sum(len(p["ids"]) for p in prompts)
    print(f"wrote {len(prompts)} prompts, {total} tokens, to {args.out}")


if __name__ == "__main__":
    main()
