#!/usr/bin/env python3
"""Astrology-vocabulary transfer battery -> data/astro_transfer/.
MEASUREMENT ONLY (never trained on). F3-style: the page supplies every fact,
so no prior knowledge is needed — tests whether the trained space works on
vocabulary alien to the curriculum. Rulerships are randomized per page
(deliberately not real astrology: page must beat prior).
"""
import json
import os
import random

PLANETS = ["Mars", "Venus", "Jupiter", "Saturn", "Mercury"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
         "Scorpio", "Aquarius", "Pisces"]  # all <=2 tokens (no Capricorn/Sagittarius)
HOUSES = ["1", "2", "3", "4", "5", "7", "8", "9"]

rng = random.Random(21)


def make_world():
    return {"sign_of": dict(zip(PLANETS, rng.sample(SIGNS, len(PLANETS)))),
            "house_of": dict(zip(PLANETS, rng.sample(HOUSES, len(PLANETS)))),
            "ruler_of": {s: rng.choice(PLANETS) for s in rng.sample(SIGNS, 4)}}


def sentences(w):
    out = [f"{p} is in {s}." for p, s in w["sign_of"].items()]
    out += [f"{p} occupies house {h}." for p, h in w["house_of"].items()]
    out += [f"The ruler of {s} is {p}." for s, p in w["ruler_of"].items()]
    return out


recall, chains = [], []
for _ in range(20):
    w = make_world()
    p = rng.choice(PLANETS)
    kind = rng.choice(["sign", "house"])
    need = (f"{p} is in {w['sign_of'][p]}." if kind == "sign"
            else f"{p} occupies house {w['house_of'][p]}.")
    page = [need] + [x for x in rng.sample(sentences(w), 5) if x != need][:4]
    rng.shuffle(page)
    q = (f"What sign is {p} in?" if kind == "sign"
         else f"Which house does {p} occupy?")
    ans = w["sign_of"][p] if kind == "sign" else w["house_of"][p]
    recall.append({"prompt": " ".join(page) + " " + q, "answer": ans,
                   "block": "astro", "type": "recall_unseen_template"})

for _ in range(30):
    w = make_world()
    s = rng.choice(list(w["ruler_of"]))
    p = w["ruler_of"][s]
    kind = rng.choice(["sign", "house"])
    ans = w["sign_of"][p] if kind == "sign" else w["house_of"][p]
    need = [f"The ruler of {s} is {p}.",
            (f"{p} is in {w['sign_of'][p]}." if kind == "sign"
             else f"{p} occupies house {w['house_of'][p]}.")]
    page = need + [x for x in rng.sample(sentences(w), 6) if x not in need][:4]
    rng.shuffle(page)
    q = (f"What sign is the ruler of {s} in?" if kind == "sign"
         else f"Which house does the ruler of {s} occupy?")
    chains.append({"prompt": " ".join(page) + " " + q, "answer": ans,
                   "block": "astro", "type": "novel_combination"})

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/Users/aayush/Models/Qwen3-1.7B-Base")
for r in recall + chains:
    assert len(tok.encode(" " + r["answer"])) <= 2, r["answer"]

os.makedirs("data/astro_transfer", exist_ok=True)
with open("data/astro_transfer/test_recall.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in recall]
with open("data/astro_transfer/test_transfer.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in chains]
print(f"astro battery: {len(recall)} reading, {len(chains)} 2-hop chains")
print("sample:", chains[0]["prompt"][:160], "->", chains[0]["answer"])
