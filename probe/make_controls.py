#!/usr/bin/env python3
"""Post-review control batteries -> data/incontext_st/, data/realworld_rare/.

C2: in-context pages about FICTIONAL subjects whose answer VALUES are real
single-token words (cities/currencies/colors) — separates page->space deficit
from invented-multi-token production artifacts.
C4: rare-direction real-world compositions ("the country whose capital is X")
— both hops individually known, composed phrasing unlikely verbatim in
pretraining; separates composition from memorized co-occurrence.
"""
import json
import os
import random

rng = random.Random(7)

SUBJECTS = ["Zorland", "Velcria", "Mardun", "Tessia", "Brovia", "Kelmar",
            "Dranvia", "Solpan", "Vintra", "Halden", "Quorast", "Nembla"]
VALUES = {
    "capital": ["Paris", "Rome", "Lima", "Cairo", "Tokyo", "Athens",
                "Berlin", "Madrid", "Moscow", "London"],
    "currency": ["euro", "yen", "peso", "ruble", "pound", "dollar", "franc"],
    "color": ["red", "blue", "green", "gold", "black", "white"],
}
RELS = ["ally", "rival"]

world = {}
for s in SUBJECTS:
    others = [o for o in SUBJECTS if o != s]
    world[s] = {a: rng.choice(v) for a, v in VALUES.items()}
    world[s]["ally"] = rng.choice(others)
    world[s]["rival"] = rng.choice([o for o in others if o != world[s]["ally"]])


def sent(e, a):
    return f"The {a} of {e} is {world[e][a]}."


def page(need, n=8):
    facts = list(need)
    while len(facts) < n:
        f = (rng.choice(SUBJECTS), rng.choice(list(VALUES) + RELS))
        if f not in facts:
            facts.append(f)
    rng.shuffle(facts)
    return " ".join(sent(*f) for f in facts)


two, one = [], []
for _ in range(40):
    start = rng.choice(SUBJECTS)
    r1, r2 = rng.choice(RELS), rng.choice(RELS)
    attr = rng.choice(list(VALUES))
    e1 = world[start][r1]
    e2 = world[e1][r2]
    ans = world[e2][attr]
    ctx = page([(start, r1), (e1, r2), (e2, attr)])
    q = f"What is the {attr} of the {r2} of the {r1} of {start}?"
    two.append({"prompt": ctx + " " + q, "answer": ans, "block": "st",
                "type": "novel_combination"})
for _ in range(20):
    e = rng.choice(SUBJECTS)
    a = rng.choice(list(VALUES))
    ctx = page([(e, a)], n=6)
    one.append({"prompt": ctx + f" What is the {a} of {e}?",
                "answer": world[e][a], "block": "st",
                "type": "recall_unseen_template"})

# C4: inverse-direction real compositions from the Phase-5 fact table
from make_realworld import L  # noqa: E402  (same folder)
rare = []
for lm, (country, cap, cur, lang) in L.items():
    for attr, val in (("currency", cur), ("language", lang)):
        if val:
            rare.append({"prompt": f"What is the {attr} of the country whose"
                                   f" capital is {cap}?",
                         "answer": val, "block": "rare",
                         "type": "novel_combination"})

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/Users/aayush/Models/Qwen3-1.7B-Base")
for r in two + one:
    assert len(tok.encode(" " + r["answer"])) <= 2, r["answer"]

os.makedirs("data/incontext_st", exist_ok=True)
os.makedirs("data/realworld_rare", exist_ok=True)
with open("data/incontext_st/test_transfer.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in two]
with open("data/incontext_st/test_recall.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in one]
with open("data/realworld_rare/test_transfer.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in rare]
# eval_model.py expects both files; reuse the standard 1-hop battery rows
with open("data/realworld/test_recall.jsonl") as f:
    rows = f.read()
with open("data/realworld_rare/test_recall.jsonl", "w") as f:
    f.write(rows)
print(f"incontext_st: {len(two)} chains, {len(one)} reading; "
      f"realworld_rare: {len(rare)} inverse 2-hop")
print("sample st:", two[0]["prompt"][:150], "->", two[0]["answer"])
print("sample rare:", rare[0]["prompt"], "->", rare[0]["answer"])
