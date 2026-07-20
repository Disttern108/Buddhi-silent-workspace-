#!/usr/bin/env python3
"""Fresh-world in-context battery -> data/incontext/ (no training involved)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fact_world_generator import build_world, RELATIONS, SCALAR_ATTRS

rng = random.Random(99)
world = build_world(20, 0.0, rng)
names = list(world)


def fact_sentence(ent, attr):
    return f"The {attr} of {ent} is {world[ent][attr]}."


def rand_fact():
    e = rng.choice(names)
    a = rng.choice(SCALAR_ATTRS + RELATIONS)
    return (e, a)


two, one = [], []
for _ in range(40):
    start = rng.choice(names)
    rels = [rng.choice(RELATIONS) for _ in range(2)]
    attr = rng.choice(SCALAR_ATTRS)
    e1 = world[start][rels[0]]
    e2 = world[e1][rels[1]]
    ans = world[e2][attr]
    need = [(start, rels[0]), (e1, rels[1]), (e2, attr)]
    facts = list(need)
    while len(facts) < 8:
        f = rand_fact()
        if f not in facts:
            facts.append(f)
    rng.shuffle(facts)
    ctx = " ".join(fact_sentence(*f) for f in facts)
    q = f"What is the {attr} of the {rels[1]} of the {rels[0]} of {start}?"
    two.append({"prompt": ctx + " " + q, "answer": ans, "block": "ic",
                "type": "novel_combination"})
for _ in range(20):
    e = rng.choice(names)
    a = rng.choice(SCALAR_ATTRS)
    facts = [(e, a)]
    while len(facts) < 6:
        f = rand_fact()
        if f not in facts:
            facts.append(f)
    rng.shuffle(facts)
    ctx = " ".join(fact_sentence(*f) for f in facts)
    one.append({"prompt": ctx + f" What is the {a} of {e}?",
                "answer": world[e][a], "block": "ic",
                "type": "recall_unseen_template"})
os.makedirs("data/incontext", exist_ok=True)
with open("data/incontext/test_transfer.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in two]
with open("data/incontext/test_recall.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in one]
print(f"in-context chains: {len(two)}, direct controls: {len(one)}")
print("sample:", two[0]["prompt"][:160], "->", two[0]["answer"])
