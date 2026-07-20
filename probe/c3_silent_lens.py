#!/usr/bin/env python3
"""C3: lens at the silent answer moment. For each gagged real-world 2-hop
item (prompt + " Answer:"), rank the latent intermediate country vs a
matched control country (never in the chain) — best lens rank over the
last 6 positions x all source layers, hop_control.py's exact logic.
Successes carry the claim; failures are the comparison group.

Usage: c3_silent_lens.py <model_dir> <lens_path> <tag>
"""
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.hf import from_hf
from jlens.lens import JacobianLens

sys.path.insert(0, os.path.dirname(__file__))
from make_realworld import L  # noqa: E402

model_dir, lens_path, tag = sys.argv[1:4]
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
model.eval()
tok = AutoTokenizer.from_pretrained(model_dir)
lm = from_hf(model, tok)
lens = JacobianLens.load(lens_path)


def first_tok(country):
    return tok(" " + country.removeprefix("the "))["input_ids"][0]


country_of = {landmark: c for landmark, (c, _, _, _) in L.items()}
tok_of = {c: first_tok(c) for c in set(country_of.values())}
# distinctive first tokens only (drops the United Kingdom / the United States)
ambiguous = {t for t in tok_of.values() if list(tok_of.values()).count(t) > 1}
usable = {c: t for c, t in tok_of.items() if t not in ambiguous}

gag_json = sys.argv[4] if len(sys.argv) > 4 else "results/rw_b17_gag.json"
gens = json.load(open(gag_json))["generations"]
items = [g for g in gens if g["type"] == "novel_combination"]
rng = random.Random(1)
out = []
for it in items:
    landmark = next((lm_ for lm_ in L if it["prompt"].endswith(f"of {lm_}?")), None)
    if landmark is None or country_of[landmark] not in usable:
        continue
    mid = country_of[landmark]
    ctrl = rng.choice([c for c in usable if c != mid])
    mid_id, ctrl_id = usable[mid], usable[ctrl]
    ans_id = tok(" " + it["answer"])["input_ids"][0]
    prompt = it["prompt"] + " Answer:"
    best = {"mid": 10 ** 9, "ctrl": 10 ** 9, "ans": 10 ** 9}
    for i in range(0, len(lens.source_layers), 9):
        chunk = lens.source_layers[i:i + 9]
        ll, mlog, _ = lens.apply(lm, prompt, layers=chunk)
        T = mlog.shape[0]
        for l in chunk:
            order = ll[l].argsort(dim=-1, descending=True)
            for p in range(max(0, T - 6), T):
                best["mid"] = min(best["mid"], int((order[p] == mid_id).nonzero()[0]))
                best["ctrl"] = min(best["ctrl"], int((order[p] == ctrl_id).nonzero()[0]))
                best["ans"] = min(best["ans"], int((order[p] == ans_id).nonzero()[0]))
    out.append({"prompt": it["prompt"][:70], "correct": it["strict"],
                "mid": mid, "ctrl": ctrl, "answer": it["answer"],
                "best_mid": best["mid"], "best_ctrl": best["ctrl"],
                "best_ans": best["ans"]})

path = os.path.join("results", tag, "layers", "c3_silent.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(path, "w"), indent=1)
for label, rows in (("SUCCESSES", [r for r in out if r["correct"]]),
                    ("failures ", [r for r in out if not r["correct"]])):
    if not rows:
        continue
    wins = sum(1 for r in rows if r["best_mid"] < r["best_ctrl"])
    med_m = sorted(r["best_mid"] for r in rows)[len(rows) // 2]
    med_c = sorted(r["best_ctrl"] for r in rows)[len(rows) // 2]
    print(f"C3 {label}: mid beats control {wins}/{len(rows)}  "
          f"median mid rank {med_m}  median ctrl rank {med_c}")
