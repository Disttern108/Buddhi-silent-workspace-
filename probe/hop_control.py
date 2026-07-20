#!/usr/bin/env python3
"""Null check: is the chain's true intermediate ranked better than a matched
non-intermediate entity from the same block? Writes results/<tag>/layers/
hop_control.json with per-chain best ranks for mid vs control."""
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.hf import from_hf
from jlens.lens import JacobianLens

model_dir, lens_path, data_dir, tag = sys.argv[1:5]
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
model.eval()
tok = AutoTokenizer.from_pretrained(model_dir)
lm = from_hf(model, tok)
lens = JacobianLens.load(lens_path)
world = json.load(open(os.path.join(data_dir, "world.json")))
chains = [json.loads(l) for l in open(os.path.join(data_dir, "test_transfer.jsonl"))
          if json.loads(l)["type"] == "novel_combination"]
rng = random.Random(1)
from collections import defaultdict
out = []
counts = defaultdict(int)
for t in chains:
    if counts[t["block"]] >= 12:
        continue
    m = re.search(r"What is the (\w+) of (.+)\?$", t["prompt"])
    if not m:
        continue
    attr, rest = m.groups()
    rels = list(reversed(re.findall(r"the (\w+) of", rest)))
    ent = rest.split()[-1]
    if ent not in world or not rels:
        continue
    mid1 = world[ent][rels[0]]
    ctrl = rng.choice([n for n, e in world.items()
                       if e["block"] == t["block"] and n not in (mid1, ent)])
    mid_id = tok(" " + mid1)["input_ids"][0]
    ctrl_id = tok(" " + ctrl)["input_ids"][0]
    if mid_id == ctrl_id:
        continue
    counts[t["block"]] += 1
    best_mid, best_ctrl = 10 ** 9, 10 ** 9
    for i in range(0, len(lens.source_layers), 9):
        chunk = lens.source_layers[i:i + 9]
        ll, mlog, _ = lens.apply(lm, t["prompt"], layers=chunk)
        T = mlog.shape[0]
        for l in chunk:
            order = ll[l].argsort(dim=-1, descending=True)
            for p in range(max(0, T - 6), T):
                best_mid = min(best_mid, int((order[p] == mid_id).nonzero()[0]))
                best_ctrl = min(best_ctrl, int((order[p] == ctrl_id).nonzero()[0]))
    out.append({"block": t["block"], "prompt": t["prompt"][:60], "mid": mid1,
                "ctrl": ctrl, "best_mid": best_mid, "best_ctrl": best_ctrl})
path = os.path.join("results", tag, "layers", "hop_control.json")
json.dump(out, open(path, "w"), indent=1)
for blk in sorted({r["block"] for r in out}):
    rows = [r for r in out if r["block"] == blk]
    wins = sum(1 for r in rows if r["best_mid"] < r["best_ctrl"])
    med = sorted(r["best_ctrl"] for r in rows)[len(rows) // 2]
    print(f"block {blk}: mid beats matched control {wins}/{len(rows)}  "
          f"(median control rank {med})")
