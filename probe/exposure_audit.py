#!/usr/bin/env python3
"""Per-fact explicit-exposure audit. Usage: exposure_audit.py <data_dir>"""
import json
import re
import statistics as st
import sys
from collections import defaultdict

data = sys.argv[1]
world = json.load(open(f"{data}/world.json"))
attrs = {"capital", "language", "currency", "continent", "ally", "rival"}
expl = {b: defaultdict(int) for b in "ABC"}
for b in "ABC":
    for line in open(f"{data}/train_{b}.jsonl"):
        r = json.loads(line)
        text = r["prompt"] + r["completion"]
        for m in re.finditer(r"[Tt]he (\w+) of (\w+) is (\w+)", text):
            a, e = m.group(1), m.group(2)
            if e in world and a in attrs:
                expl[b][f"{e}::{a}"] += 1
json.dump({b: dict(expl[b]) for b in "ABC"},
          open(f"{data}/exposure_explicit.json", "w"))
for b in "ABC":
    v = list(expl[b].values())
    print(f"block {b}: facts stated {len(v)}, median stmts/fact "
          f"{int(st.median(v)) if v else 0}, mean {st.mean(v):.1f}" if v
          else f"block {b}: 0 explicit statements (flashcard format)")
