#!/usr/bin/env python3
"""Phase 3 analysis: chain-rate cross-tab + exposure-stratified accuracy.

Runs on saved eval JSONs (no GPU). Usage:
  .venv/bin/python probe/phase3_analysis.py results/eval_run3.json [gagged.json]
"""
import json
import os
import re
import sys
from collections import defaultdict

DATA = os.environ.get("FW_DATA", "data/run3")
world = json.load(open(f"{DATA}/world.json"))
expl = json.load(open(f"{DATA}/exposure_explicit.json"))
ATTRS = {"capital", "language", "currency", "continent", "ally", "rival"}


def chain_stats(gen):
    """(n_trace_sentences, used_answer_marker)"""
    n = len([m for m in re.finditer(r"[Tt]he (\w+) of (\w+) is (\w+)", gen)
             if m.group(1) in ATTRS and m.group(2) in world])
    return n, ("answer is" in gen.lower() or "answer:" in gen.lower())


def question_exposure(prompt, block):
    """Sum of explicit training statements over the facts this question uses."""
    m = re.match(r"What is the (\w+) of (.+)\?", prompt)
    counts = expl.get(block, {})
    if m:
        attr, rest = m.groups()
        rels = list(reversed(re.findall(r"the (\w+) of", rest)))
        ent = rest.split()[-1]
        if ent in world and rels:
            total, cur = 0, ent
            for r in rels:
                total += counts.get(f"{cur}::{r}", 0)
                cur = world[cur][r]
            return total + counts.get(f"{cur}::{attr}", 0)
    for a in ATTRS:  # direct recall phrasings
        if a in prompt:
            for n in world:
                if n in prompt:
                    return counts.get(f"{n}::{a}", 0)
    return None


def analyze(path, label):
    d = json.load(open(path))
    print(f"\n=== {label}: {d['model']} suffix={d.get('suffix','')!r} ===")
    tab = defaultdict(lambda: [0, 0, 0, 0])  # (type,block)->[chained,multi,ok,n]
    strata = defaultdict(lambda: [0, 0])     # (type,block,expo_bin)->[ok,n]
    for g in d["generations"]:
        n_steps, marker = chain_stats(g["generated"])
        key = (g["type"], g["block"])
        tab[key][0] += (n_steps >= 1 or marker)
        tab[key][1] += (n_steps >= 2)
        tab[key][2] += g["strict"]
        tab[key][3] += 1
        e = question_exposure(g["prompt"], g["block"])
        if e is not None and g["type"] == "novel_combination":
            b = "lo" if e <= 8 else ("mid" if e <= 16 else "hi")
            s = strata[(g["block"], b)]
            s[0] += g["strict"]
            s[1] += 1
    print("type                      blk  chain%  multi-hop%  strict-acc     n")
    for (typ, blk), (c, m2, ok, n) in sorted(tab.items()):
        print(f"{typ:24s}  {blk}   {c/n:5.0%}   {m2/n:6.0%}     {ok/n:5.0%}      {n:4d}")
    if strata:
        print("transfer acc by training-exposure bin (facts' explicit statements):")
        for (blk, b) in sorted(strata):
            ok, n = strata[(blk, b)]
            print(f"  block {blk} expo-{b:3s}: {ok}/{n} = {ok/n:.0%}")


for i, path in enumerate(sys.argv[1:]):
    analyze(path, ["ungagged", "gagged"][min(i, 1)])
