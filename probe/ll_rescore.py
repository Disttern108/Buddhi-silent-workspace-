#!/usr/bin/env python3
"""Closed-set log-likelihood rescoring: for each item, rank all candidate
answers (unique answers in the same test file) by logP(" <cand>" | prompt +
" Answer:"). Top-1 == gold -> correct. No generation, so immune to rambling,
token-budget, and cue-continuation artifacts.

Usage:
  .venv/bin/python probe/ll_rescore.py --model <path> --data data/realworld \
      --out results/ll_rw_b17.json [--limit N]
"""
import argparse
import json
from collections import defaultdict

import mlx.core as mx
from mlx_lm import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--suffix", default=" Answer:")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

model, tok = load(args.model)


def cand_logprob(prefix_ids, cand_ids):
    full = mx.array([prefix_ids + cand_ids])
    logits = model(full)[0]
    lp = 0.0
    for i, t in enumerate(cand_ids):
        pos = len(prefix_ids) - 1 + i
        row = logits[pos] - mx.logsumexp(logits[pos])
        lp += row[t].item()
    return lp


results = defaultdict(lambda: [0, 0])
details = []
for name in ("test_recall.jsonl", "test_transfer.jsonl"):
    with open(f"{args.data}/{name}") as f:
        tests = [json.loads(l) for l in f]
    if args.limit:
        tests = tests[: args.limit]
    cands = sorted({t["answer"] for t in tests})
    cand_ids = {c: tok.encode(" " + c) for c in cands}
    for t in tests:
        prefix = tok.encode(t["prompt"] + args.suffix)
        scores = {c: cand_logprob(prefix, cand_ids[c]) for c in cands}
        top = max(scores, key=scores.get)
        ok = top == t["answer"]
        key = (name, t["type"], t["block"])
        results[key][0] += ok
        results[key][1] += 1
        details.append({**t, "file": name, "top": top, "correct": ok,
                        "gold_rank": sorted(scores, key=scores.get,
                                            reverse=True).index(t["answer"]) + 1,
                        "n_cands": len(cands)})

table = [{"file": k[0], "type": k[1], "block": k[2], "correct": v[0],
          "total": v[1], "acc": round(v[0] / v[1], 3)}
         for k, v in sorted(results.items())]
with open(args.out, "w") as f:
    json.dump({"model": args.model, "data": args.data, "suffix": args.suffix,
               "table": table, "details": details}, f, indent=1)
for r in table:
    print(f"{r['file']:22s} {r['type']:24s} block {r['block']}  "
          f"LL top-1 {r['correct']:3d}/{r['total']:3d}={r['acc']}")
