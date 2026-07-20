#!/usr/bin/env python3
"""Evaluate a (fine-tuned) model on fact-world test sets, split by condition.

Usage:
  .venv/bin/python eval_model.py --model Qwen/Qwen3-0.6B-Base \
      [--adapter adapters] --data data/run1 --out results/phase1_eval.json
"""
import argparse
import json
from collections import defaultdict

from mlx_lm import load, generate


def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", default=None)
p.add_argument("--data", default="data/run1")
p.add_argument("--out", required=True)
p.add_argument("--max-tokens", type=int, default=80)
p.add_argument("--limit", type=int, default=None, help="cap per test file")
p.add_argument("--suffix", default="", help='e.g. " The answer is" to gag chaining')
args = p.parse_args()

model, tok = load(args.model, adapter_path=args.adapter)

def strict_ok(gen, answer):
    """Score only the model's committed answer: text after the last
    commitment marker, or the first sentence when no marker is present."""
    g = gen.lower()
    for mark in ("answer:", "answer is"):
        i = g.rfind(mark)
        if i >= 0:
            return answer.lower().strip() in g[i + len(mark):]
    return answer.lower().strip() in g.split(".")[0]


results = defaultdict(lambda: [0, 0, 0])  # (file,type,block) -> [loose, strict, n]
generations = []
for name in ("test_recall.jsonl", "test_transfer.jsonl"):
    tests = rows(f"{args.data}/{name}")
    if args.limit:
        tests = tests[: args.limit]
    for t in tests:
        gen = generate(model, tok, prompt=t["prompt"] + args.suffix,
                       max_tokens=args.max_tokens)
        loose = t["answer"].lower() in gen.lower()
        strict = strict_ok(gen, t["answer"])
        key = (name, t["type"], t["block"])
        results[key][0] += loose
        results[key][1] += strict
        results[key][2] += 1
        generations.append({**t, "file": name, "generated": gen,
                            "loose": loose, "strict": strict})

table = [{"file": k[0], "type": k[1], "block": k[2],
          "loose": v[0], "strict": v[1], "total": v[2],
          "acc_loose": round(v[0] / v[2], 3), "acc_strict": round(v[1] / v[2], 3)}
         for k, v in sorted(results.items())]
with open(args.out, "w") as f:
    json.dump({"model": args.model, "adapter": args.adapter,
               "suffix": args.suffix, "table": table,
               "generations": generations}, f, indent=1)
for r in table:
    print(f"{r['file']:22s} {r['type']:24s} block {r['block']}  "
          f"strict {r['strict']:3d}/{r['total']:3d}={r['acc_strict']}  "
          f"loose {r['loose']:3d}/{r['total']:3d}={r['acc_loose']}")
