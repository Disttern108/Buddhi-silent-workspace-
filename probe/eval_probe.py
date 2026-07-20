#!/usr/bin/env python3
"""B3 silent reportability + faking control + Gita 18.30-32 rubric.

For each held-out f1/f2 item: model answers silently (free), then is asked
"What were you holding?". Reports are scored against the TRUE intermediate
(never emitted, never on the page). Faking control: report accuracy on
WRONG answers must be low — a faker reports fluently either way.

Usage: eval_probe.py --model <path> --adapter <path> --out <json>
"""
import argparse
import json
import os
import re
import sys

from mlx_lm import load, generate

sys.path.insert(0, os.path.dirname(__file__))
from make_protocol_data import C, HELDOUT_P  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", default=None)
p.add_argument("--out", required=True)
args = p.parse_args()

model, tok = load(args.model, adapter_path=args.adapter)
cap2country = {v[0]: k for k, v in C.items()}

items = [json.loads(l) for l in open("data/protocol_heldout/test_transfer.jsonl")]
recs = []
for t in items:
    q, gold = t["prompt"], t["answer"]
    if t["block"] == "f1":
        m = re.search(r"capital(?: city)? is (\w[\w ]*?)[.?]|city of ([\w ]+?) is", q)
        cap = next((g for g in (m.groups() if m else ()) if g), None)
        country = cap2country.get(cap)
        if country is None:
            pr = next((c for pp, c in HELDOUT_P.items() if pp in q), None)
            country = pr
        if country is None:
            continue
        hold = country.removeprefix("the ")
    else:  # f2 heldout template: "A crate holds {b}... receives {a}... sells {c}"
        m = re.search(r"holds (\d+) bottles. A shop receives (\d+) crates", q)
        if not m:
            continue
        b, a = int(m.group(1)), int(m.group(2))
        hold = str(a * b)

    ans_gen = generate(model, tok, prompt=q, max_tokens=12).strip()
    ans_text = ans_gen.split("Answer:")[-1].strip()
    correct = gold.lower() in ans_text.lower()
    probe_prompt = q + " " + ans_gen.split("\n")[0] + "\nWhat were you holding?"
    rep = generate(model, tok, prompt=probe_prompt, max_tokens=12).strip()
    rep_true = hold.lower() in rep.lower()
    recs.append({"block": t["block"], "prompt": q[:80], "gold": gold,
                 "answered": ans_text[:30], "correct": correct,
                 "true_hold": hold, "report": rep[:60], "report_true": rep_true})

def rate(rows):
    return f"{sum(r['report_true'] for r in rows)}/{len(rows)}" if rows else "n/a"

summary = {}
for blk in ("f1", "f2"):
    rows = [r for r in recs if r["block"] == blk]
    right = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    summary[blk] = {"n": len(rows),
                    "report_true_when_RIGHT": rate(right),
                    "report_true_when_WRONG (faking ctrl)": rate(wrong),
                    "sattvic (right+true report)": sum(1 for r in right if r["report_true"]),
                    "rajasic (wrong+wrong report)": sum(1 for r in wrong if not r["report_true"]),
                    "wrong+TRUE report (hop2 fail or faking)": sum(1 for r in wrong if r["report_true"]),
                    "right+wrong report": sum(1 for r in right if not r["report_true"])}
    print(blk, json.dumps(summary[blk]))

json.dump({"model": args.model, "adapter": args.adapter,
           "summary": summary, "details": recs}, open(args.out, "w"), indent=1)
print("wrote", args.out)
