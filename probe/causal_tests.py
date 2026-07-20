#!/usr/bin/env python3
"""Causal workspace tests on the invented fact-world (Phase 2).

Test 1 — ablation dissociation: project out the top-k J-lens directions at
band layers during generation. Paper's signature: flexible tasks (serial
chains) collapse, automatic tasks (direct recall) survive.

Test 2 — intermediate swap: replace the first latent hop entity's lens
direction with another same-block entity's and check whether the model's
answer flips to the counterfactual chain's answer (ground truth known by
construction from world.json).

Usage:
  .venv/bin/python probe/causal_tests.py --model models/run1_full_hf \
      --lens results/<tag>/lens.pt --data data/run1 --tag <tag> \
      --band 9:19 --k 10
"""
import argparse
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "jspace-replication", "src"))
from interventions import DirectionSwapHooks, token_direction  # noqa: E402

from jlens.hf import from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402


class AblateHooks(DirectionSwapHooks):
    """Project out an orthonormal basis Q at every hooked block output:
    h' = h - Q Q^T h."""

    def __init__(self, blocks, bases):  # bases: {layer: [k, d] orthonormal}
        self._blocks = blocks
        self._bases = dict(bases)
        self._handles = []
        self._dirs = {}

    def _make_hook(self, layer):
        Q = self._bases[layer]
        per_device = {}

        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            h = tensor.float()
            q = per_device.get(h.device)
            if q is None:
                q = Q.to(h.device)
                per_device[h.device] = q
            h = h - (h @ q.T) @ q
            new = h.to(tensor.dtype)
            return new if torch.is_tensor(output) else (new, *output[1:])

        return hook

    def __enter__(self):
        for layer in self._bases:
            self._handles.append(
                self._blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self


def strict_answer_ok(gen, answer):
    g = gen.lower()
    for mark in ("answer:", "answer is"):
        i = g.rfind(mark)
        if i >= 0:
            return answer.lower().strip() in g[i + len(mark):]
    return answer.lower().strip() in g.split(".")[0]


def parse_chain(prompt):
    """'What is the capital of the ally of the rival of X?' ->
    (attr, [rels in resolution order], entity)."""
    m = re.match(r"What is the (\w+) of (.+)\?", prompt)
    if not m:
        return None
    attr, rest = m.groups()
    rels = re.findall(r"the (\w+) of", rest)
    entity = rest.split()[-1]
    if not rels:
        return None
    return attr, list(reversed(rels)), entity


def first_tok(tok, name):
    return tok(" " + name)["input_ids"][0]


def gen_text(model, tok, device, prompt, max_new=60):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--data", default="data/run1")
    p.add_argument("--tag", required=True)
    p.add_argument("--band", default="14:24", help="layer band start:end")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-per-cell", type=int, default=30)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = from_hf(model, tok)
    lens = JacobianLens.load(args.lens)
    W = model.get_output_embeddings().weight.detach()
    band = list(range(*map(int, args.band.split(":"))))
    band = [l for l in band if l in lens.source_layers]
    world = json.load(open(os.path.join(args.data, "world.json")))
    rng = random.Random(0)

    def load(name):
        return [json.loads(l) for l in open(os.path.join(args.data, name))]

    recall = load("test_recall.jsonl")
    chains = [t for t in load("test_transfer.jsonl")
              if t["type"] == "novel_combination"]
    rng.shuffle(recall)
    rng.shuffle(chains)

    # ---- Test 1: ablation dissociation --------------------------------
    mid = band[len(band) // 2]
    abl_results = []
    for kind, tests in (("direct_recall", recall), ("serial_chain", chains)):
        for blk in sorted({t["block"] for t in tests}):
            subset = [t for t in tests if t["block"] == blk][: args.n_per_cell]
            for t in subset:
                clean = gen_text(model, tok, device, t["prompt"])
                # top-k lens tokens at last position, mid-band, from clean pass
                ll, _, _ = lens.apply(lm, t["prompt"], layers=[mid],
                                      positions=[-1])
                top = ll[mid][0].argsort(descending=True)[: args.k].tolist()
                bases = {}
                for l in band:
                    D = torch.stack([token_direction(lens, W, i, l) for i in top])
                    Q, _ = torch.linalg.qr(D.T)
                    bases[l] = Q.T[: args.k].contiguous()
                with AblateHooks(lm.layers, bases):
                    ablated = gen_text(model, tok, device, t["prompt"])
                # control: matched-rank random directions
                rnd_ids = rng.sample(range(W.shape[0]), args.k)
                rbases = {}
                for l in band:
                    D = torch.stack([token_direction(lens, W, i, l) for i in rnd_ids])
                    Q, _ = torch.linalg.qr(D.T)
                    rbases[l] = Q.T[: args.k].contiguous()
                with AblateHooks(lm.layers, rbases):
                    control = gen_text(model, tok, device, t["prompt"])
                abl_results.append({
                    "kind": kind, "block": blk, "prompt": t["prompt"],
                    "answer": t["answer"],
                    "clean_ok": strict_answer_ok(clean, t["answer"]),
                    "ablated_ok": strict_answer_ok(ablated, t["answer"]),
                    "control_ok": strict_answer_ok(control, t["answer"]),
                })

    # ---- Test 2: intermediate swap ------------------------------------
    swap_results = []
    for t in chains[: args.n_per_cell * 2]:
        parsed = parse_chain(t["prompt"])
        if not parsed:
            continue
        attr, rels, ent = parsed
        if ent not in world:
            continue
        mid1 = world[ent][rels[0]]                    # first latent hop
        block_ents = [n for n, e in world.items()
                      if e["block"] == t["block"] and n not in (mid1, ent)]
        alt = rng.choice(block_ents)
        cur = alt                                     # counterfactual resolve
        for r in rels[1:]:
            cur = world[cur][r]
        cf_answer = world[cur][attr]
        ta, tb = first_tok(tok, mid1), first_tok(tok, alt)
        if ta == tb or cf_answer == t["answer"]:
            continue
        clean = gen_text(model, tok, device, t["prompt"])
        dirs = {l: (token_direction(lens, W, ta, l),
                    token_direction(lens, W, tb, l)) for l in band}
        with DirectionSwapHooks(lm.layers, dirs):
            swapped = gen_text(model, tok, device, t["prompt"])
        swap_results.append({
            "prompt": t["prompt"], "block": t["block"],
            "intermediate": mid1, "alt": alt,
            "answer": t["answer"], "cf_answer": cf_answer,
            "clean_ok": strict_answer_ok(clean, t["answer"]),
            "swap_to_cf": strict_answer_ok(swapped, cf_answer),
            "swap_kept_orig": strict_answer_ok(swapped, t["answer"]),
            "swapped_gen": swapped[:160],
        })

    outdir = os.path.join("results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "causal_tests.json"), "w") as f:
        json.dump({"band": band, "k": args.k,
                   "ablation": abl_results, "swap": swap_results}, f, indent=1)

    def rate(rows, key):
        rows = list(rows)
        return f"{sum(r[key] for r in rows)}/{len(rows)}" if rows else "n/a"

    for kind in ("direct_recall", "serial_chain"):
        rows = [r for r in abl_results if r["kind"] == kind]
        print(f"{kind:14s} clean {rate(rows,'clean_ok')}  "
              f"ablated {rate(rows,'ablated_ok')}  control {rate(rows,'control_ok')}")
    correct = [r for r in swap_results if r["clean_ok"]]
    print(f"swap (on clean-correct): to_cf {rate(correct,'swap_to_cf')}  "
          f"kept_orig {rate(correct,'swap_kept_orig')}  (n_all={len(swap_results)})")


if __name__ == "__main__":
    main()
