#!/usr/bin/env python3
"""Workspace detection battery (completion-compatible tests only).

Runs the structural J-space signatures from the Anthropic workspace paper on
any HF causal LM + fitted Jacobian lens, and saves per-layer tracking files:

  results/<tag>/layers/layer_metrics.csv   lens top-1 agreement, excess kurtosis,
                                           position autocorrelation, effective dim
  results/<tag>/layers/cka_matrix.npy      layer x layer CKA of lens-vector geometry
  results/<tag>/layers/ignition.json       per-layer A/B projection share vs mixture
  results/<tag>/layers/hop_rank_heatmap.csv  lens rank of two-hop intermediates
  results/<tag>/layers/readouts_hops.json  top-k lens tokens near answer positions

Usage:
  .venv/bin/python probe/battery.py --model Qwen/Qwen3-0.6B-Base \
      --lens results/qwen3-0.6b-base/lens.pt --tag qwen3-0.6b-base
"""
import argparse
import csv
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.examples import load_wikitext_prompts
from jlens.hf import from_hf
from jlens.lens import JacobianLens

SKIP = 16          # attention-sink positions excluded, as in lens fitting
LAYER_CHUNK = 8    # lens.apply layer chunking to bound logits memory
TOPK = 5

# two-hop completions with single-token latent intermediates (checked at runtime)
TWO_HOPS = [
    ("The capital of the country where the Eiffel Tower stands is the city of",
     " France", " Paris"),
    ("The capital of the country where the Great Wall was built is the city of",
     " China", " Beijing"),
    ("The capital of the country where Mount Fuji rises is the city of",
     " Japan", " Tokyo"),
    ("The capital of the country where the Colosseum stands is the city of",
     " Italy", " Rome"),
    ("The capital of the country where the Kremlin stands is the city of",
     " Russia", " Moscow"),
    ("The capital of the country where the Taj Mahal stands is the city of",
     " India", " Delhi"),
    ("The capital of the country where the pyramids of Giza stand is the city of",
     " Egypt", " Cairo"),
    ("The capital of the country where Machu Picchu lies is the city of",
     " Peru", " Lima"),
    ("The currency used in the country where the Eiffel Tower stands is the",
     " France", " euro"),
    ("The language spoken in the country where the Kremlin stands is",
     " Russia", " Russian"),
]

IGNITION_NAMES = [" France", " China", " Japan", " India", " Italy", " Russia",
                  " Brazil", " Egypt", " Spain", " Germany", " Canada", " Mexico"]
IGNITION_CARRIERS = [
    "Many tourists travel to{X} every year because the country is famous",
    "The long history of{X} is taught in schools all over the world",
    "My neighbor recently moved to{X} to start a completely new life",
    "Traditional food from{X} has become popular in many other countries",
]


def excess_kurtosis(x):
    x = x - x.mean(dim=-1, keepdim=True)
    var = (x ** 2).mean(dim=-1)
    return ((x ** 4).mean(dim=-1) / (var ** 2 + 1e-12)) - 3.0


def participation_ratio(s):
    lam = s ** 2
    return float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12))


def layer_metrics_and_hops(lm, lens, prompts, outdir):
    layers = lens.source_layers
    agg = {l: {"agree": 0, "n": 0, "kurt": 0.0, "kn": 0,
               "ac_adj": 0.0, "ac_null": 0.0, "acn": 0} for l in layers}
    g = torch.Generator().manual_seed(0)
    for prompt in prompts:
        for i in range(0, len(layers), LAYER_CHUNK):
            chunk = layers[i:i + LAYER_CHUNK]
            lens_logits, model_logits, ids = lens.apply(
                lm, prompt, layers=chunk, max_seq_len=128)
            T = model_logits.shape[0]
            if T <= SKIP + 8:
                break
            pos = list(range(SKIP, T - 1))
            model_top1 = model_logits.argmax(-1)
            for l in chunk:
                ll = lens_logits[l]
                top1 = ll.argmax(-1)
                a = agg[l]
                a["agree"] += int((top1[pos] == model_top1[pos]).sum())
                a["n"] += len(pos)
                a["kurt"] += float(excess_kurtosis(ll[pos]).sum())
                a["kn"] += len(pos)
                logp = torch.log_softmax(ll, dim=-1)
                adj = logp[[p + 1 for p in pos], top1[pos]]
                perm = torch.randperm(len(pos), generator=g)
                far = [p for p in pos]  # shuffled partner positions
                null = logp[[far[j] for j in perm], top1[pos]]
                a["ac_adj"] += float(adj.sum())
                a["ac_null"] += float(null.sum())
                a["acn"] += len(pos)
    rows = []
    for l in layers:
        a = agg[l]
        J = lens.jacobians[l].float()
        pr = participation_ratio(torch.linalg.svdvals(J))
        rows.append({
            "layer": l,
            "lens_top1_agreement": a["agree"] / max(1, a["n"]),
            "excess_kurtosis": a["kurt"] / max(1, a["kn"]),
            "pos_autocorr_dlogp": (a["ac_adj"] - a["ac_null"]) / max(1, a["acn"]),
            "effective_dim_pr": pr,
        })
    with open(os.path.join(outdir, "layer_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("layer_metrics.csv written")


def cka_matrix(lm, lens, prompts, outdir, n_tokens=512):
    ids = set()
    for p in prompts:
        ids.update(lm.encode(p, max_length=128)[0].tolist())
    ids = sorted(ids)[:n_tokens]
    W = lm._hf_model.get_output_embeddings().weight.detach().float().cpu()[ids]
    layers = lens.source_layers
    grams = []
    for l in layers:
        V = W @ lens.jacobians[l].float()      # token lens-vectors in layer-l space
        V = V - V.mean(0, keepdim=True)
        G = V @ V.T
        G = G - G.mean(0, keepdim=True) - G.mean(1, keepdim=True) + G.mean()
        grams.append(G)
    n = len(layers)
    M = torch.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            num = (grams[i] * grams[j]).sum()
            den = grams[i].norm() * grams[j].norm() + 1e-12
            M[i, j] = M[j, i] = num / den
    import numpy as np
    np.save(os.path.join(outdir, "cka_matrix.npy"),
            {"layers": layers, "cka": M.numpy()}, allow_pickle=True)
    print("cka_matrix.npy written")


def ignition(model, tok, device, n_layers, outdir):
    single = [n for n in IGNITION_NAMES if len(tok(n)["input_ids"]) == 1]
    pairs = [(single[i], single[i + 1]) for i in range(0, len(single) - 1, 2)]
    E = model.get_input_embeddings().weight.detach()
    alphas = [i / 10 for i in range(11)]
    acc = {l: {a: [] for a in alphas} for l in range(n_layers + 1)}
    for A, B in pairs:
        ida = tok(A)["input_ids"][0]
        idb = tok(B)["input_ids"][0]
        for carrier in IGNITION_CARRIERS:
            base_ids = tok(carrier.replace("{X}", B))["input_ids"]
            if idb not in base_ids:  # name merged with neighbors; skip carrier
                continue
            ent_pos = base_ids.index(idb)
            ids_t = torch.tensor([base_ids], device=device)
            embeds = E[ids_t]
            ref = {}
            for name, idx in (("A", ida), ("B", idb)):
                e = embeds.clone()
                e[0, ent_pos] = E[idx]
                with torch.no_grad():
                    hs = model(inputs_embeds=e, output_hidden_states=True).hidden_states
                ref[name] = torch.stack([h[0, ent_pos].float().cpu() for h in hs])
            d = ref["A"] - ref["B"]
            dn = (d * d).sum(-1) + 1e-12
            for a in alphas:
                e = embeds.clone()
                e[0, ent_pos] = (1 - a) * E[idb] + a * E[ida]
                with torch.no_grad():
                    hs = model(inputs_embeds=e, output_hidden_states=True).hidden_states
                h = torch.stack([x[0, ent_pos].float().cpu() for x in hs])
                share = ((h - ref["B"]) * d).sum(-1) / dn
                for l in range(len(share)):
                    acc[l][a].append(float(share[l]))
    out = {str(l): {str(a): {"mean": sum(v) / len(v),
                             "n": len(v)} for a, v in per.items() if v}
           for l, per in acc.items()}
    with open(os.path.join(outdir, "ignition.json"), "w") as f:
        json.dump({"pairs": pairs, "share_by_layer": out}, f, indent=1)
    print(f"ignition.json written ({len(pairs)} pairs)")


def two_hop(lm, tok, lens, outdir):
    layers = lens.source_layers
    heat, readouts = [], {}
    for pi, (prompt, mid, ans) in enumerate(TWO_HOPS):
        mid_ids = tok(mid)["input_ids"]
        ans_ids = tok(ans)["input_ids"]
        if len(mid_ids) != 1:
            continue
        mid_id, ans_id = mid_ids[0], ans_ids[0]
        for i in range(0, len(layers), LAYER_CHUNK):
            chunk = layers[i:i + LAYER_CHUNK]
            lens_logits, model_logits, ids = lens.apply(lm, prompt, layers=chunk)
            T = model_logits.shape[0]
            for l in chunk:
                ll = lens_logits[l]
                order = ll.argsort(dim=-1, descending=True)
                for p in range(max(0, T - 6), T):
                    rank_mid = int((order[p] == mid_id).nonzero()[0])
                    rank_ans = int((order[p] == ans_id).nonzero()[0])
                    heat.append({"prompt_id": pi, "layer": l, "position": p,
                                 "rank_intermediate": rank_mid,
                                 "rank_answer": rank_ans})
                readouts.setdefault(str(pi), {})[str(l)] = [
                    tok.decode([int(t)]) for t in order[T - 1][:TOPK]]
        readouts[str(pi)]["_prompt"] = prompt
        readouts[str(pi)]["_intermediate"] = mid
        readouts[str(pi)]["_answer"] = ans
    with open(os.path.join(outdir, "hop_rank_heatmap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(heat[0].keys()))
        w.writeheader()
        w.writerows(heat)
    with open(os.path.join(outdir, "readouts_hops.json"), "w") as f:
        json.dump(readouts, f, indent=1, ensure_ascii=False)
    print("hop_rank_heatmap.csv + readouts_hops.json written")


def invented_two_hop(lm, tok, lens, data_dir, outdir, per_block=15):
    """Lens rank of the FIRST latent hop entity on invented-world chain
    questions, tagged by condition block. The B>A readability comparison."""
    import re
    world = json.load(open(os.path.join(data_dir, "world.json")))
    chains = [json.loads(l) for l in open(os.path.join(data_dir, "test_transfer.jsonl"))
              if json.loads(l)["type"] == "novel_combination"]
    from collections import defaultdict
    layers = lens.source_layers
    rows, counts = [], defaultdict(int)
    for t in chains:
        if counts[t["block"]] >= per_block:
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
        mid_id = tok(" " + mid1)["input_ids"][0]
        ans_id = tok(" " + t["answer"])["input_ids"][0]
        counts[t["block"]] += 1
        for i in range(0, len(layers), LAYER_CHUNK):
            chunk = layers[i:i + LAYER_CHUNK]
            lens_logits, model_logits, _ = lens.apply(lm, t["prompt"], layers=chunk)
            T = model_logits.shape[0]
            for l in chunk:
                order = lens_logits[l].argsort(dim=-1, descending=True)
                for p in range(max(0, T - 6), T):
                    rows.append({
                        "block": t["block"], "layer": l, "position": p,
                        "prompt": t["prompt"][:60],
                        "rank_intermediate": int((order[p] == mid_id).nonzero()[0]),
                        "rank_answer": int((order[p] == ans_id).nonzero()[0])})
    if rows:
        with open(os.path.join(outdir, "invented_hop_rank.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("invented_hop_rank.csv written",
              {k: v for k, v in sorted(counts.items())})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default=None,
                   help="fact-world dir; adds invented-chain hop tracking")
    p.add_argument("--n-metric-prompts", type=int, default=20)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = from_hf(model, tok)
    lens = JacobianLens.load(args.lens)

    outdir = os.path.join("results", args.tag, "layers")
    os.makedirs(outdir, exist_ok=True)

    # held-out prompts: fitting used the first 100 wikitext records
    prompts = load_wikitext_prompts(100 + args.n_metric_prompts)[100:]

    layer_metrics_and_hops(lm, lens, prompts, outdir)
    cka_matrix(lm, lens, prompts, outdir)
    ignition(model, tok, device, lm.n_layers, outdir)
    two_hop(lm, tok, lens, outdir)
    if args.data:
        invented_two_hop(lm, tok, lens, args.data, outdir)
    print("battery complete:", outdir)


if __name__ == "__main__":
    main()
