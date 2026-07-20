#!/usr/bin/env python3
"""run8 route-reward RL (see results/phase8_predictions.md for the contract).

GRPO-style REINFORCE on LoRA only, starting from run7_s4. Reward = fruit
+ sattvic bonus − tamasic inconsistency − silence leak. K samples per item,
advantage vs group mean, loss on answer+report tokens.

Usage: rl_route.py [--steps 300] [--out adapters/run8_rl]
"""
import argparse
import json
import os
import random
import re
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import load_adapters

sys.path.insert(0, os.path.dirname(__file__))
from make_protocol_data import C, PERSONS, TRAIN_C, f1_items, f2_items  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default="/Users/aayush/Models/Qwen3-1.7B-Base")
p.add_argument("--adapter", default="adapters/run7_s4")
p.add_argument("--out", default="adapters/run8_rl")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--k", type=int, default=4)
p.add_argument("--prompts-per-step", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-5)
args = p.parse_args()

model, tok = load(args.model)
model.freeze()
load_adapters(model, args.adapter)  # adds LoRA modules (trainable) + weights
opt = optim.Adam(learning_rate=args.lr)
sampler = make_sampler(temp=0.9)

rng = random.Random(31)
items = f1_items(TRAIN_C, PERSONS) + f2_items(600, rng)
for it in items:
    it["gold_hold"] = it["hold_short"]
rng.shuffle(items)


def parse_ans(text):
    m = re.search(r"Answer:\s*([^\n.]+)", text)
    return m.group(1).strip() if m else None


def f2_consistent(q, held, ans):
    """Does the reported product explain the model's own answer?"""
    try:
        held_v, ans_v = int(held), int(ans)
    except (ValueError, TypeError):
        return False
    nums = [int(x) for x in re.findall(r"\d+", q)]
    cands = {held_v + n for n in nums} | {held_v - n for n in nums} | \
            {n - held_v for n in nums} | {held_v}
    return ans_v in cands


def f1_consistent(held, ans):
    for country, vals in C.items():
        if held and held.lower() in country.lower():
            return ans in vals
    return False


def reward(it, ans_text, rep_text):
    ans = parse_ans("Answer: " + ans_text if "Answer" not in ans_text else ans_text)
    rep = rep_text.split("Holding was:")[-1].strip().rstrip(".") if rep_text else ""
    r = 0.0
    if ans is None or "Holding:" in ans_text:
        r -= 0.5
    correct = ans is not None and it["ans"].lower() == ans.lower()
    if correct:
        r += 1.0
        if it["gold_hold"].lower() in rep.lower():
            r += 0.5
    if ans is not None:
        ok = (f2_consistent(it["q"], rep, ans) if it["family"] == "f2"
              else f1_consistent(rep, ans))
        if not ok:
            r -= 1.0
    return r, correct


def seq_logprob_loss(mdl, full_ids, prefix_len, adv):
    logits = mdl(full_ids[None, :-1])[0]
    targets = full_ids[1:]
    lp = nn.losses.cross_entropy(logits, targets, reduction="none")
    span = lp[prefix_len - 1:]
    return adv * span.mean()  # cross_entropy = -logp, so +adv*CE = -adv*logp


state = [model.state, opt.state]


def step(batch):
    trajs = []
    for it in batch:
        group = []
        for _ in range(args.k):
            a = generate(model, tok, prompt=it["q"], max_tokens=14,
                         sampler=sampler)
            probe = it["q"] + " " + a.split("\n")[0].strip() + \
                "\nWhat were you holding?"
            rep = generate(model, tok, prompt=probe, max_tokens=10,
                           sampler=sampler)
            r, correct = reward(it, a, rep)
            group.append((it, a, probe, rep, r, correct))
        mean_r = sum(g[4] for g in group) / len(group)
        for it_, a, probe, rep, r, correct in group:
            adv = r - mean_r
            if abs(adv) < 1e-6:
                continue
            trajs.append((it_["q"], a, adv))
            trajs.append((probe, rep, adv))
    if not trajs:
        return 0.0, 0.0
    lvag = nn.value_and_grad(model, lambda m, f, pl, ad: seq_logprob_loss(m, f, pl, ad))
    from mlx.utils import tree_map
    acc = None
    for prompt, compl, adv in trajs:
        pre = tok.encode(prompt)
        full = mx.array(pre + tok.encode(compl, add_special_tokens=False))
        _, grads = lvag(model, full, len(pre), mx.array(adv / len(trajs)))
        acc = grads if acc is None else tree_map(mx.add, acc, grads)
    opt.update(model, acc)
    mx.eval(state)
    rs = [g for g in trajs]
    return sum(t[2] for t in rs) / len(rs), len(trajs)


os.makedirs(args.out, exist_ok=True)
hist = []
for s in range(1, args.steps + 1):
    batch = [items[(s * args.prompts_per_step + j) % len(items)]
             for j in range(args.prompts_per_step)]
    # quick reward telemetry on the batch
    _, n = step(batch)
    if s % 10 == 0:
        probe_items = rng.sample(items, 8)
        rs, cs = [], 0
        for it in probe_items:
            a = generate(model, tok, prompt=it["q"], max_tokens=14)
            r, c = reward(it, a, "")
            rs.append(r)
            cs += c
        print(f"step {s}: greedy reward {sum(rs)/len(rs):+.2f} "
              f"correct {cs}/8  trajs {n}", flush=True)
        hist.append({"step": s, "greedy_r": sum(rs) / len(rs), "correct8": cs})
    if s % 50 == 0:
        from mlx.utils import tree_flatten
        w = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(f"{args.out}/adapters.safetensors", w)
        import shutil
        shutil.copy(f"{args.adapter}/adapter_config.json",
                    f"{args.out}/adapter_config.json")
        print(f"  checkpoint saved at step {s}", flush=True)

json.dump(hist, open(f"{args.out}/history.json", "w"), indent=1)
print("RL DONE")
