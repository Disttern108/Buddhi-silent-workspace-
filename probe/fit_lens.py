#!/usr/bin/env python3
"""Fit a Jacobian lens for a (small) HF causal LM and save it.

Usage:
  .venv/bin/python probe/fit_lens.py --model Qwen/Qwen3-0.6B-Base \
      --out results/qwen3-0.6b-base/lens.pt --n-prompts 100
"""
import argparse
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.examples import load_wikitext_prompts
from jlens.fitting import fit
from jlens.hf import from_hf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n-prompts", type=int, default=100)
p.add_argument("--dim-batch", type=int, default=32)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
# ponytail: float32 on 0.6B fits easily in 24GB; cleaner Jacobians than bf16
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
model.eval()
tok = AutoTokenizer.from_pretrained(args.model)
lm = from_hf(model, tok)

prompts = load_wikitext_prompts(args.n_prompts)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
lens = fit(
    lm, prompts,
    dim_batch=args.dim_batch,
    max_seq_len=128,
    checkpoint_path=args.out + ".ckpt",
    checkpoint_every=5,
)
lens.save(args.out)
print("saved lens to", args.out)
