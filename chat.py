#!/usr/bin/env python3
"""Raw-prompt REPL (no chat template — safe for base models).
Usage: .venv/bin/python chat.py [model_path]   (default: trained run7 model)
"""
import sys

from mlx_lm import load, generate

path = sys.argv[1] if len(sys.argv) > 1 else "models/run7_s4_fused"
print(f"loading {path} ... (Ctrl+C to quit)")
model, tok = load(path)
while True:
    try:
        q = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if q:
        print(generate(model, tok, prompt=q, max_tokens=60))
