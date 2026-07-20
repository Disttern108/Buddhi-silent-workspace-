# Phase 5 — pre-registered predictions (real-world external validation)

Written 2026-07-18 BEFORE any real-world eval ran. Battery: ~45 real 2-hop
questions (landmark → country → capital/currency/language, unambiguous
answers) + ~30 one-hop controls, phrased in the trained QA style. Models:
base Qwen3-0.6B-Base and run4_full. Modes: free (80 tokens) and gagged
(" Answer:" + 4 tokens; "Answer:" is a common pretraining convention, so
plausibly in-distribution for the base model too — its 1-hop gagged score is
the validity check).

1. **Base 1-hop**: decent (famous capitals; guess 35-65%). Base gagged 1-hop
   within ~10 points of free → gag valid for base.
2. **run4 1-hop real facts ≤ base** (narrow fine-tune damaged general
   knowledge; run3 showed OOD brittleness).
3. **run4 free 2-hop > run4 gagged 2-hop** IF the page procedure generalizes
   to real facts (the trace syntax "The X of Y is Z" transfers even if the
   facts are new). Collapse ratio comparable to invented-world (80→8 was
   ~90% relative drop; predict ≥50% relative drop here).
4. **The signature interaction (the external-validation claim): gag hurts
   2-hop far more than 1-hop in any model that chains.** If gag hurts both
   equally, the page-loop finding does not transfer to real facts.
5. Base free 2-hop vs gagged 2-hop: exploratory — base rarely chains
   spontaneously; if it doesn't chain, gag should cost it little (its 2-hop
   is latent). A base showing NO gag effect on 2-hop while run4 collapses
   would be the cleanest dissociation: pretraining put real 2-hops partly in
   weights; our fine-tune put invented 2-hops on the page.
