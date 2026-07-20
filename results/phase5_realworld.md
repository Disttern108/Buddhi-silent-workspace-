# Phase 5 — External validation on real-world facts

**Two findings. (1) The page-dependence mechanism validates on real facts —
in the BASE model: gagging costs base 2-hop questions 50 points (61%→11%)
while costing 1-hop facts only 11 (57%→46%). (2) run4 scores ZERO on all
real-world items in every mode: the conditioning caused catastrophic
forgetting — our installed abilities displaced general knowledge.**

Battery: 46 real 2-hop items (landmark → country → capital/currency/language,
unambiguous single answers, trained-QA phrasing) + 28 one-hop controls.
Models × modes: {base, run4_full} × {free 80 tokens, gagged " Answer:" + 4}.
Predictions pre-registered in `phase5_predictions.md`.

## Results (strict scoring)

| model | mode | 1-hop recall | 2-hop transfer |
|---|---|---|---|
| base | free | 57.1% | 60.9% |
| base | gagged | 46.4% | **10.9%** |
| run4 | free | 0% | 0% |
| run4 | gagged | 0% | 0% |

## Scored against predictions

- P1 base 1-hop 57% ✓; gag validity marginal (−11 pts, band was ~10) — the
  interaction below is 5× larger than this format cost, so interpretable.
- P2 run4 ≤ base ✓ in the extreme (0%): **catastrophic forgetting.** 200
  wikitext rows did not protect general knowledge through 3,500 full-FT steps.
- P3 (run4 free vs gagged on real facts): unresolvable — floor effect.
- **P4 the signature interaction ✓ — the external validation.** Gag cost:
  1-hop −11 points; 2-hop −50 points (82% relative). Composite questions at
  0.6B depend on generating text before committing; direct facts do not.
  The invented-world mechanism (facts in weights, composition on the page)
  reproduces on real knowledge, in a model we never fine-tuned.
- P5 mediation nuance: base free 2-hop uses mixed strategies — 18/46
  verbalize the country first (true page chains); many others hit direct
  landmark→city associations ("The Eiffel Tower is located in Paris"). Free
  text lets either strategy run; the gag blocks both unpacking routes. So the
  clean claim is "indirect queries need generation room," with strategy mix
  unresolved at item level.

## What "real capability" means after this

- The **mechanism** we found generalizes: it is a property of how 0.6B-class
  models handle composition, confirmed outside our sandbox.
- The **fine-tuned model** is a specialist: near-perfect inside its invented
  world (80% transfer), zero outside it. Benchmarking run4 on public
  leaderboards would measure forgetting, not the finding. To build a model
  that keeps both: much heavier general-data mixing (10-30× more), lower lr /
  fewer steps, or LoRA (which in run1 changed behavior while touching 0.5% of
  weights — the natural preservation candidate).

## Program status after five phases

Forced: fact grooves (weights) · routing gate (internal, trainable) ·
reasoning loop (external/page — now externally validated) · workspace content
(verbalizable intermediates). Not forced: internal composition (the silent
workspace) — target of C-full (Quiet-STaR internalization) and/or 1.7B.
New constraint for all future runs: preserve general capability (mixing/LoRA),
both for honest benchmarking and because a workspace that costs the world
isn't the workspace we want.
