# Phase 1 — Behavioral results: does training format leave deeper samskaras?

**Status: LoRA results final (strict scoring). Full-parameter run being re-trained
(first attempt at lr 1e-5 learned the trace format but not the invented facts).**

## Setup

- Model: Qwen/Qwen3-0.6B-Base, LoRA (2.88M trainable = 0.48%), 800 iters,
  batch 4, lr 1e-4, MLX. Loss 5.83 → 1.02.
- Data: `data/run1` — depth 3, serial chains, interference 0.3, novelty 0.25,
  seed 7. 39 invented entities (20 block A / 19 block B), within-block links,
  fact-format disjointness asserted at generation.
- Condition A (flashcard): direct Q→A, exposure-padded.
  Condition B (integration): Q→verbalized trace→A. One combined training run.
- Scoring: **strict** = answer must appear after the last "answer is" (or in
  the first sentence when no marker). Loose = substring anywhere. All
  generations saved in `results/eval_lora_strict.json`.

## Results (LoRA, strict scoring)

| Test | Block A (flashcard) | Block B (trace) |
|---|---|---|
| Novel combinations (held out) | 10/43 = **23.3%** | 25/41 = **61.0%** |
| Recall, unseen templates | 50/168 = **29.8%** | 37/172 = **21.5%** |
| Reversals (control) | 0/8 = 0% | 1/12 = 8.3% |

Controls: base model = 0.0 on everything (no leakage; all learning is from the
fine-tune). Reversal curse unaffected by format, as predicted.

## The two findings

**1. The transfer effect survives strict scoring.** On never-trained fact
combinations, integration-format entities transfer at 2.6× the flashcard rate
(61% vs 23%). This is the samskara claim in behavioral form: the format of
training, at matched fact exposure, changes how flexibly the knowledge recombines.

**2. Over-serialization (unexpected, important).** B's loose recall was 50% but
strict only 21.5% — and the gap is NOT a scoring artifact. Inspecting the saved
generations: asked a *direct* question, the B-conditioned model states the
correct fact and then **keeps chaining hops**, committing to the end of a longer
chain: "The ally of Tisfidon is Vaor. The ally of Vaor is Gorlinshi. … So the
answer is Gorkrotol." Trace training didn't just teach reasoning — it made the
serial route the *default*, even where a lookup suffices. This is the mirror
image of the paper's automatic-vs-flexible dissociation: Condition B shifted
routing from automatic toward deliberate globally. (For Condition C, this
suggests the reflection data must include direct-answer cases so the model
learns *when* to reason, not just *how*.)

## Caveats

1. Token-count confound: B traces are 2.1× A's tokens at matched fact exposure
   (`data/run1/audit_report.json`). A token-matched ablation is the follow-up.
2. Single seed, single knob config. The sweep is future work.
3. Full-parameter run pending (first attempt underfit facts at lr 1e-5 —
   converged loss can hide unlearned rare tokens; retraining at 5e-5 × 1500).
