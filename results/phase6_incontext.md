# Phase 6 — Fresh-world in-context test: did a transferable skill form?

**Verdict: NO. The fine-tune fused procedure with content. run4 cannot apply
its reasoning format to a new world's facts — it cannot even READ a fact
stated in the prompt (15% vs base's 100%). Meanwhile the base model
independently re-confirmed page-dependence a second time: with facts
pre-written in context, free chains 33% → gagged 5%.**

Design: fresh world (seed 99, zero entity overlap), needed facts + 5
distractors written into every prompt in trace syntax, chain questions in
trained phrasing. Controls: direct in-context recall (answer is literally in
the prompt). Predictions pre-registered in `phase6_predictions.md`.

## Results (strict / n)

| model | mode | in-context recall (read a stated fact) | in-context chains |
|---|---|---|---|
| base | free | **100%** (20/20) | **32.5%** (50% loose) |
| base | gagged | 25% | 5% |
| run4 | free | **15%** | **0%** |
| run4 | gagged | 15% | 10% |

## Scored against predictions

- P1 (run4 skill transfers, ≥40%) — **REJECTED, 0%.**
- P3's failure branch triggered exactly: run4 fails even the reading control,
  so what it learned is "not even reading — world-bound token grooves only."
  Generations show it *ignoring the prompt's facts* and emitting
  trained-world-style fact patterns about entities it has never seen
  (inventing "The ally of Gortolka is Branvunal" — not in context), sometimes
  answering with training-world entities.
- P2 ✓ base in-context chains 32.5% (in predicted 10-30%+ range).
- P4 ✓ and important: even with the facts PRE-WRITTEN on the page, gagged
  composition fails in both models (base 33%→5%). The page-loop needs
  self-written derivation steps, not merely a readable page. Third
  independent confirmation of page-dependence (invented world, real world,
  fresh world).
- P5 resolved in reverse: base > run4 by 33 points.

## What the whole program now says, in one paragraph

Fine-tuning on reasoning-formatted data taught the model its world deeply
(80% novel-combination transfer inside), but the "skill" and the "facts"
are a single fused groove: outside its world the model is not base-equivalent
but base-MINUS — it lost real-world knowledge (Phase 5: 0% vs base 57-61%)
AND lost in-context reading (Phase 6: 15% vs base 100%). Nothing content-
independent was installed. The reasoning that does exist — in every model,
tuned or not — runs through self-written text on the page, not through an
internal workspace. A workspace that separates procedure from content,
holds intermediates internally, and survives gagging is exactly what data-
format conditioning does NOT create at 0.6B, and is the specific target of
the two remaining levers: the Quiet-STaR-style thought-reward loss (C-full,
the user's original two-signal proposal, still unrun) and scale (1.7B), both
with real capability-preservation (heavy mixing or LoRA).
