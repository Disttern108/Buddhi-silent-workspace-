# Phase 3 — Condition C: can the reasoning groove carry its own gate?

**Headline: yes — behaviorally. The mixed regime (traces + direct QA on the same
entities) dominated both pure formats on both axes AND gated its reasoning by
question type. Architecture still did not emerge, readability results got noisy
at this world size, and the weights-vs-page-loop question remains open because
every gag variant proved out-of-distribution for so narrow a model.**

Setup: run3 = 100 families (+30 twins = 130 entities), blocks A/B/C 44/39/47,
within-block links, contamination asserts green. Training: 2,434 rows
(587 A-flashcard / 347 B-trace / 1,324 C-mixed / 200 wikitext anti-narrowing),
full FT lr 5e-5 × 3500. Exposure audit (`data/run3/exposure_explicit.json`):
**trace statements per fact matched B=C at median 4**; C's extra rows are
answer-only pads (the treatment itself); A states nothing explicitly (that IS
flashcard format). Predictions pre-registered in `phase3_predictions.md`
before any number was read.

## Scoreboard vs pre-registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Transfer C≈B≫A | ✓ 42% / 33% / 12% (magnitudes lower than guessed — bigger world is harder) |
| 2 | Recall C≈A≫B | ✓✓ **C 60% > A 34% ≫ B 13%** — C beat flashcards at recall |
| 3 | Reversals ~0 | not measured (eval `--limit 300` cut reversal rows; twice confirmed in runs 1-2) |
| 4 | C gates | ✓ multi-hop chain-rate **49% on transfer vs 10% on recall** |
| 5 | B does not gate | ✗ B also gated (41% vs 6%) — but failed recall anyway (13%); B's failure mode changed from run2's over-chaining to plain interference |
| 6 | A never chains | ✓ (16% / 5%) |
| 7-9 | Gagged eval dissociation | **INVALID — manipulation check failed** (below) |
| 10 | C transfer survives exposure stratification | ✓ matched high-exposure bin: C 48% vs B 34% (n=80 each) |
| 11 | hop_control: B,C ≥9/12, A chance | ✗ noisy: B 8/12, C 5/12, A 6/12 — run2's clean B=10/12 did not replicate at 130 entities |
| 12 | No architecture change | ✓ ignition still linear everywhere (slopes 1.2-1.8) |

## The main positive result

The chain-rate cross-tab is the finding:

| block | multi-hop on transfer | multi-hop on recall | acc transfer / recall |
|---|---|---|---|
| A | 16% | 5% | 12% / 34% |
| B | 41% | 6% | 33% / 13% |
| **C** | **49%** | **10%** | **42% / 60%** |

Condition C produced a model that *reasons where reasoning is needed and looks
up where it isn't* — and executes both modes better than the single-format
blocks. In J-space terms: the groove is not just deep, it is **conditional** —
the task distribution can install a routing policy, not merely a pathway.
With the exposure stratification holding (pred. 10), this is not a
more-practice artifact.

## Honest negatives and opens

1. **Gagged eval invalid, question open.** All three gag variants (append
   " The answer is"; append " So the answer is"; bare 4-token budget) collapsed
   every block including never-chaining A — and inspection shows why: on any
   unseen phrasing the model opens with garbage tokens and only surfaces the
   answer ~15 tokens in. The gags measured format brittleness, not knowledge
   location. *Where the groove lives (weights vs page-procedure) is still
   unresolved.* Fix for Phase 4: bake the gag into the treatment — train C
   with an explicit "Answer: X" direct format so an in-distribution
   commitment cue exists at eval time.
2. **Readability noise.** hop_control at 130 entities: B 8/12, C 5/12 — n=12
   with first-subword scoring is too blunt at this scale. Needs more chains
   and multi-token scoring before concluding anything about C's internals.
3. **Architecture: three strikes.** No ignition after run1-LoRA, run2-full,
   run3-mixed. Whatever these conditions build, they build it *within* the
   existing smooth processing — 0.6B × thousands of examples does not
   reorganize the machine. One suggestive crumb: run3's lens-agreement
   profile is the crispest two-stage shape yet (flat ~5% to L18 → 83% at
   L26 vs base 64%), and wikitext mixing did cure run2's narrowing artifacts.
4. **Scoring softness.** "Strict" on bare-format answers = first
   period-delimited span; on OOD phrasings the right name can surface amid
   noise. All blocks scored identically so relative comparisons stand, but
   absolute accuracies are soft.

## Where the program stands

Forced so far: workspace **content** (Phase 2, nulled), pathway depth by
exposure **shape** (Phases 1-3), and now a **trainable gate** (Phase 3).
Not forced: workspace **architecture** (ignition/bands/causal privilege).
Next levers, in order: (a) Phase 4 gag-by-design + more entities per cell to
firm up mechanism and readability; (b) Quiet-STaR-style thought-utility loss
(C-full) — the remaining untested route to architecture; (c) Qwen3-1.7B, same
protocol, where precursors should be stronger.
