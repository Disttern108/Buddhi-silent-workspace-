# Phase 4 — Gag-by-design: WHERE does the reasoning groove live?

**Verdict: the groove is split. Atomic facts consolidated into the weights
(gag-proof — recall *improves* when forced to commit). Multi-hop composition
lives on the page (gagged transfer collapses from 80% to A's floor in both
trace blocks). The "workspace" that trace-training builds at 0.6B is
EXTERNAL: the model's own output stream is the workspace.**

Design: run4 = run3 with one change — every condition's completions end with
the universal cue "Answer: X", making an appended " Answer:" an
in-distribution gag (4-token budget). Predictions pre-registered in
`phase4_predictions.md` before training.

## Validity gate — PASSED (predictions 1, 6)

Block A (never chains): gagged recall 27.8% vs ungagged 25.9%. The gag costs
a non-chaining model nothing — unlike run3's three invalid gag variants, this
one measures knowledge, not format brittleness.

## Results

| block | transfer (ungag → gag) | recall (ungag → gag) |
|---|---|---|
| A flashcard | 10% → 9% (unaffected) | 26% → 28% |
| B trace | **80% → 4.5%** | 27% → **39%** (+13) |
| C mixed | **80% → 7.5%** | 48% → **65%** (+17) |

Cross-tab (multi-hop trace-rate, ungagged): transfer B/C 100%, recall B/C
6-10% — clean gating in all trace-fed blocks under the cue format.
Exposure-matched hi bin: B 82% ≈ C 85% transfer.

## Scored against predictions

- P1 validity gate ✓ · P2 replication ✓ (C best recall; B=C transfer; gating ✓)
- **P3 page-loop hypothesis ✓ decisively** — both trace blocks retain <10% of
  gagged transfer. P4/P5 weights-retention rejected for composition; C's
  direct-QA component did NOT internalize the chains.
- P6 ✓ exceeded: gagging *helps* recall in trace blocks (+13/+17) — the
  commitment cue rescues answers from wandering into unnecessary chains.

## Side discovery: the format effect

The universal "Answer: X" cue alone (only change vs run3) lifted trace-block
transfer from 33-42% → 80%, fixed B's recall collapse (13% → 27%), and made
gating universal. At this scale, terminal-format consistency is worth more
than any knob we've swept. Caveat: run4 transfer chains can re-derive from the
page, so the 80% is page-loop performance, as the gag proves.

## What this means for the J-space program

The full mechanism picture at 0.6B, four runs in:

1. **Weights hold atoms.** Single facts consolidate internally regardless of
   format; forcing commitment retrieves them *better* (external chaining was
   interference, not access).
2. **The page holds composition.** Trace training installs a serial
   *procedure* — read question, write hop, read own output, write next hop.
   Working memory is the context window, not the residual stream. This
   retroactively explains every internal negative: no ignition, no causal
   handles, swap failures — there was no internal loop to grab.
3. **The gate is real and internal** — the decision *whether* to run the page
   loop is made before any token is written, and it is trainable (Phase 3-4).

So task-distribution conditioning at 0.6B builds: internal fact grooves +
an internal routing gate + an EXTERNAL reasoning loop. What it does not
build is internal composition — the actual silent workspace. That is now a
precise target, and it is exactly what Quiet-STaR-style training claims to
internalize (thoughts generated then *distilled away* — Fast Quiet-STaR).
C-full is no longer optional: it is the experiment that tests whether the
page loop can be pulled inside the model. Second lever: scale (1.7B), where
latent multi-hop composition is known to be stronger.
