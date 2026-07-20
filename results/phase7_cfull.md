# run7 (C-full v3) — Final report: forcing a holding space in Qwen3-1.7B-Base

2026-07-19. All numbers scored against pre-registrations written before the
data existed (`phase7_predictions.md`, `phase7_controls_predictions.md`,
`phase7_astro_predictions.md`, amendments in `phase7_baseline.md` /
`phase7_phaseB_log.md`). Model: Qwen3-1.7B-Base + LoRA r=16 (q/v, 16
layers), 4-stage Holding curriculum, real content only, ~6k iters total.

## The result in one paragraph

The curriculum created a **functional holding space**: after training, the
model performs silent composition it could not perform before — most
decisively **two-step arithmetic at 26.7% gagged vs ~0% baseline (4× its
6.7% cued floor), on never-trained problems with nothing on the page** —
and the free/gagged gap closed to zero across every battery, including a
never-trained astrology vocabulary (gagged chains 0%→50%). The lens finds
the silent intermediates readable at the answer moment and page-selection
emerging (0/5→3/5 vs on-page nulls). What did NOT form is the workspace
**architecture**: no ignition, no CKA blocks, unchanged readout ramp —
exactly as predicted (B6). Training relocated computation inward; it did
not reorganize the dynamics.

## Scorecard (every pre-registered bet)

| bet | criterion | outcome |
|---|---|---|
| A1 baseline free ≥65% | — | REFUTED (32%; free = compliance artifact, verified from generations) |
| A2 gagged 2-hop collapse | ≤½ free | REFUTED (54.3% ≥ free; capacity was the 0.6B binding constraint) |
| A3 in-context pattern | 3 bands | confirmed |
| A4 no workspace architecture | — | confirmed (no ignition/blocks; late single ramp) |
| A5 held-out baseline | tracks A2 | confirmed; f2 ≈ 0% everywhere (the clean family) |
| C1-C4 controls | decision rule | **kill condition fired — baseline "page→space deficit" claim RETIRED as production artifact** (LL: ic chains 85% ≈ rw 80%) |
| C3 baseline lens | mid > ctrl ≥70% | confirmed 95-100%, rank 1 — but equally on failures: commitment, not content, was missing |
| B1 S1 format ≥70% free | after root-cause fix | **95.8%** (v1/v2 failed: mlx chat-template landmine, documented) |
| B2 internalization (decisive) | gagged ≥25% abs AND ≥2× base | **F2 PASS: 26.7%, 4.0×**. F1 90.9% clears abs; its 2× was arithmetically impossible from 72.7% (criterion flaw, reported) |
| B3 silent reportability ≥60% | + faking control | f1 PASS 90% (54/60). **f2 VOID by own rule**: true-product reports 91% even on wrong answers — probe-time recomputation indistinguishable (reviewer's predicted failure mode) |
| B4 preservation gates | rw ≥39.3%, ic ≥70% | all green at every stage; rw 2-hop gagged IMPROVED 54→72% |
| B5 space generalizes | ic ≥ base; gagged ≥2× | ic chains free 62.5→72.5%, gagged 0→72.5%; **astro (never-trained vocab): gagged reading 10→95%, chains 0→50%** (bars: 50/25) |
| B6 lens after | intermediates readable; ignition absent | both confirmed: C3 28/30 (med rank 3 vs ctrl 125); page-selectivity 0/5→3/5 (low n, suggestive); ignition/CKA unchanged |

## The before/after grid (strict %, gagged = " Answer:" + 4 tokens)

| battery | base free | base gag | S4 free | S4 gag |
|---|---|---|---|---|
| held-out f1 real 2-hop | 61 | 73 | 91 | **91** |
| held-out f2 arithmetic | 0 | 7 | 27 | **27** |
| held-out f3 page selection | 100 | 33 | 100 | **100** |
| held-out f4 lookups | 21* | 71 | 83 | 83 |
| real-world 1-hop / 2-hop | 32*/48* | 79/54 | 89/72 | 89/**72** |
| in-context read / chains | 100/62.5 | 10/0 | 95/72.5 | 95/**72.5** |
| astro read / chains | 70/20 | 10/0 | 95/50 | **95/50** |

*compliance-artifact cells (model rambled; artifact eliminated by training
— every S4 answer is "Answer: X" + stop).

## What the experiment actually taught us

1. **The three missing operations were commitment, selection, arithmetic —
   not latent content.** The controls killed our own favorite baseline
   claim: composition-content was present all along (LL 80-100%); what the
   base model lacked was committing it on demand. The curriculum installed
   commitment (everywhere), selection (behaviorally 33→100% on f3;
   lens-level 0/5→3/5), and partially the silent arithmetic itself (0→27%).
2. **Silence was the active ingredient.** Arithmetic stayed at 0% gagged
   through S1 (visible holding) and S2 (compressed holding); it lifted only
   at S3 when the holding was deleted and the report still demanded —
   the counterfactual-reflection lever behaving as designed. S2's honest
   failure mode (correct −c applied to a wrong held product) showed the
   space→answer route forms before the silent computation that feeds it.
3. **Functional space ≠ architectural workspace.** No ignition bimodality,
   no CKA reorganization, readout ramp unchanged. At 1.7B with LoRA-scale
   pressure, behavior reorganized; dynamics did not. The paper-grade
   signatures remain a scale/architecture phenomenon, not a curriculum one.
4. **Reportability is real but not fully verifiable behaviorally.** f1
   reports are 90% true when right; f2's probe is void by our own faking
   rule (fluent true products regardless of answer) — the lens, not the
   probe, carries the silent-content claim, and it does (28/30).
5. **Tooling truths, paid in dead stages**: mlx's completions format
   chat-templates base models silently (S1-v1/v2 died of it); token-share,
   not row-share, is what a data mix means; the free column of any base
   model measures compliance first.

## Scope limits and next levers (pre-declared order)

- n is small everywhere (30-66/cell); selectivity n=5. Directionally
  consistent across seven independent batteries, but confidence intervals
  are wide.
- f2 at 27% is far from ceiling — the silent multiply remains the
  bottleneck. Next levers, in order: Quiet-STaR-style route-reward RL
  (2.47: reward the process), workspace register tokens (an architectural
  seat), 4B scale.
- The 18.30-32 rubric found **zero tamasic items** (no fluent-wrong-answer
  + fabricated-holding pairs) in inspected cells; f2's void probe means
  fabrication cannot be ruled out there by behavior alone.

Artifacts: adapters/run7_s{1..4}, models/run7_s4_fused,
results/run7-s4/ (lens + battery), all eval JSONs under results/.
