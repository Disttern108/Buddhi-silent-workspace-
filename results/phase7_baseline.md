# Phase 7A — Qwen3-1.7B-Base baseline (the "before" of run7)

Evals run 2026-07-18 immediately after pre-registration; scored against
`phase7_predictions.md` §A. Strict scorer; gag = " Answer:" + 4 tokens.
Lens fit in progress (A4 pending).

## Numbers (strict)

| battery | mode | 1-hop / reading | 2-hop / chains |
|---|---|---|---|
| real-world | free | 32.1% (9/28) | 47.8% (22/46) |
| real-world | gagged | **78.6%** (22/28) | **54.3%** (25/46) |
| in-context (fresh world) | free | 100% (20/20) | 62.5% (25/40) |
| in-context | gagged | 10% (2/20) | **0%** (0/40) |

0.6B reference (frozen, results/rw_*.json, ic_*.json): rw 1-hop 57→46,
rw 2-hop 61→11, ic reading 100→25, ic chains 32.5→5.

## Scored predictions

- **A1 REFUTED (direction unexpected)**: free 1-hop 32% << predicted 65%,
  but gagged 79% — the knowledge is there; free mode rambles without an
  answer cue and strict first-sentence scoring punishes it. The " Answer:"
  cue is a large positive treatment at 1.7B (Phase-4 format effect, again).
- **A2 REFUTED, the big surprise**: real-world 2-hop gagged (54.3%) ≥ free
  (47.8%), nowhere near the predicted ≤half collapse. **At 1.7B,
  page-dependence for WEIGHTS-content composition is essentially gone**
  (0.6B collapsed 61→11). Capacity was a binding constraint.
- **A3 CONFIRMED all three**: reading 100%; chains free 62.5% (predicted
  40-70); gagged chains 0% (predicted ≤15).
- **The dissociation that now defines run7's job**: silent composition
  WORKS for facts in the weights (54%) and FAILS TOTALLY for facts on the
  page (0%; even gagged in-context reading is 10%). The 1.7B brain has a
  proto-workspace for what it already knows, and none for what it is
  currently seeing. The curriculum's real target is therefore the
  page→space route (F3 family + B5 bet), not just weights→space.

## Post-review artifact findings (2026-07-18, from stored generations)

Peer review flagged the free column and the 0% cell. Both flags verified:

1. **Free column = compliance, not knowledge.** The base model continues
   instead of answering: "What is the currency of France?" -> a list of
   MORE currency questions, "euro" absent from all 80 tokens (19/28 1-hop
   misses contain the answer nowhere). The 1-hop<2-hop inversion (32%<48%)
   is this artifact: 2-hop phrasing happens to trigger answering more often.
   All free-vs-gag comparisons above are contaminated; knowledge claims
   must rest on cued/LL measures.
2. **Gagged in-context 0% is NOT token length.** Restricted to items with
   <=2-token answers: still 0/22. Generations show the real failure: after
   " Answer:" the model continues the page's sentence pattern ("The
   currency of the") — no commitment. Whether that is a true page->space
   deficit or a cue-validity failure on fact-page contexts is exactly what
   controls C1/C2 decide.

**The headline dissociation is therefore DOWNGRADED to provisional** pending
controls C1-C4 (`phase7_controls_predictions.md`, with decision rule).

**2026-07-18 UPDATE: controls ran; the kill condition fired (C1-ic LL
chains 85% ≈ C1-rw 80.4%). Dissociation RETIRED — see
`phase7_controls.md` for the artifact story and the surviving reframe
(missing piece = commitment on demand, not latent content).**

## A4 + C3 — lens detection battery (2026-07-19, 30-prompt lens)

Lens: refit at 30 prompts / dim_batch 8 (50-prompt recipe was memory-bound
at 1.7B float32; converged, final mean rel-change 3.7%).

- **A4 CONFIRMED — still no workspace architecture.** No mid-layer
  ignition: the sharpest interpolation steps sit at L25-28 (max step .36-.39
  vs .1 linear) — a late output commitment, not an ignition band. CKA is a
  soft gradient (diagonal thirds .93/.98/.95; early-late off-block .75),
  not three crisp blocks. Lens-output agreement is a single late ramp
  (≈0 through L18 → 18% L21 → 61% L26).
- Descriptive baselines for the after-portrait: effective dim peaks L24
  (~259) then collapses to 30 at L26; positional autocorrelation of lens
  content peaks mid-stack (L15-18).
- Real-world 2-hop completions: intermediates lens-readable from L8 on
  (median best rank 19 at L8, 13 at L12).
- **C3 (lens at the silent answer moment, gagged rw 2-hop): intermediate
  rank-1.** Beats absent-from-prompt matched controls 21/22 on successes
  and 18/18 on failures (median mid rank 1 both; ctrl 41 / 128). The first
  hop is ALWAYS computed and readable; readability does not separate
  success from failure — commitment does. (Caveat: strong landmark→country
  surface association contributes; C3 shows the intermediate is
  represented, not that a workspace mechanism exists.)
- **Page content: no selection.** In-context chains — true intermediate
  rank ~1, but so is every other page entity: vs on-page controls the
  intermediate wins 0/5 (both rank 0-2); vs off-page controls 7/7
  (ctrl 119-1043). The page is held wholesale; the missing operation is
  SELECTION of the relevant item — exactly F3's training target.

**Amendment (pre-training): B6 gains a page-selectivity metric** — after
training, on in-context chains at the silent answer moment, the true
intermediate must beat matched ON-PAGE controls (the hard null; baseline
0/5 ≈ chance-or-losing). Instrument: hop_control with on-page control
sampling, larger n.

## Amendments (declared now, before any post-training numbers)

1. **Preservation gauge B4 anchors to the cued baseline** (the artifact-free
   measurement): real-world 1-hop gate = ≥ 39.3% (half of 78.6% cued), not
   half of the rambling 32.1%. In-context reading gate stays ≥70% free.
2. **Adopt the three-way output rubric for Phase C** (sattvic/rajasic/
   tamasic per Gita 18.30-32, user's framework): (i) right answer +
   consistent true holding report; (ii) engaged-but-wrong (visible or
   reported holding, wrong answer); (iii) confidently inverted (fluent
   wrong answer + fluent fabricated holding — the report-faking signature).
   This is the B3 faking control made into a full classification of every
   held-out item.
