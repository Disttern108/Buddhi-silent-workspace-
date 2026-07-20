# Phase 7B — stage log (gates enforced)

## A5 baseline (untrained, held-out protocol battery, 2026-07-19)

| block | free | gagged |
|---|---|---|
| f1 real 2-hop (held-out countries) | 60.6% | 72.7% |
| f2 two-step arithmetic | 0% | 6.7% |
| f3 selection pages | 100% | 33.3% |
| f4 lookups | 20.8% (compliance artifact) | 70.8% |

A5 scored: f1 tracks A2's cued band as predicted; f2 near-zero everywhere
(the genuinely hard family — arithmetic composition is this brain's real
deficit); f3/f4 free numbers carry the known compliance artifact.

## S1-v1 (1500 iters, 45%-row wikitext mix) — GATE B1 FAILED

- Training healthy: loss 2.6→2.1, 12 min, peak 9.6GB.
- **Format NOT acquired**: free generations contain "Holding" 1/144,
  "Answer:" 4/144 (base: 0/144). B1 (free ≥70%) fails at 34% overall.
  19/46 f1 free misses are holding-flavored fragments without an answer
  ("The country is Ireland.") — the direction started, the template didn't.
- **Content moved anyway (gagged)**: f1 72.7→89.4%, f3 33→100%,
  f4 70.8→95.8%, f2 6.7→10%. Preservation IMPROVED: rw gagged 1-hop
  78.6→96.4%, 2-hop 54.3→69.6%.
- **Root cause (measured)**: wikitext rows ~130 tokens vs ~35 for protocol
  rows — the "45% mix" by rows was ~70% of TOKENS. Protocol signal diluted.

**Amendment S1-v2 (declared before retraining)**: wikitext passages
truncated to ~240 chars and reduced to 25% of rows (protocol ≈ 2/3 of
tokens); iters 1500→2500; fresh adapter (no resume from v1). Preservation
risk accepted knowingly: v1 showed preservation improving, leaving room to
push protocol exposure; the B4 gates still bind at stage end.

## S1-v2 (2500 iters, rebalanced mix) — GATE B1 FAILED AGAIN, root cause found

- Free: STILL 0/144 "Holding:" mentions; misses are pure base-model
  question-continuation loops. Gagged DROPPED vs v1 (f1 89→79, f3 100→75,
  f4 96→79). ic free reading 65% — below the 70% preservation gate.
- **ROOT CAUSE (both v1 and v2): mlx_lm's CompletionsDataset wraps every
  prompt/completion row in the tokenizer's CHAT TEMPLATE** (`<|im_start|>`
  frames). The format WAS being learned — inside a dialogue frame our
  raw-text evals never enter. Same landmine as the CLI garbage incident,
  training-side. Dilution (v1→v2 fix) was real but secondary.
- **Amendment S1-v3**: data emitted as {"text": prompt+completion} rows —
  mlx's text dataset does raw encode + auto-EOS (no template; EOS also
  teaches stopping, which should clear the free-mode ramble artifact).
  Cost: no prompt masking (loss on question text too) — standard for
  base-model SFT, accepted. Config: mask_prompt removed; iters 2500;
  fresh adapter. v1/v2 gagged/preservation shifts are reinterpreted as
  generic drift from template-framed training, not protocol learning.

## S1-v3 (text rows, 2500 iters) — GATE B1 PASSED 95.8%

- **Format perfect: 144/144 free = "Holding: X. Answer: Y" + stop** (EOS
  learned; ramble artifact gone). Free per family: f1 95.5%, f2 93.3%(!,
  baseline 0%), f3 100%, f4 95.8%. B1 (≥70%) passed decisively.
- Gagged: f1 87.9%, f3 100%, f4 100%, f2 0% — bare answers fail on
  arithmetic without the visible holding ("7" for 47): exactly the
  internalization target for S3/S4.
- **Preservation all green**: rw gagged 92.9%/67.4% (gates ≥39.3%),
  ic free reading 95% (gate ≥70%), ic free chains 70% (baseline 62.5% —
  improved). The protocol helps, not hurts, fresh-world reading.
- Astro baseline banked (untrained): free 70%/20% strict (loose 80%/80% —
  compliance artifact on alien vocab too), gagged 10%/0% as predicted.

## S2 (≤3-word Holding, 1000 iters, resume from S1-v3) — GATE PASSED 79.9%

- Free: f1 95.5%, f3 100%, f4 91.7%, **f2 93→20%**. Gate 79.9% ≥ 57.5% ✓.
- **The f2 drop is the first internalization signal, and it is HONEST:**
  30/30 keep perfect format; the held number is wrong but the second step
  is applied faithfully to it ("Holding: 100. Answer: 97" = correct −3 on
  a bad product; gold 47). Compression forces the multiply inward, the
  silent multiply fails, the space→answer route stays consistent. Rajasic
  class (engaged-but-wrong), zero fabrication observed.
- Gagged: f1 87.9%, f2 0% (unchanged — expected pre-S3).
- Preservation green: rw gagged 92.9%/**73.9%** (2-hop still rising),
  ic reading 95%, ic chains 67.5%.

## S3 (Holding DELETED, probes retained, 1500 iters) — GATE PASSED 80.6%; INTERNALIZATION SIGNAL

(Chain log says "STAGE S2_*" — sed case slip in echo labels only; config/
data/adapters/results verified S3.)

- Free (fully silent now — 0/144 Holding mentions): f1 93.9%, f2 26.7%,
  f3 100%, f4 91.7% → 80.6%, gate ✓.
- **Gagged: f2 0% → 23.3%** — bare "47"/"86" in ≤4 tokens on never-trained
  two-step problems, nothing on the page: the multiply+adjust now happens
  INSIDE. f1 gagged 92.4% (base 72.7), f3 100% (base 33), f4 91.7%.
  **The free/gag gap has essentially closed** at S3.
- Preservation: rw 92.9%/73.9%, ic reading 100%, ic chains 70% — all green.
- B2 status (formal scoring after S4 as registered): f2 at 3.5× its
  baseline but 23.3% < 25% absolute (7/30; one item short). f1 92.4%
  clears absolute easily; its 2× criterion is ceiling-impossible from a
  72.7% baseline — will be reported as such.

## S4 (probes sparse, 1000 iters) — GATE PASSED 77.8%; PHASE B COMPLETE

- **Free == gagged EXACTLY on every family** (f1 90.9%, f2 26.7%, f3 100%,
  f4 83.3% both modes; free format is bare "Answer: X"). The model is
  silent by default — the curriculum's endpoint reached.
- **B2 FORMALLY SCORED (the decisive bet, honest prior ~40%):**
  - **F2: PASS in full** — 26.7% ≥ 25% absolute ✓, 4.0× baseline (6.7%) ✓.
    Silent two-step arithmetic on never-trained problems.
  - F1: 90.9% clears absolute ✓; the ≥2× criterion was arithmetically
    impossible from its 72.7% baseline (criterion flaw, reported as such;
    +18 pts absolute). B2's own text: partial pattern = localized result.
    Actual pattern: F2 internalized fully; F1 rose within its ceiling.
- Preservation after 4 stages: rw gagged 89.3%/71.7% (gates ≥39.3% ✓✓),
  ic reading 95% (≥70% ✓), ic chains 72.5% (baseline 62.5% — ABOVE).
- All four stage gates passed; no stage retrained past a failure.
