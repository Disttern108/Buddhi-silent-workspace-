# Phase 3 — pre-registered predictions

Written 2026-07-18, BEFORE any run3 eval/probe number was seen (run3 chain was
mid-flight: training done, eval not yet read). Exposure audit done first —
`data/run3/exposure_explicit.json`: trace statements per fact matched B=C
(median 4); C adds 920 answer-only pads (the treatment); A has 0 explicit
statements (flashcard format states nothing).

## Behavioral (eval_run3.json, strict scoring)

1. **Transfer (novel combinations)**: C ≈ B ≫ A. Both trace-fed blocks near
   55-65%; A near 25-35%.
2. **Recall (unseen templates)**: C ≈ A ≫ B. C avoids B's over-serialization
   collapse: C ≥ 40%, B ≤ 25%.
3. **Reversals**: ~0 everywhere, all blocks (format-independent control).

## Gating (chain-rate cross-tab, phase3_analysis.py)

4. **C gates**: multi-hop chain-rate high (>60%) on chain-shaped transfer
   questions, low (<30%) on direct recall questions.
5. **B does not gate**: similar multi-hop chain-rate on both question types.
6. **A never chains** (<10% everywhere — it was never shown traces).
   If C's rates are similar across question types, "learned when to reason"
   is REJECTED even if C's accuracy table looks good (averaging, not gating).

## Gagged eval (suffix " The answer is", max 8 tokens)

7. **B recall improves gagged** (over-serialization was the failure mode:
   forcing commitment should recover the fact it states first).
8. **Transfer gagged**: if the reasoning groove is weight-level, B and C keep
   a meaningful fraction (>half) of their ungagged transfer; if it is a
   page-procedure (external loop), gagged transfer collapses toward A's level.
   No strong prior between these — this is the genuinely open question.
9. **A roughly unchanged** gagged (it never chained anyway).

## Exposure stratification

10. Within matched exposure bins, C's transfer advantage over A persists.
    If C > B appears, it must survive stratification to count; C ≈ B is the
    expected and sufficient outcome (the C claim is about recall + gating,
    not beating B at transfer).

## Workspace probes (battery, hop_control on run3-full)

11. **hop_control**: B and C both beat matched controls (≥9/12); A at chance.
12. **Ignition/CKA**: still no architecture change (sadly expected at 0.6B;
    a positive here would be a major surprise).
