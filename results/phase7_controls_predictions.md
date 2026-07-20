# Phase 7A controls — pre-registered predictions (post-review)

Written 2026-07-18 after the peer review of phase7_baseline.md, BEFORE any
control was run. Two artifacts already confirmed from stored data: (i) free
column = compliance, not knowledge (model continues with more questions;
answer absent from 80 tokens in 19/28 1-hop misses); (ii) gagged in-context
0% is NOT token-length (0/22 even on <=2-token answers) — generations show
pattern-continuation ("The currency of the") instead of commitment.

Instruments: C1 closed-set LL rescore (probe/ll_rescore.py — ranks candidate
answers by logP under prompt + " Answer:"; no generation, immune to all
production artifacts). C2 single-token page battery (data/incontext_st —
fictional subjects, real single-token values). C4 rare-direction real
compositions (data/realworld_rare — "the country whose capital is X").
C3 (lens at the silent answer moment) runs after the lens fit completes.

## Predictions

- **C1-rw**: LL 1-hop >= 70% (knowledge was there; free 32% was compliance).
  LL 2-hop 45-65%, consistent with gagged 54% — the silent-composition
  claim survives rescoring.
- **C1-ic (the money cell)**: LL reading >= 60%. LL chains: intermediate —
  clearly above chance (~1/#cands) but well below LL-rw-2-hop; predict
  15-40%. If instead LL chains ~= LL rw 2-hop, the baseline "page->space
  deficit" was ENTIRELY a production artifact and the dissociation claim
  dies. If ~chance, deficit fully real.
- **C2 (generation, free + gag)**: free reading ~100%, free chains >= 50%
  (easier than invented: single-token real values). Gagged reading recovers
  substantially vs 10% (predict >= 50%) — the cue commits more readily to
  common tokens. Gagged chains recover only partially (predict 10-30%,
  still far below free): production artifact explains part of the 0%,
  page->space deficit explains the rest.
- **C4 (gag + LL)**: rare-direction gagged accuracy BELOW the standard
  battery's 54% (predict 25-45%). Some of the "silent 2-hop" was memorized
  co-occurrence; a real proto-workspace residue should survive above the
  1-hop-free floor. If rare drops to ~single digits, "silent composition
  over weights" was co-occurrence and the proto-workspace claim retracts.

## C3 addendum (written 2026-07-19, after lens fit, BEFORE C3 ran)

Instrument: `probe/c3_silent_lens.py` — for each gagged rw 2-hop item
(prompt + " Answer:"), best lens rank over the last 6 positions × all
source layers of (a) the latent intermediate country's distinctive first
token vs (b) a matched control country from the same table, never in the
chain (fixed rng; UK/US excluded — shared " United" first token).
hop_control.py's exact rank logic, applied at the silent answer moment.

Predictions: on the 25 gagged SUCCESSES, intermediate beats its matched
control in ≥70% of items, median intermediate rank ≤50. Secondary
(failures, n=21): if intermediates are equally readable on failures,
readability is not the binding constraint on production. Kill direction:
if intermediate ≈ control on successes, the silent 2-hop answers are
reached WITHOUT a represented intermediate — the co-occurrence-shortcut
reading strengthens (with the C4 caveat) and "silent composition over
weights" weakens to "direct association".

## Decision rule (stated in advance)

The baseline dissociation ("proto-workspace for weights-content, none for
page-content") is asserted in the final report ONLY if: C1-rw 2-hop >= 45%
AND C4 >= 25% AND (C1-ic chains + C2 gagged chains both land below half of
their rw/free counterparts). Any other pattern: report the artifact story.
