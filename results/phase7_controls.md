# Phase 7A controls — results and verdict (2026-07-18)

Scored against `phase7_controls_predictions.md` (written before any control
ran). Chance floor for LL = 1/n_cands, shown per cell.

## Numbers

| control | measure | result | chance | prediction | scored |
|---|---|---|---|---|---|
| C1-rw | LL 1-hop | **100%** (28/28) | 1/24 | ≥70% | ✓ confirmed |
| C1-rw | LL 2-hop | **80.4%** (37/46) | 1/37 | 45-65% | ✓ passed threshold (above band) |
| C1-ic | LL reading | **100%** (20/20) | 1/14 | ≥60% | ✓ confirmed |
| C1-ic | LL chains | **85%** (34/40) | 1/20 | 15-40% | **✗ KILL CONDITION FIRED** |
| C2-st | LL reading / chains | 85% / 55% | 1/9 | — | (aux) |
| C2-st | gen free reading / chains | 100% / 62.5% | — | ~100% / ≥50% | ✓ confirmed |
| C2-st | gen GAGGED reading | **0%** (0/20) | — | ≥50% recovery | ✗ refuted |
| C2-st | gen gagged chains | **0%** (0/40) | — | 10-30% | ✗ refuted |
| C4-rare | gen gagged 2-hop | **82.1%** (23/28) | — | 25-45% | ✓ passed threshold (above band) |
| C4-rare | LL 2-hop | 92.9% (26/28) | 1/19 | — | (aux) |

## Decision rule applied (pre-registered verbatim)

Dissociation asserted ONLY if C1-rw 2-hop ≥45% ✓ AND C4 ≥25% ✓ AND
(C1-ic chains AND C2 gagged chains both below half their rw/free
counterparts). C1-ic chains 85% vs counterpart C1-rw 2-hop 80.4% — **not
below half; ABOVE it.** Rule fails → report the artifact story.

## The artifact story (what the baseline actually showed)

1. **Composition over page content is fine.** Rank the answers by
   likelihood instead of demanding generation, and in-context chains hit
   85% — as good as weights-content 2-hop (80.4%). There is **no
   page→space deficit** at 1.7B.
2. **The 0% gagged cells are a cue-context interaction.** After a
   fact-page, " Answer:" makes the base model continue the page's sentence
   pattern instead of committing — even with real single-token answers
   (C2-st gagged reading 0/20 while its LL reading is 85% and free
   generation 100%). The cue works without a page (rw gagged 79%/54%/82%)
   and fails after one. Production, not knowledge.
3. **C4 caveat (honest note):** rare-direction 82% passes the threshold,
   but capital→currency co-occurrence ("Paris ... euro") is high in text,
   so C4 does not cleanly separate composition from co-occurrence. LL-rw
   2-hop at 80.4% over 37 candidates is the stronger composition evidence.

**Retired claim:** "proto-workspace for weights-content, none for
page-content." The reviewer was right; the controls did their job.

**Surviving reframe:** the model's latent content is present and correctly
ranked (LL 80-100% everywhere), and free production from pages works
(62.5-100%); what is missing is **commitment on demand** — producing the
held answer when cued, in-context. That is workspace property #3
(availability on demand), not property #1 (existence of the content).

## Pre-Phase-B amendments (declared NOW, before any training)

1. **Gagged in-context cells are format-confounded as evidence.** The
   Holding protocol trains the " Answer:" format directly, so post-training
   jumps from 0% → high on gagged page cells are expected from format
   learning alone and are NOT evidence of workspace creation. B-predictions
   that lean on those cells (B5 family) are reinterpreted accordingly.
2. **What still counts as space evidence after training:** (a) held-out
   silent composition on NEVER-trained families, (b) probe reports
   consistent with answers (incl. the wrong-answer faking control and the
   18.30-32 rubric), (c) lens readability of never-emitted intermediates at
   the silent answer moment (C3), (d) LL headroom: baseline LL is already
   80-100%, so LL gains cannot be claimed; LL *stability* is a
   preservation gauge instead.
3. **C3 is now load-bearing**: with behavior format-confounded, the lens at
   the silent answer moment is the primary adjudicator of whether a space
   exists before vs after training.
