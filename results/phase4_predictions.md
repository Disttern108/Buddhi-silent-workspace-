# Phase 4 — pre-registered predictions (gag-by-design)

Written 2026-07-18 BEFORE run4 training/eval. Change vs run3: single format
change — every condition's completions end with the universal cue "Answer: X".
Gag = append " Answer:" to the prompt, max 4 tokens. Everything else identical
to run3 (same world knobs, seed, training recipe, wikitext mixing).

## Validity gate (must pass before anything else is interpreted)

1. **Block A gagged recall within 10 points of A ungagged recall.** A never
   chains; a valid gag costs it nothing. If this fails, the gag is still
   invalid and no mechanism conclusion may be drawn (as in run3's three
   failed gag variants).

## Replication under the new format

2. Ungagged run4 reproduces run3's pattern: C best on both axes; gating
   cross-tab (C multi-hop high on transfer, low on recall).

## Mechanism (the point of Phase 4) — gagged TRANSFER, conditional on gate 1

3. **Page-loop hypothesis**: B's gagged transfer collapses to ≤ half its
   ungagged value (the chain needs the page; no room to write → no answer).
4. **Weights hypothesis**: gagged transfer retains > half of ungagged.
5. No strong prior for B. For C, weak lean toward *partial retention*: the
   direct-QA component may consolidate some chains into weights. If C retains
   gagged transfer and B collapses, the mixed regime moved knowledge into the
   weights — the strongest possible Phase 4 outcome. If both collapse, the
   "workspace" these formats build is the page (external loop), and internal
   consolidation needs Quiet-STaR-style training (C-full).
6. Gagged RECALL: all blocks should hold near ungagged recall (recall never
   needed the page). B's recall may IMPROVE gagged (commitment cue rescues it
   from wandering into chains).
