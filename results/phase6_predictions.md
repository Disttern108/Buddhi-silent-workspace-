# Phase 6 — pre-registered predictions (fresh-world in-context test)

Written 2026-07-18 BEFORE any eval. Question: did fine-tuning install a
TRANSFERABLE reasoning skill, or only world-specific grooves?

Design: fresh invented world, seed 99 (zero entity overlap with training
worlds). Prompts contain the needed facts as trace-style sentences (chain
facts + 5 shuffled distractors), then a chain question in trained syntax.
Controls: in-context direct recall (fact stated in prompt, direct question).
Models × modes: {base, run4} × {free, gagged}. Note this is also the
"pre-written page" condition: the facts are ON the page already — only
self-written derivation is blocked in gagged mode.

1. **run4 free in-context chains: the skill transfers.** Its trained
   procedure is literally "read facts from the page, chain them, commit."
   With facts pre-written, predict ≥40% — far above its 0% real-world and
   above base free.
2. **Base free in-context chains: modest** (10-30%; in-context multi-hop is
   hard for an 0.6B base).
3. **In-context direct recall (control)**: both models decent free (copying a
   stated fact); run4 possibly better (format match). If run4 fails this,
   its skill is not even reading — world-bound token grooves only.
4. **Gagged chains, both models: low** — even with facts pre-written, silent
   composition is the missing capacity. If run4 gagged does WELL here, the
   page-loop claim needs revision (it would mean the loop only needs a
   readable page, not self-written steps).
5. Directional bet: run4 free > base free on chains by ≥15 points = skill
   transferred. run4 ≈ base = the fine-tune taught nothing general.
