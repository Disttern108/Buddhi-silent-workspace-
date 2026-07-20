# Phase 7 (run7, C-full v3) — pre-registered predictions

Written 2026-07-18 BEFORE any Qwen3-1.7B-Base number was read. New brain,
first use in the program; all comparisons to 0.6B cite frozen results/ files.
Training = 4-stage Holding-protocol curriculum (LoRA r=16, lr 1e-4) on real
content only. Scoring rules fixed in advance: strict scorer of eval_model.py;
gag = append " Answer:" + 4-token budget (in-protocol cue, validity
established in Phase 4).

## A. Baseline predictions (fresh 1.7B-Base, before any training)

- **A1 real-world 1-hop free ≥ 65%** (0.6B: 57%); gagged within 15 pts of
  free (1-hop needs no page loop — Phase 5 pattern).
- **A2 real-world 2-hop free 65-85%** (0.6B: 61%); **gagged ≤ half of free**
  — page-dependence is not a 0.6B quirk. If gagged 2-hop is ALREADY near
  free, capacity was the whole story and the curriculum's job is easier.
- **A3 in-context reading ~100%** (0.6B: 100%); **in-context chains free
  40-70%** (0.6B: 32.5%); gagged chains ≤ 15% (0.6B: 5%).
- **A4 detection battery: still no workspace.** No ignition bimodality, no
  three-block CKA structure; output ramp single and late. 1.7B is ~40× below
  the scale where Anthropic found the signatures.
- **A5 held-out protocol items (never-trained baseline)**: F1 2-hop free
  roughly tracks A2; F2 two-step arithmetic free 30-60%; both gagged near
  their 2-hop gagged floor. F4 lookups behave like 1-hop.

## B. Post-training predictions (the bets that matter)

- **B1 S1 format acquisition is easy**: free held-out ≥ 70% after S1
  (Phase 4 showed terminal-format consistency alone is a large lift).
- **B2 THE decisive bet — internalization**: after S3/S4, gagged held-out
  F1/F2 ≥ 25% absolute AND ≥ 2× this brain's own gagged baseline.
  Composition moved inside = the first non-page workspace result of the
  program. Honest prior: uncertain (~40%); at 0.6B nothing moved inside.
  Partial pattern (F2 arithmetic internalizes, F1 entity-hops don't, or
  vice versa) counts as a localized result, predicted more likely than
  all-or-nothing.
- **B3 silent reportability**: on held-out items answered correctly with no
  visible Holding, probe question recovers the true intermediate ≥ 60%.
  **Faking control**: on items answered WRONG, reported holdings should be
  inconsistent/wrong — if probe accuracy is high-and-fluent regardless of
  answer correctness, the probe head is a faker and B3 is void.
- **B4 preservation (hard gates, vs THIS brain's Phase-A numbers)**:
  real-world 1-hop ≥ half its own baseline; in-context reading ≥ 70%.
  LoRA + 45% wikitext should hold these (run4's collapse was full-FT with
  thin mixing). If these fail, everything else is moot — run4's lesson.
- **B5 generalization of the space**: fresh-world in-context chains (free)
  after S4 ≥ its own baseline (no run4-style fusion regression); in-context
  chains GAGGED improve ≥ 2× over its own gagged baseline if B2 holds
  (the space should work on prompt-supplied content, not just weights
  content).
- **B6 lens**: if B2 holds, hop_control on silently-answered items shows
  intermediates readable mid-layer above matched controls (never emitted on
  the page). Ignition still predicted ABSENT (architecture signatures are
  the highest bar; curriculum shapes content, not necessarily dynamics).

## Falsification lines

- B2 fails + B4 holds → curriculum insufficient at 1.7B; next lever is Fast
  Quiet-STaR RL (pre-declared), then 4B.
- B4 fails → redesign mixing before interpreting anything else.
- B3 passes but faking control fails → drop reportability claims entirely.
