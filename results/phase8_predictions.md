# run8 — route-reward RL ("ancient rewards"), pre-registered

Written 2026-07-19 BEFORE any RL step ran. Start point: run7_s4 adapter
(functional holding space; silent f2 arithmetic 26.7% = the bottleneck).
Lever: the pre-declared Quiet-STaR-class route-reward RL, with the reward
translated from the user's framework — Gita 2.47 (the fruit alone is not
the reward; the route earns) and 18.30-32 (sattvic/rajasic/tamasic as
reward classes).

## Algorithm

GRPO-style REINFORCE on LoRA params only (base frozen): per training item,
sample K=4 trajectories (answer at temp 0.9, then probe report); advantage
= reward − group mean; loss = −adv · logp(answer tokens + report tokens);
Adam 1e-5; 300 steps × 2 prompts; checkpoint every 50. Fresh f1/f2 train
items (train anchors only — heldout untouched), gold + intermediate known
to the grader.

## The reward (exact formula)

- **fruit** +1.0: answer == gold. (2.47: necessary, not sufficient)
- **sattvic bonus** +0.5: answer correct AND probe report == true
  intermediate (right act + true self-knowledge).
- **tamasic penalty** −1.0: report does NOT explain the model's own answer
  (f2: answer ≠ the problem's op applied to the reported product; f1:
  reported country's attribute ≠ given answer). Mechanical consistency —
  fabrication is punished without needing gold.
- **silence leak** −0.5: "Holding:" appears in the answer body, or
  unparseable answer (form discipline).
- rajasic (engaged-but-wrong, consistent) = 0: engagement is not punished.

## Predictions

- P1: held-out f2 gagged rises 26.7% → ≥35% (the silent multiply
  strengthens under route pressure). Honest prior ~45%.
- P2: preservation gates stay green (rw ≥39.3% cued 1-hop, ic reading
  ≥70%); held-out f1 stays ≥85%.
- P3: report-answer consistency (the tamasic rate) does NOT rise; the
  faking confound shrinks or holds (reports become MORE explanatory of
  answers, since inconsistency is the only negatively rewarded class).
- P4: no architectural change expected (ignition stays absent) — not
  re-measured unless P1 lands big (>45%).

## Falsification / abort lines (stated in advance)

- f2 gagged ≤26.7% after 300 steps → RL at this scale/recipe insufficient;
  report and stop (next lever: register tokens, then 4B).
- Held-out free drops >10 pts or any preservation gate fails at ANY
  checkpoint eval → revert to run7_s4, report the collapse.
- Entropy collapse to a single wrong mode (all K samples identical AND
  reward flat for 50 steps) → abort, report.
