# run8 — route-reward RL: results (2026-07-19)

Scored against `phase8_predictions.md` (written before the first RL step).
300 GRPO-style steps on LoRA from run7_s4; reward = fruit + sattvic bonus
− tamasic inconsistency − silence leak (2.47 / 18.30-32 operationalized).
One tooling crash (mx.utils → mlx.utils) cost 50 unsaved steps; relaunch
clean.

## Scorecard

- **P1 PASSED: held-out f2 gagged 26.7% → 36.7%** (11/30; bar ≥35%).
  Free identical (36.7%) — silence discipline retained. Full trajectory of
  the cleanest cell: ~0% (base) → 26.7% (S3/S4 curriculum) → 36.7% (RL) =
  5.5× the cued baseline floor.
- **P2 PASSED: preservation green.** f1 held-out 92.4% (≥85% ✓); rw gagged
  82.1%/69.6% (gates ≥39.3% ✓; 1-hop dipped 89→82 — mild RL cost, noted);
  ic reading 95% (≥70% ✓); ic chains 70% (above 62.5% baseline ✓).
- **P3 HELD (flat, not improved):** inconsistency did not rise (f1
  wrong+true-report 4→3; f2 pattern unchanged). f2 probe remains VOID for
  reportability per run7's faking rule (18/19 true products on wrong
  answers — recomputation confound persists; RL did not create it and did
  not cure it). f2 sattvic 11/11 on correct answers.
- P4: not re-measured (P1 landed below the >45% trigger).
- No abort line fired.

## Notes

- Greedy telemetry drifted negative (−0.12→−0.50) while true held-out
  accuracy rose — telemetry samples no report, so its consistency penalty
  fires by construction; trust the evals, fix the telemetry next time.
- Artifacts: adapters/run8_rl (+history.json), results/heldout_rl_*.json,
  rw_rl_gag.json, ic_rl_free.json, probe_rl.json.
- Remaining pre-declared levers: workspace register tokens, then 4B.
