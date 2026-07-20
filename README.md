# Verbalizable Workspaces in Small Language Models

### What replicates from the Global-Workspace paper, what is a lens artifact, and whether a working holding-space can be *installed*

This repository contains two linked interpretability projects and one thesis:

1. **An independent audit** ([`tao-hpu/jspace-replication`](https://github.com/tao-hpu/jspace-replication), included here as a submodule) of Anthropic's
   *["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)* (2026),
   re-running its highest-stakes claims from scratch on small open-weight models and
   separating what holds from what is an instrument artifact.
2. **A constructive follow-on** (the **Buddhi** program, this repo's root) that asks:
   if the workspace *architecture* isn't present at small scale, can a curriculum
   install the *function* — silent, internal multi-step composition — anyway?

**The full paper is [`results/FINDINGS.md`](results/FINDINGS.md).** Every headline below
traces to a dated log entry and a results JSON; nothing here is aspirational.

---

## One-line synthesis

> **At 1.7B you can relocate computation *inward* (function) without reorganizing the
> *dynamics* (architecture).** The paper-grade workspace signatures are a
> scale/architecture phenomenon, not something a curriculum manufactures — and several
> of them, read carefully, are properties of the lens rather than the model.

---

## Part I — Auditing the Global-Workspace claims

Qwen3-1.7B/4B with Neuronpedia's pre-fitted Jacobian lenses (plus a 1.7B→27B dense
ladder and Gemma-2 for the capture work). Method rule held throughout: every result
separates three sources — the **paper's claim**, the **external review's verdict**
([Nanda, LessWrong](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)),
and **our own measurement** — never blended; the official `jacobian-lens` code is never
modified (pinned in [`third_party/`](https://github.com/tao-hpu/jspace-replication/tree/master/third_party)).

| Claim | Paper | Review | **Our measurement** |
|---|---|---|---|
| **C1** shared representations (edit one concept, all dependents flip) | ✅ strongest | ✅ replicates | **Replicated, strengthens with scale.** 97.2% hit / 0% stayed at 4B; carried by the early-mid band (layers 4–13). Only *associative* links move; *operations* on the concept fail. |
| **C2** thought-swap (rewrite a two-hop intermediate) | ✅ | ⚠️ weak | **Partial (50–65%), non-surgical.** Along lens directions the answer-token control *beats* the intermediate swap (4B 85.4% vs 50.0%, p=0.0005). Survives with trained probes at reduced magnitude; the "surgical thought edit" reading does not. |
| **C3** rhyme planning | ❌ | ❌ | **Not replicated.** Rhyme word never in lens top-10 at readout (pass@10 = 0%). The "emergence curve" is the word climbing into the model's own next-token distribution. |
| **C4** internal arithmetic, lens-readable | ❌ | ❌ | **Not replicated.** Order/operation readout at/near base rate (1.7B exactly 0 above chance). |
| **C5** lens reliability | "imperfect" | ⚠️ heavy FPs | **Confirmed & quantified.** Permutation control fires 9–11% on multihop; **no consistent advantage over a vanilla logit lens** on its own eval sets. |
| **C6** global-workspace dynamics (ignition, capacity) | claimed | least confident | **Not tested** (by design; not compute-funded). |

**Two phenomena the public review never tested, both real and causal:**
- **Covert registers (E6).** A mass-mean language axis flips output language 91–94%
  (random control 0%, p<1e-4) while content flips stay ~30% on the covert subset —
  *the register carries the setting (language, intended form), not the content.*
- **Perspectival capture (E7) — no prior demonstration we know of.** A mid-band entity
  swap makes the model **restate the question itself** with the swapped entity (1.7B:
  answer flips 78.6%, restatement rewritten 71.4%). **Scale-stable 1.7B→14B.** Mechanism:
  a composite cutoff — whichever comes first of a coordinate *drift wall* and the task's
  *consumption deadline* (commit-layer rank-correlates with the writable-zone cutoff at ρ=0.86).

**What dissolves under a control:** the thought-swap is **directed steering, not surgery**
— swapping a *non-answer* in-context concept flips the final answer about as often as
swapping the answer itself (77.1% vs 85.4%, p=0.29); flips track source↔answer direction
overlap, and an *absent* word ("piano") harvests comparable amplitude and flips **nothing (0%)**.

---

## Part II — Constructing a holding-space ("Buddhi")

A fact-world generator ([`fact_world_generator.py`](fact_world_generator.py)) emits invented
worlds in two matched formats — flashcard (Q→A) vs integration (Q→verbalized working→A) —
with atomic-fact exposure audited equal, so any difference is attributable to *format*, not
exposure. **Every phase's predictions were committed before the data existed; failures were
logged the same day.** The *gag* protocol (append `" Answer:"`, 4-token budget) forces
commitment so the model can't chain out loud — isolating *silent* composition.

**Phases 0–6 (Qwen3-0.6B) — the reasoning loop is *external*.** Gag the model and multi-hop
composition collapses to the floor (80% → ~5%) while single-fact recall *improves* (+13/+17):
the 0.6B "workspace" is the model's own output stream. Content installs; architecture does not.

**Phase 7 (Qwen3-1.7B-Base) — the headline.** A four-stage Holding curriculum (LoRA r=16,
q/v, 16 layers; real content only): **S1** full visible Holding → **S2** ≤3-word Holding →
**S3** Holding deleted, report still demanded → **S4** deleted + sparse probes.

Scorecard against pre-registered bets:

| bet | criterion | outcome |
|---|---|---|
| B1 | S1 free format ≥70% | **95.8%** (after root-causing the mlx chat-template landmine) |
| **B2** | **gagged held-out ≥25% abs AND ≥2× base** | **F2 arithmetic PASS: 26.7%, 4.0× the 6.7% cued floor** — silent two-step on never-trained problems, nothing on the page. |
| B3 | silent reportability ≥60% + faking control | F1 PASS (90% true-when-right); **F2 VOID by our own rule** — 91% true products even on *wrong* answers (probe-time recomputation confound; the reviewer's predicted failure). |
| B4 | preservation gates | all green; real-world 2-hop gagged **improved** 54→72%. |
| B5 | space generalizes | in-context chains gagged 0→72.5%; **astrology (never-trained vocab): gagged reading 10→95%, chains 0→50%.** |
| B6 | intermediates readable; ignition absent | both confirmed: lens C3 28/30 (median rank 3 vs control 125); page-selectivity 0/5→3/5; ignition/CKA unchanged. |

Before → after (strict %, gagged = `" Answer:"` + 4 tokens):

| battery | base free | base gag | S4 free | S4 gag |
|---|---|---|---|---|
| held-out F1 real 2-hop | 61 | 73 | 91 | **91** |
| held-out F2 arithmetic | 0 | 7 | 27 | **27** |
| held-out F3 page selection | 100 | 33 | 100 | **100** |
| real-world 1-hop / 2-hop | 32\* / 48\* | 79 / 54 | 89 / 72 | 89 / **72** |
| in-context read / chains | 100 / 62.5 | 10 / 0 | 95 / 72.5 | 95 / **72.5** |
| astro read / chains | 70 / 20 | 10 / 0 | 95 / 50 | **95 / 50** |

\* compliance-artifact cells (base model rambled; eliminated by training — every S4 answer is `"Answer: X"` then stop).

**Phase 8 — route-reward RL (run8).** A GRPO-style reward operationalizing Gita 2.47 (*the
route earns, not just the fruit*) and the 18.30–32 sattvic/rajasic/tamasic rubric
(`+fruit +sattvic −tamasic-inconsistency −silence-leak`), 300 steps on LoRA from S4:
**F2 gagged 26.7% → 36.7%** (pre-registered bar ≥35%), free identical. Full trajectory of the
cleanest cell: **~0% (base) → 26.7% (curriculum) → 36.7% (RL) = 5.5× the cued floor.**
Preservation green. The F2 report probe remains void by the faking rule.

**What the curriculum installed:** commitment (everywhere), selection (behaviorally 33→100%
on F3; lens-level 0/5→3/5), and *partially the silent arithmetic itself* (0→27→37%).
**What it did not install:** the workspace *architecture* — no ignition bimodality, no CKA
reorganization, readout ramp unchanged — **exactly as the audit predicted.**

---

## Repository layout

```
Buddhi/
├── README.md                  ← you are here (repo front door)
├── results/FINDINGS.md        ← the full consolidated paper
├── results/phase{0..8}*.md    ← per-phase reports + pre-registrations (predictions written first)
├── results/*.json             ← every eval battery, lens output, LL-rescore
│
├── fact_world_generator.py    ← invented-world generator (depth/serial/interference/novelty knobs)
├── eval_model.py              ← battery runner (--suffix " Answer:" gags chaining)
├── chat.py                    ← raw-prompt REPL for the trained models
├── configs/run7_s{1..4}.yaml  ← the 4-stage Holding curriculum (mlx_lm.lora)
├── probe/                     ← data builders + instruments:
│   ├── make_{protocol,realworld,incontext,astro,controls}.py   (batteries)
│   ├── battery.py, fit_lens.py, c3_silent_lens.py, hop_control.py  (lens / structure)
│   ├── ll_rescore.py          (likelihood rescoring — immune to production artifacts)
│   ├── eval_probe.py          (silent reportability + faking control)
│   └── rl_route.py            (phase-8 route-reward RL)
├── adapters/run7_s{1..4}, adapters/run8_rl     ← LoRA weights (tracked)
├── models/run{7_s4,8_rl}_fused                 ← fused 1.7B models (gitignored; rebuild via mlx_lm.fuse)
│
└── jspace-replication/        ← Part I: the audit (own README, docs/, experiments/e0..e7, third_party/)
```

---

## Reproducibility

Two runtimes. **MLX** (`mlx_lm`) drives training / fusing / eval / RL for Part II; **PyTorch +
`transformers` + `jlens`** drive the lens and structural battery for Part I. Base model:
`Qwen3-1.7B-Base` (and `Qwen3-0.6B` for phases 0–6).

**Part II — the Holding curriculum, end to end:**

```bash
# 1. build the batteries (all pre-registered, disjoint from training)
.venv/bin/python probe/make_protocol_data.py     # curriculum + held-out F1..F4
.venv/bin/python probe/make_realworld.py          # real-fact transfer battery
.venv/bin/python probe/make_incontext.py          # fresh-world in-context battery
.venv/bin/python probe/make_astro.py              # never-trained alien vocabulary

# 2. train the 4 stages (S2+ resume from the previous adapter — see resume_adapter_file)
for s in 1 2 3 4; do .venv/bin/mlx_lm.lora -c configs/run7_s${s}.yaml; done

# 3. evaluate free vs gagged (the gag is the whole point)
.venv/bin/python eval_model.py --model ~/Models/Qwen3-1.7B-Base --adapter adapters/run7_s4 \
    --data data/protocol_heldout --out results/heldout_s4_free.json
.venv/bin/python eval_model.py --model ~/Models/Qwen3-1.7B-Base --adapter adapters/run7_s4 \
    --data data/protocol_heldout --out results/heldout_s4_gag.json --suffix " Answer:" --max-tokens 4

# 4. phase-8 route-reward RL from the S4 adapter
.venv/bin/python probe/rl_route.py --steps 300 --out adapters/run8_rl

# 5. talk to either brain (raw prompts — mlx_lm.chat template-poisons base models, use this instead)
.venv/bin/mlx_lm.fuse --model ~/Models/Qwen3-1.7B-Base --adapter-path adapters/run8_rl --save-path models/run8_rl_fused
.venv/bin/python chat.py models/run8_rl_fused
```

**Part I — the audit:** see the [jspace-replication README](https://github.com/tao-hpu/jspace-replication#readme)
and [replication log](https://github.com/tao-hpu/jspace-replication/blob/master/docs/replication-log.md);
the lens is pinned in `third_party/` (`PINNED_COMMIT.txt`), never modified.

### Tooling truths (paid for in dead runs)

- **mlx's completions / prompt+completion format silently applies the tokenizer's chat
  template** to base-model rows — it killed two training stages before diagnosis. Use `{"text": ...}` rows for base models.
- **A data mix means TOKEN-share, not row-share** (45% wikitext by rows was ~70% by tokens).
- **The free-mode accuracy of any base model measures compliance, not knowledge** — always
  cue with `" Answer:"` and cross-check with likelihood rescoring (`probe/ll_rescore.py`).

---

## Rigor & honest limitations

- **Pre-registration throughout** — every construction phase committed numeric predictions,
  decision rules, and falsification lines before the data existed. **Our own controls refuted
  two of our own headline claims** (the "page→space deficit" production artifact; the F2
  reportability probe, voided by the faking rule) and we retired them on the record.
- **Small n everywhere** (30–66 per cell; page-selectivity n=5). Directionally consistent
  across seven independent batteries, but confidence intervals are wide.
- **F2 at 27–37% is far from ceiling**; its report probe is void, so fabrication cannot be
  ruled out on that cell by behavior alone (though inspection found zero tamasic
  fluent-wrong + fabricated-holding pairs).
- **The Part I lens caveats temper every lens-based claim in Part II** — the lens shows the
  intermediate is *represented*, not that a workspace *mechanism* exists.

**Next levers, in pre-declared order:** workspace **register tokens** (an architectural seat
for the holding-space — to test whether the *dynamics* can be bought too), then **4B scale**
(where latent composition is stronger and, per the audit, the real signatures begin to appear).

---

*Read the full paper: [`results/FINDINGS.md`](results/FINDINGS.md). Read the audit in depth:
[tao-hpu/jspace-replication](https://github.com/tao-hpu/jspace-replication).*
