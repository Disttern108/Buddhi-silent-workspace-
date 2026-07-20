# Phase 2 — Did the training conditions create workspace machinery?

**Verdict in one line: the conditions installed workspace *content* — block-specific,
verbalizable latent intermediates and a routing shift toward serial reasoning — but
not workspace *architecture*: no ignition, no band structure, and no causally
load-bearing lens directions at 0.6B with 290 examples.**

Subject: `models/run2_full_hf` (full-parameter FT, lr 5e-5 × 1500 iters).
Lens: same 50-prompt recipe as base. Data: `results/run2-full/layers/`.

## Before → after, signature by signature

### 1. Ignition (the commitment test) — UNCHANGED, still absent
Mixed-embedding inputs still tracked linearly at every layer (share ≈ 0.5 at
α = 0.5, max slope 1.7 vs base 2.1; a commitment step would be ≫5). Fine-tuning
did not create winner-take-all dynamics.

### 2. Layer metrics — confounded by narrowing, no workspace stage
Autocorrelation flattened (L18: 5.8 → 2.6) and early-layer lens agreement jumped
(L5: 0.1% → 15.8%). Both are signatures of the model narrowing onto the fact
world (these metrics run on wikitext, where the fine-tuned model degraded), not
of workspace formation. No two-stage structure emerged.

### 3. Latent intermediate readability — POSITIVE, block-specific (the key result)
On invented chain questions ("What is the capital of the ally of the rival of
X?"), rank of the true first-hop entity vs a matched same-block control entity
(`hop_control.json`):

| | true intermediate beats matched control |
|---|---|
| Block A (flashcard) | 5/12 — chance |
| Block B (trace) | **10/12** |

Both blocks show a global bias toward invented tokens (controls reach rank 2-4 —
which is why the uncontrolled heatmap looked uniformly good; documented as a
trap). But only the trace-trained block carries *chain-specific* intermediate
information. The format of training determined whether latent intermediates are
verbalizable — the samskara claim, measured internally and properly nulled.

### 4. Causal tests — NEGATIVE, and informative about why
- **Ablation dissociation** (`causal_tests.json`): ablating top-10 lens
  directions in L14-23 dropped chains 20/40 → 10/40, but the matched-rank
  *random-direction control* dropped them to 4/40. No selectivity: this small,
  narrowly fine-tuned model is brittle to any 10-direction removal across 10
  layers. The paper's dissociation is not reproducible with this intervention
  at this scale.
- **Intermediate swap**: 0/16 clean-correct chains flipped to the counterfactual
  answer (6/16 kept original, rest degraded). Two candidate reasons, both
  consistent with the data: (a) multi-token invented names give poor
  single-token direction handles (the paper's own worst category was
  low-workspace-loading vocabulary); (b) **externalization** — the B-model
  re-derives the chain on the page token by token, re-reading the *written*
  question and partial trace, so internal-direction interventions have less to
  grab. The paper saw the same shielding: chain-of-thought math was more
  ablation-robust than direct math.

## Combined before/after scoreboard

| | Base (before) | Conditioned (after) |
|---|---|---|
| Fact knowledge | 0% | A: 50% recall / 29% transfer · B: 13% recall / 59% transfer |
| Default routing | n/a | B-block: compulsive serial chaining (over-serialization) |
| Latent intermediates | late, inside output ramp | **B-block: specific & verbalizable (10/12 vs null)**; A-block: chance |
| Ignition / commitment | absent | absent |
| CKA band structure | soft only | unchanged/confounded |
| Causal lens handles | n/a (knows nothing) | not demonstrated |

## What this means for the program

The task-distribution knobs successfully shaped *what lives in verbalizable
space* (content) but 290 examples on a 0.6B base did not reorganize *processing
architecture* (ignition, bands, causal privilege). The levers that plausibly
close the gap, in cost order:

1. **More data & steps**: sweep `--families 100+ --exposure 30+`, longer
   training with general-text mixing to prevent narrowing (the narrowing
   confound also disappears then).
2. **Condition C**: reward thinking *selectively* — the over-serialization
   result shows the model learned "always reason"; C must teach *when* to
   reason (reflection data mixing direct-answer and reasoning cases, then
   Quiet-STaR-style thought-utility reward).
3. **Scale**: Qwen3-1.7B-Base, same protocol (the replication repo already
   probes 1.7B; workspace precursors are stronger with scale in every
   published result).
