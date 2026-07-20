# Verbalizable Workspaces in Small Language Models
### What replicates from the Global-Workspace paper, what is a lens artifact, and whether a working holding-space can be *installed*

*Consolidated findings, 2026-07. Two linked projects: an independent audit of
Anthropic's "Verbalizable Representations Form a Global Workspace in Language
Models" (2026) on small open-weight models (`jspace-replication/`), and a
constructive follow-on that tries to force a functional holding-space into a
1.7B base model by curriculum (root "Buddhi" program). Every headline traces to a
dated log entry and a results JSON.*

---

## Summary in one page

The Global-Workspace paper argues that language models carry an emergent,
verbalizable "workspace": shared representations that multiple computations read
and write, holding intermediate thoughts that a Jacobian lens can read out. We
tested this two ways.

**Part I — audit.** On Qwen3-1.7B/4B (and a 1.7B→27B / Gemma-2 ladder for the
capture work), the claims split cleanly:

- **Real:** shared/associative fact representations (edit *France→China*, every
  dependent answer flips together) — replicates and *strengthens* with scale.
  Two phenomena the public review did not test are also real and causal: a
  covert **language/format register** that carries the *setting* but not the
  content, and **perspectival capture** — a mid-band entity swap that rewrites
  the model's *restatement of the question itself*, a phenomenon with no prior
  demonstration we know of.
- **Not what it looked like:** the "rewrite an intermediate thought" result is
  **directed steering**, not surgery on a designated thought (an absent word
  flips nothing; flips track direction overlap, not the edited concept). Rhyme
  "planning" is next-word plausibility ramping up, not a plan. Mental arithmetic
  is not lens-readable above chance at these scales.
- **The instrument itself:** the Jacobian lens has real false positives and
  **no consistent advantage over a vanilla logit lens** on its own evaluation
  sets; its transported directions collapse into a narrow cone universally, with
  a self-trained 124M model as the load-bearing counterexample proving the
  collapse is a property of the *fit*, not a mathematical necessity.

**Part II — construction.** If the *architecture* (ignition, workspace bands)
isn't there at small scale, can task structure install the *function* — silent,
internal multi-step composition — anyway? Across eight pre-registered phases we
show: at 0.6B the reasoning loop is **external** (it runs through self-written
text on the page; gag the model and composition collapses to the floor while
single-fact recall *improves*). At 1.7B a four-stage "Holding" curriculum
installs a **functional holding-space**: never-trained two-step arithmetic done
silently rises from ~0% to **26.7%** (4× its cued floor), lifting to **36.7%**
under a route-reward RL stage; the free-vs-silent gap closes to zero on every
battery, and the skill transfers to a **never-trained astrology vocabulary**
(silent chains 0%→50%). What never forms is the *architecture*: no ignition, no
CKA blocks, unchanged readout ramp — exactly the audit's least-supported,
most scale-dependent signatures.

**The one-line synthesis:** at 1.7B you can relocate computation *inward*
(function) without reorganizing the *dynamics* (architecture). The paper-grade
workspace signatures are a scale/architecture phenomenon, not something a
curriculum manufactures — and several of them, read carefully, are properties of
the lens rather than the model.

---

# Part I — Auditing the Global-Workspace claims

**Setup.** Qwen3-1.7B and Qwen3-4B with Neuronpedia's pre-fitted Jacobian lenses
(GPT-2 124M for pipeline sanity; the perspectival-capture campaign adds a
1.7B/4B/8B/14B/27B dense ladder plus hybrid Qwen3.5-9B/3.6-27B and
Gemma-2-2B/9B/27B). Method rules, held throughout: every result separates three
sources — the paper's claim, the external review's verdict (Nanda, LessWrong),
and our own measurement — never blended; every run is logged the day it happens,
failures included; the official `jacobian-lens` code is never modified
(adaptations live in `src/`); headline rates carry bootstrap CIs and a
prompt-set sensitivity reanalysis.

### Verdict table

| Claim | Paper | Review | **Our measurement** |
|---|---|---|---|
| **C1** shared representations (edit one concept, all dependents flip) | ✅ strongest result | ✅ replicates | **Replicated, strengthens with scale.** 97.2% hit / 0% stayed at 4B; the early-mid band (layers 4–13 of 28) alone carries the whole effect. Boundary mapped: only *associative* links move — *operations* on concepts fail with three signatures (stayed / echo / broken). |
| **C2** thought-swap (rewrite a two-hop intermediate) | ✅ | ⚠️ weak, "close to substituting the answer token" | **Partially replicated (50–65%), and non-surgical.** Along lens directions the answer-token control *beats* the intermediate swap (4B 85.4% vs 50.0%, p=0.0005). Trained linear probes restore the intermediate swap to 50–65% and drop the control to ~40% — so C2 survives at reduced magnitude, but the "surgical thought edit" reading does not (see E5). |
| **C3** rhyme planning | ❌ failed | ❌ | **Not replicated.** Rhyme word never in lens top-10 at the readout position (pass@10 = 0%, both scales). The apparent late "emergence curve" is the rhyme word climbing into the model's own next-token distribution; strict anticipation falls to the lens false-positive base rate (~3–4%). |
| **C4** internal arithmetic, lens-readable | ❌ failed | ❌ | **Not replicated** (readability route). Order/operation readout at or near base rate: 1.7B exactly 0 above chance; 4B 14.5% vs 8.2% control. |
| **C5** lens reliability | "undoubtedly imperfect" | ⚠️ heavy false positives | **Confirmed and quantified.** Best-set sensitivity 41% pass@10; permutation control fires at 9–11% on multihop; and the Jacobian lens shows **no consistent advantage over the vanilla logit lens** on its own eval sets (typo, 4B: J 26.0% vs logit 69.8%). |
| **C6** global-workspace dynamics (ignition, capacity, selectivity) | claimed | least confident | **Not tested** (by design; not compute-funded). |

### The three positive anchors

1. **Shared representations are real (E1).** France→China flips capital,
   language, continent, and currency together, 97.2% at 4B, and *cleaner* than at
   1.7B — the one claim that gets stronger with scale. The effect is localized to
   an early-mid band and is specific to associative links; asking the swapped
   concept to be *operated on* fails.

2. **Covert registers are causal (E6 / E6t).** A mass-mean language axis,
   translated at α=0.125, flips the output language 91–94% of the time (random
   control 0%, McNemar p<1e-4); on the *covert* subset (target form outside the
   clean top-100) the register still flips language 79–96% while full content
   flips drop to ~30%. **The register carries the setting — the language, the
   intended form of a typo — not the content.** A typo-axis erasure confirms the
   second register is causally load-bearing. This is the honest residue of the
   "workspace holds covert plans" claim: what is genuinely covert and steerable
   is *context registers*, not thought plans.

3. **Perspectival capture is a new phenomenon (E7).** In a shared-KV
   answer→restatement→self-report protocol, a mid-band entity swap makes the
   model **restate the question itself with the swapped entity** (1.7B: answer
   flips 78.6%, restatement rewritten 71.4%, lens-behavior agreement 96%). It is
   **scale-stable from 1.7B to 14B**; the model's *self-report* about the edit,
   by contrast, changes shape at every scale. Where the model detects the edit
   ("Wait…" self-corrections at 4B), detection is not escape. Mechanism, across
   nine models: the writable zone's cutoff is a **composite — whichever comes
   first of a coordinate **drift wall** (window-mean drift ~0.5–0.8) and the
   task's **consumption deadline** (the layer where the model commits the
   answer). Width-6 windows are writable *somewhere* in every model; matched-dose
   random directions flip 0% (the effect is direction-specific); the
   intervention-free commit layer rank-correlates with the zone cutoff at
   ρ=0.86. The earlier "open/closed gate" taxonomy was a single-layer dose
   artifact and is retired on the record.

### What dissolves under a control

- **The thought-swap is directed steering, not surgery (E5).** Swapping a
  *non-answer* in-context concept into the target flips the final answer about as
  often as swapping the answer itself (4B 77.1% vs 85.4%, p=0.29). Flips are
  predicted by the cosine between source and answer transported directions
  (0.755 for flipped vs 0.324 for not), **not** by how much amplitude was
  harvested; an absent word ("piano") harvests comparable amplitude and flips
  **nothing** (0%). The operator is *directed steering whose gain is whatever
  coherent in-context signal the source direction happens to harvest* — including,
  through direction overlap, the answer's own.

- **Covert "plans" are mostly registers (E4b mouth-exclusion audit).** Scoring
  every lens hit against the model's own next-token distribution, covert content
  (lens rank ≥100, i.e. not already in the "mouth") survives above the
  permutation floor almost only for multilingual and typo registers (multilingual
  31.3%/23.6% vs floor 3.3%/1.4%); poetry is 0%, order-ops at/below floor,
  association 1–2%, multihop a 3–5% residue. The paper's one-off ablation filter,
  promoted to a formal score, splits its own evaluation sets into **surviving
  registers and dead plans.**

- **The lens lives in a collapsing cone (transport-cone geometry).** Raw
  unembedding directions span an effective dimensionality of 75–128 across the
  family; J-transported directions collapse to 2–12 everywhere (11–51×). The
  **self-trained 124M GPT-2** is the load-bearing counterexample: its transport is
  *more* isotropic than its raw directions (23.4→31.2), robust across fit scale
  (150→1000 prompts) and backend (MPS→CUDA) — proving the collapse elsewhere is a
  property of the *fitted lens*, not a mathematical necessity of the transport.
  The two tightest cones also carry the highest baseline lens false positives
  (33.9%) — a behavioral consequence of the geometry, reported as a two-regime
  split, not a law.

**Audit takeaway.** The workspace's *shared-representation* core is real and
scale-robust. Its *covert-content* story is real only for registers. Its
*intermediate-thought-editing* and *planning* stories are, at these scales, lens
artifacts or non-specific steering. And the instrument that reads the workspace
has no measured edge over a logit lens on its own benchmarks. This is the ground
truth the construction program was built on: **do not expect the architecture at
1.7B; ask instead whether the function can be installed.**

---

# Part II — Constructing a holding-space ("Buddhi")

**The question.** The conditions for a workspace are properties of the *task
distribution*, not just parameter count. A fact-world generator
(`fact_world_generator.py`) makes invented worlds with four knobs — integration
**depth**, **serial** dependency, **interference** (confusable twins), and
**novelty** (held-out combinations) — emitted in two matched formats: Condition A
(flashcard, Q→A) and Condition B (integration, Q→verbalized working→A), with
atomic-fact exposure audited equal across formats so any difference is
attributable to *format*, not exposure. Every phase's predictions were written
and committed **before the data existed**; failures were logged the same day; the
gag protocol (append `" Answer:"`, 4-token budget) forces commitment so the model
cannot chain out loud, isolating *silent* composition.

### Phases 0–2 — content installs, architecture does not (Qwen3-0.6B)

- **Phase 0 (baseline): no workspace.** Every defining signature is absent or
  degenerate. Ignition is the clean negative: mixed-embedding inputs track the
  input mixture *linearly at every layer* (slope 2.1; a commitment step would be
  ≫5) — the model never commits, no winner-take-all anywhere. No CKA block
  structure, a single late lens ramp. Latent two-hop intermediates *are* readable
  but only inside the output ramp (71–93% depth), not in a separate earlier band.
  The raw material exists; the architecture does not.
- **Phase 1 (behavior): the samskara/format effect.** On never-trained fact
  combinations, integration-format entities transfer at **2.6× the flashcard
  rate (61% vs 23%)**, surviving strict scoring; base model = 0% everywhere (no
  leakage). Side finding: trace training makes the serial route the *default* —
  the model keeps chaining even where a lookup suffices (over-serialization).
- **Phase 2 (internals): content yes, architecture no.** Only the trace-trained
  block carries *chain-specific* verbalizable intermediates (10/12 vs a matched
  same-block control; flashcard block 5/12 = chance). But ignition is unchanged,
  causal ablation is non-selective (random-direction control drops chains as
  much as the lens directions), and intermediate-swap flips 0/16. Format shapes
  *what lives in verbalizable space*; 290 examples on 0.6B do not reorganize the
  machine.

### Phases 3–4 — the reasoning loop is *external* (the pivotal negative)

- **Phase 3 (Condition C): a trainable, conditional gate.** A mixed regime
  (traces + direct QA on the same entities) dominates both pure formats on both
  axes *and* gates by question type: multi-hop chain-rate 49% on transfer vs 10%
  on recall; recall C 60% > A 34% > B 13%. The task distribution can install a
  *routing policy*, not just a pathway. Architecture still absent (three strikes).
- **Phase 4 (gag-by-design): where the groove lives.** With every completion
  ending in `"Answer: X"` (making the gag in-distribution), the split is decisive:
  **atomic facts are in the weights** (gagged recall *improves*, +13/+17 points —
  external chaining was interference, not access), while **multi-hop composition
  is on the page** (gagged transfer collapses 80% → 4.5–7.5% in both trace
  blocks). The 0.6B "workspace" is the model's own output stream. This
  retroactively explains every internal negative — there was no internal loop to
  grab. (Also: terminal-format consistency alone lifted transfer 33–42% → 80% —
  format is worth more than any knob at this scale.)

### Phases 5–6 — external validation, and the forgetting lesson

- **Phase 5 (real facts):** the mechanism reproduces on real knowledge in a model
  we never fine-tuned — gagging costs the base model **−50 points on 2-hop
  (61%→11%) but only −11 on 1-hop**. But the fine-tuned run4 scores **0% on all
  real-world items**: full-parameter FT with thin mixing caused catastrophic
  forgetting. New standing constraint: preserve general capability (LoRA + heavy
  mixing).
- **Phase 6 (fresh-world in-context):** run4 cannot apply its format to a new
  world — it cannot even *read* a fact stated in the prompt (15% vs base 100%).
  Skill and facts are a single fused groove; nothing content-independent was
  installed. Third independent confirmation of page-dependence: even with facts
  pre-written on the page, gagged composition fails (base 33%→5%) — the loop needs
  *self-written* steps, not merely a readable page.

### Phase 7 — the Holding curriculum at 1.7B (**the headline result**)

**Design.** Qwen3-1.7B-Base + LoRA (r=16, q/v, 16 layers), a four-stage "Holding"
curriculum on real content only (~6k iters), 45% wikitext mix for preservation.
Four never-trained held-out families: F1 real 2-hop, F2 two-step arithmetic, F3
selection-under-distractors, F4 lookups. Stages: **S1** full visible Holding →
**S2** ≤3-word Holding → **S3** Holding deleted, report still demanded → **S4**
deleted + sparse probes.

**The baseline reframing (controls did their job).** Two pre-registered claims
were *refuted by our own controls*, and we retired them on the record:

- At 1.7B, page-dependence for *weights-content* composition is essentially gone
  (gagged 2-hop 54.3% ≥ free 47.8%) — capacity was the 0.6B binding constraint.
- The favorite "page→space deficit" was a **production artifact**: rank answers by
  likelihood instead of demanding generation and in-context chains hit 85% ≈
  real-world 80.4% (kill condition fired). The base model's latent content was
  present all along; what it lacked was **commitment on demand, selection, and the
  silent arithmetic itself** — not content.

**Silence was the active ingredient.** Arithmetic stayed at 0% gagged through S1
(visible holding) and S2 (compressed holding); it lifted only at **S3, when the
holding was deleted and the report still demanded.** S2's honest failure mode
(correct −c applied to a *wrong* held product) showed the space→answer route
forms *before* the silent computation that feeds it.

**Scorecard against pre-registered bets:**

| bet | criterion | outcome |
|---|---|---|
| B1 | S1 free format ≥70% | **95.8%** (after root-causing the mlx chat-template landmine) |
| **B2** | **gagged held-out ≥25% abs AND ≥2× base (decisive)** | **F2 arithmetic PASS: 26.7%, 4.0× the 6.7% cued floor** — silent two-step on never-trained problems, nothing on the page. F1 90.9% clears absolute (+18 pts); its 2× was arithmetically impossible from a 72.7% baseline. |
| B3 | silent reportability ≥60% + faking control | F1 PASS 90% true-when-right; **F2 VOID by our own rule** — the probe reports true products 91% even on wrong answers (probe-time recomputation indistinguishable; the reviewer's predicted failure mode). |
| B4 | preservation gates | all green every stage; real-world 2-hop gagged **improved** 54→72%. |
| B5 | space generalizes | in-context chains gagged 0→72.5%; **astrology (never-trained vocab): gagged reading 10→95%, chains 0→50%.** |
| B6 | intermediates readable; ignition absent | both confirmed: lens C3 28/30 (median rank 3 vs control 125); page-selectivity 0/5→3/5 (low n, suggestive); ignition/CKA unchanged. |

**Before → after grid (strict %, gagged = `" Answer:"` + 4 tokens):**

| battery | base free | base gag | S4 free | S4 gag |
|---|---|---|---|---|
| held-out F1 real 2-hop | 61 | 73 | 91 | **91** |
| held-out F2 arithmetic | 0 | 7 | 27 | **27** |
| held-out F3 page selection | 100 | 33 | 100 | **100** |
| real-world 1-hop / 2-hop | 32\*/48\* | 79/54 | 89/72 | 89/**72** |
| in-context read / chains | 100/62.5 | 10/0 | 95/72.5 | 95/**72.5** |
| astro read / chains | 70/20 | 10/0 | 95/50 | **95/50** |

\*compliance-artifact cells (base model rambled; the artifact is eliminated by
training — every S4 answer is `"Answer: X"` then stop).

**What the curriculum installed:** commitment (everywhere), selection
(behaviorally 33→100% on F3; lens-level 0/5→3/5), and *partially the silent
arithmetic itself* (0→27%). The free/gag gap closed to zero across every battery.

**What it did not install:** the workspace *architecture* — no ignition
bimodality, no CKA reorganization, readout ramp unchanged, **exactly as predicted
(B6).** Training relocated computation inward; it did not reorganize the dynamics.

### Phase 8 — route-reward RL (run8)

A GRPO-style route reward (operationalizing Gita 2.47 "the route earns, not just
the fruit," and the 18.30–32 sattvic/rajasic/tamasic rubric: `+fruit +sattvic
−tamasic-inconsistency −silence-leak`), 300 steps on LoRA from S4. **F2 gagged
26.7% → 36.7%** (pre-registered bar ≥35%), free identical (silence retained) —
the full trajectory of the cleanest cell is ~0% → 26.7% (curriculum) → 36.7% (RL)
= **5.5× the cued floor.** Preservation green. The F2 report probe remains void by
the faking rule (RL neither created nor cured the recomputation confound).

---

## Synthesis — function without architecture

The two projects meet at one sentence: **at 1.7B you can force the *function* of a
workspace (silent multi-step composition, commitment, selection, generalization to
alien vocabulary) without any of its *architecture* (ignition, CKA blocks, a
separated readout stage).** That is consistent with, and predicted by, the audit:
the architectural signatures are the paper's least-supported, most
scale-dependent claims, and several "workspace content" phenomena are lens
artifacts or non-specific steering rather than held internal plans. The audit
tells you *not to expect the machine*; the construction shows you can still get
the *behavior*, and pinpoints the three operations that were actually missing at
baseline — commitment, selection, silent arithmetic — none of which is "latent
content."

## Methods and rigor

- **Pre-registration throughout.** Every construction phase committed numeric
  predictions, decision rules, and falsification lines before the data existed;
  two headline baseline claims were *refuted by our own controls* and retired.
- **Controls over generation.** Likelihood-rescoring (immune to production
  artifacts), single-token real-value batteries, rare-direction compositions, and
  a faking control that voided one of our own reportability claims.
- **Three-way separation and same-day failure logging** on the audit side;
  bootstrap CIs and a prompt-set sensitivity reanalysis on every headline rate;
  vendored reference code never modified.
- **Tooling truths paid for in dead runs:** mlx's completions format silently
  wraps base-model prompts in a chat template (killed two training stages); a data
  mix means token-share, not row-share; the free column of any base model measures
  compliance before knowledge.

## Limitations and next levers

- **Small n** everywhere (30–66 per cell; page-selectivity n=5). Directionally
  consistent across seven independent batteries, but CIs are wide.
- **F2 at 27–37% is far from ceiling** — the silent multiply is the remaining
  bottleneck. Its report probe is void, so *fabrication cannot be ruled out by
  behavior alone* on that cell (though the inspected rubric found zero
  fluent-wrong-answer + fabricated-holding pairs).
- **Lens caveats from Part I temper every lens-based claim in Part II** (C3/C5,
  the collapsing cone, ~3–4% false-positive floor); the lens shows the
  intermediate is *represented*, not that a workspace *mechanism* exists.
- **Next, in pre-declared order:** workspace **register tokens** (an
  architectural seat for the holding-space, to test whether the *dynamics* can be
  bought too), then **4B scale** (where latent composition is stronger and, per
  the audit, the real signatures begin to appear).

## Artifacts and reproducibility

- Audit: `jspace-replication/` — `docs/{claims-inventory,plan,replication-log}.md`,
  `experiments/e{0..7}*`, `results/*.json` (E1–E7, cone geometry, bootstrap CIs,
  multi-seed); official `jacobian-lens` pinned in `third_party/`.
- Construction: `results/phase{0..8}*.md` (reports + pre-registrations),
  `adapters/run7_s{1..4}`, `adapters/run8_rl`, `models/run7_s4_fused`,
  `models/run8_rl_fused`, per-battery eval JSONs and lens outputs under
  `results/`. Data generators: `fact_world_generator.py`, `probe/make_*.py`;
  training configs `configs/run7_s{1..4}.yaml`; eval `eval_model.py`,
  `probe/{battery,fit_lens,c3_silent_lens,eval_probe,ll_rescore,rl_route}.py`.

*Two runtimes: MLX (`mlx_lm`) for training/fusing/eval/RL; PyTorch + `transformers`
+ `jlens` for the lens and structural battery.*
