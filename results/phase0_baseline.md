# Phase 0 — Does Qwen3-0.6B-Base already have a J-space/workspace?

**Verdict: No workspace. Weak precursors at most.** The base model processes
smoothly and continuously; every signature that defines a workspace in the
Anthropic paper is absent or degenerate here. This is the "before" baseline
that the conditioned models are measured against.

Lens: fitted with the official `jlens` code, 50 WikiText prompts × 128 tokens,
27 source layers, target L27 (recipe fixed for ALL models in this project).
Per-layer data: `results/qwen3-0.6b-base/layers/`.

## Signature-by-signature

### 1. Four layer metrics (`layer_metrics.csv`) — no two-stage structure
- **Lens top-1 agreement**: ~0% through L15, then one smooth ramp to 64% (L26).
  The paper's models show plateau→sharp final jump; here it's a single ramp —
  ordinary output convergence, no workspace stage.
- **Excess kurtosis**: high at L0-L2 (embedding artifact), decays, weak bump
  0.7→1.2 at L19-24. Paper: near-zero early, sharp rise at onset. Not present.
- **Position autocorrelation**: soft peak ~L16-19 (6.0 at L17). The one hint
  of a deliberation-like band. Weak.
- **Effective dimensionality (PR of J_l)**: rises near-monotonically 6→317
  with no plateau; no distinct workspace block.

### 2. CKA of lens geometry (`cka_matrix.npy`) — soft banding only
Block means: adjacent blocks blend smoothly (L8-16 vs L16-24: 0.82). No sharp
three-block sensory/workspace/motor structure like the paper's Figure 28.

### 3. Ignition test (`ignition.json`) — CLEAN NEGATIVE, strongest evidence
Mixed-embedding inputs (12 country pairs × 4 carriers, α∈[0,1]): the
activation share tracks the input mixture *linearly at every layer* —
share≈0.50 at α=0.5 everywhere, max slope 2.1 (a commitment step would be ≫5).
The paper's models switch bimodally at workspace onset (~L38/96). **This model
never commits — no winner-take-all dynamics anywhere in the stack.**

### 4. Two-hop latent intermediates (`hop_rank_heatmap.csv`, `readouts_hops.json`)
Latent intermediates ARE readable — "France" hits rank 0-4 in the lens on
8/10 prompts — but only at L20-26 (71-93% depth), the same layers where the
final answer converges (L22 readout: Paris > France). The paper's workspace
holds intermediates in a *separate, earlier* band (~17% before answers).
Here deliberation and output are not separated: latent 2-hop capability
exists, but it lives inside the output ramp, not in a workspace.

## Reading

0.6B-Base has the raw material (latent intermediates, a faint autocorrelation
band) but none of the workspace architecture: no ignition, no band structure,
no separated deliberation zone. Whatever appears in the conditioned models
after fine-tuning was **not already there**.

## Next (Phase 2 targets)

Same lens recipe + same battery on the fused fine-tuned model(s), plus causal
tests (`probe/causal_tests.py`): ablation dissociation (chains collapse,
recall survives) and intermediate swap (answer flips to counterfactual).
Delta vs this baseline is the claim.
