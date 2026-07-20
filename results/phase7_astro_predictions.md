# Astro transfer battery — pre-registered predictions (2026-07-19)

Written BEFORE any model (base included) saw `data/astro_transfer/`.
Instrument: measurement-only, never trained on. F3-style pages in
astrology vocabulary (planets/signs/houses, randomized rulerships — page
must beat prior); 20 reading + 30 two-hop chains; answers all ≤2 tokens.
Purpose: does the trained space generalize to vocabulary alien to the
curriculum (B5-family evidence)? User-motivated domain (jyotish); the
market-prediction layer of the source repo is excluded — no verifiable
ground truth there.

## Baseline (untrained 1.7B) predictions

- Free reading 80-100%; free chains 40-70% (tracks ic chains 62.5%).
- Gagged reading AND chains ≤10% (the established page+cue commitment
  artifact; astro pages should behave like other fact pages).
- LL rescore: reading ≥80%, chains well above chance (content present).

## Post-curriculum (after S3/S4) predictions

- Gagged reading ≥50%, gagged chains ≥25% — the commitment discipline
  transfers to alien vocabulary despite zero astro training rows.
- Free: no regression beyond 10 pts of the base's free numbers.
- If gagged astro stays ≤10% while gagged in-context recovers, the space
  is vocabulary-bound (trained-format-specific), NOT general — reported
  as a scope limit on any workspace claim.
