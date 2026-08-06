# Dataset Methodology

<!-- BEGIN AUTO-GENERATED COMPOSITION TABLE (written by src/dataset/freeze_dataset.py — do not hand-edit) -->

### Dataset Composition

Total questions: **131**

| Category | Easy | Medium | Hard (AIME) | Total |
|---|---|---|---|---|
| Algebra | 10 | 10 | 16 | 36 |
| Number Theory | 10 | 10 | 11 | 31 |
| Probability | 10 | 10 | 8 | 28 |
| Combinatorics | 10 | 10 | 16 | 36 |
| **Total** | 40 | 40 | 51 | 131 |

Hard tier is **AIME-only** (Hendrycks Level-5 excluded — see `docs/limitations.md` for the empirical justification). Category counts within Hard are **unbalanced by design**: Probability caps out at 8 available curated AIME questions, and the other categories are not trimmed to match, to preserve statistical power. Category is treated as a covariate in analysis rather than requiring equal N.

Easy/Medium tiers are Hendrycks MATH, balanced to 10 questions per category (seed=25116096).

<!-- END AUTO-GENERATED COMPOSITION TABLE -->
