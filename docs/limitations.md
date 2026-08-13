# Known Limitations

Factual technical notes on known constraints and threats to validity,
consolidated from comments scattered across the codebase (each source file
below already flagged these inline). This is a reference for writing the
thesis's Limitations/Discussion sections in your own words — it is not
itself thesis prose.

## Judge conflict of interest

Gemini is both the primary error-classification judge and one of the three
models under evaluation, which is a direct conflict of interest in the
central model comparison (a judge could be more lenient toward its own
outputs). The mitigation was an independent validation sample re-judged by
a neutral model not under evaluation, with inter-judge agreement (Cohen's
kappa) reported.

**Status: not completed.** The neutral judge (`llama-3.3-70b-versatile` via
Groq — the original choice, Kimi K2, was removed from Groq's model catalog
entirely) hit Groq's free-tier daily token quota (100,000 tokens/day)
after roughly 20-25 of the planned 120 validation calls; each call costs
~4,000+ tokens because the judge prompt includes the full question,
category rubric, and the model's complete reasoning text. The full 120-row
sample would need ~5-6 days spread across daily quota resets to complete
on the free tier. This was a deliberate scope decision given project time
constraints, not an oversight — see `src/models/neutral_judge.py` and
`src/evaluation/error_judge.py`.

Source: `src/models/neutral_judge.py`, `src/evaluation/error_judge.py`.

## Answer-grading validation

The automatic answer evaluator (`src/evaluation/answer_evaluator.py`) is
covered by unit tests (`tests/test_answer_evaluator.py`) against known
tricky cases (fractions, percentages, mixed numbers, symbolic equivalence).
A separate planned validation step — sampling ~30-40 rows for manual human
grading, to check the automatic evaluator against human judgment on real
model outputs rather than only hand-picked test cases — was never
implemented (`validate_evaluator.py` does not exist).

## Reasoning-effort confound (controlled, not eliminated)

Gemini and GPT-OSS-120B (Groq) both expose an internal "thinking
effort"/"reasoning effort" control. Both are pinned to their lowest
available setting and held constant across CoT and ToT, so the
prompting-strategy manipulation (RQ3) is the intended variable being
measured, not a side effect of a changing internal reasoning budget.
Pinning to the lowest setting does not mean zero internal reasoning —
GPT-OSS in particular is reasoning-native, meaning it may still perform
some internal chain-of-thought regardless of prompt. This is a controlled
confound (held fixed across conditions), not an eliminated one (its
absolute contribution to accuracy is unmeasured).

Source: `config/config.py` (MODELS section), `src/models/groq_gptoss.py`,
`src/models/gemini.py`.

## Hard tier: AIME-only, and why

The Hard tier uses AIME questions only, not Hendrycks MATH Level 5. AIME is
a single, consistent competition-mathematics source with a uniform integer
answer format (0--999), keeping answer-parsing and grading consistent
across the tier. Hendrycks MATH Level 5 is a separate item bank with a
different answer format (including fractional answers) and no independent
validation that its internal "Level 5" label represents the same difficulty
as an AIME problem. Mixing the two under one difficulty label would risk
treating non-comparable item banks as equivalent, so the Hard tier was
restricted to AIME only.

Source: `src/dataset/freeze_dataset.py`.

## Category imbalance in the Hard tier

The Hard tier uses every currently curated, category-labelled AIME
question (51 total) rather than trimming categories to match the smallest.
Category counts: Algebra 16, Combinatorics 16, Number Theory 11,
Probability 8. This is an unbalanced design by choice — trimming
Algebra/Combinatorics/Number Theory down to Probability's 8 would discard
curated data for no benefit. Category should be treated as a covariate in
analysis, not assumed to have equal statistical power across categories at
this tier.

Source: `src/dataset/freeze_dataset.py`, `config/config.py`
(`HARD_TIER_SOURCE`, `HARD_TIER_BALANCED`).

## Data contamination risk (Hendrycks MATH)

Hendrycks MATH (used for the Easy/Medium tiers) is a widely-used public
benchmark, very likely present in the pretraining data of all three study
models — a genuine risk that Easy/Medium accuracy reflects memorization
rather than reasoning. AIME was chosen for the Hard tier partly to reduce
this risk (AIME questions are still public, but the specific combination
of recency and lower republication volume makes memorization somewhat less
likely than for a benchmark dataset built specifically for ML training/eval
use).

Source: `src/dataset/extract_hendrycks.py`.

## AIME PDF extraction correction

The original AIME text extraction (`src/dataset/extract_aime.py`, via
`pdfplumber`) corrupted subscript/superscript LaTeX notation from the
source PDFs — simple single exponents were usually still inferable from
context, but multi-index notation (sequences, recurrences, log bases,
summation indices) produced genuinely ambiguous or unsolvable text,
independent of which model was asked to solve it. All 51 curated AIME
questions were manually re-transcribed by reading the source PDF pages
directly (`src/dataset/fix_aime_extraction.py`), correcting the source
file so the fix is reproducible rather than a one-off patch.

## Model pilot baseline is not perfectly clean

The pilot comparison used to justify the current model choices (Gemini,
GPT-OSS-120B, Mistral Large) cited a ~11% accuracy baseline for the prior
pipeline's models on Hard/AIME. That baseline was measured *before* the
AIME text-extraction fix above, so it is not a perfectly clean comparison
point — some of the prior models' apparent weakness on Hard/AIME may
partly reflect the corrupted question text rather than pure model
capability.

Source: `src/experiments/run_pilot.py`.

## Small subgroup sample sizes

Some analysis cells have small N (e.g. Hard-tier accuracy broken down by
category, or by model within a difficulty tier). `src/utils/stats.py` uses
the Wilson score interval for accuracy confidence intervals specifically
because it behaves better than the normal approximation at small n or
extreme proportions, but small-N subgroup comparisons should still be
interpreted cautiously in the write-up — a chi-square test run for RQ1 was
flagged with `caution=True` due to low expected cell counts (minimum
expected count 0.26) in the Model x Error Type table.

## RQ3: paired testing vs. raw percentages

Raw accuracy percentages alone suggested all three models show a similar
small CoT-to-ToT effect. McNemar's exact test on the paired (same
question, same model) CoT vs. ToT outcomes tells a different, more precise
story on the current full dataset:

| Model | CoT Accuracy | ToT Accuracy | McNemar p-value | Significant (p<0.05) |
|---|---|---|---|---|
| Gemini | 79.4% | 80.9% | 0.754 | No |
| Groq | 83.2% | 89.3% | 0.0215 | Yes |
| Mistral | 78.6% | 84.7% | 0.0078 | Yes |

Gemini's apparent ToT improvement is not statistically distinguishable
from chance at this sample size; Groq's and Mistral's are. Reporting only
raw percentages would have missed this distinction.

Source: `src/analysis/prompt_comparison.py`, `outputs/tables/rq3_prompt_comparison.csv`.
