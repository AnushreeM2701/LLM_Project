# Model Pilot Results

Pilot run before committing to the full-dataset rerun with the new model roster (gemini-3.5-flash, GPT-OSS-120B via Groq, mistral-large-latest). See config/config.py for exact settings (temperature, thinking_level/reasoning_effort) and docs/limitations.md for why these controls don't fully eliminate the reasoning-model confound, only hold it constant.

Note: this pilot ran on the CORRECTED AIME question text (see src/dataset/fix_aime_extraction.py) -- the prior pipeline's ~11% AIME-Hard accuracy figure was measured on corrupted text and old models both, so it is not a clean baseline for comparison, only a rough prior indicator.

## Accuracy by model/prompt

| Model   | Prompt   |     mean |   count |
|:--------|:---------|---------:|--------:|
| gemini  | cot      | 0.833333 |      12 |
| gemini  | tot      | 0.75     |      12 |
| groq    | cot      | 0.75     |      12 |
| groq    | tot      | 0.833333 |      12 |
| mistral | cot      | 0.666667 |      12 |
| mistral | tot      | 0.75     |      12 |

## Accuracy on Hard (AIME) only

| Model   | Prompt   |     mean |   count |
|:--------|:---------|---------:|--------:|
| gemini  | cot      | 0.666667 |       6 |
| gemini  | tot      | 0.5      |       6 |
| groq    | cot      | 0.5      |       6 |
| groq    | tot      | 0.666667 |       6 |
| mistral | cot      | 0.333333 |       6 |
| mistral | tot      | 0.5      |       6 |

## Leaked reasoning-token check

No heuristic leakage indicators (`<think>` tags etc.) found in any of the 72 pilot responses collected so far.

## Response length by model (chars)

| Model   |    mean |   min |   max |
|:--------|--------:|------:|------:|
| gemini  | 2718.08 |   783 | 17532 |
| groq    | 1415.62 |   689 |  2701 |
| mistral | 5687.88 |   667 | 23212 |
