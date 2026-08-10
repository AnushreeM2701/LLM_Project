# Output Tables

What each table in this folder represents, since the CSVs themselves are kept
as plain data (no comment/caption row) so they open cleanly in any spreadsheet
tool. See `docs/methodology.md` for the full dataset construction details.

Two different Hard-tier pools are used across these tables — the note for
each table below says which one applies.

- **Balanced 40-question pool**: a seeded random sample of 40 of the 51 Hard
  questions, matching Easy/Medium's 40-question size, so all three difficulty
  tiers are compared on equal N.
- **Common-wrong-question pool**: only the Hard-tier questions that *all
  three models* answered incorrectly under a given prompt (a smaller,
  variable-size set), so every model's error breakdown is drawn from exactly
  the same questions.
- **Full Hard tier**: all 51 Hard questions, no restriction.

| Table | Columns | Hard-tier pool | Notes |
|---|---|---|---|
| `accuracy_by_model_prompt_difficulty.csv` | Model, Prompt, Difficulty, N, Correct, Accuracy, CI Lower, CI Upper | Balanced 40 | Wilson score confidence intervals. N = responses in that cell. |
| `accuracy_by_model_category.csv` | Model, Category, Difficulty, Prompt, N, Correct, Accuracy, CI Lower, CI Upper | Balanced 40 | Full Model x Category x Difficulty x Prompt breakdown (72 rows). |
| `rq1_error_type_by_model.csv` | Model x Error Type (contingency table) | Balanced 40 | Incorrect responses only. Input to the chi-square test below. |
| `rq1_independence_test.csv` | test, statistic, p_value, dof, min_expected_count, caution | Balanced 40 | Chi-square test of independence on the contingency table above. `caution=True` flags expected cell counts below 5. |
| `error_type_distribution_{cot,tot}.csv` | Model, Error Type, Easy, Medium, Hard, Total | Balanced 40 | Error Type x Difficulty counts, all 3 models, one file per prompt. |
| `error_type_frequency_hard_{cot,tot}.csv` | Model, Error Type, Count | Common-wrong-question | Hard tier only, restricted to questions all 3 models got wrong under that prompt. |
| `error_subtype_word_frequency_hard_{cot,tot}.csv` | Model, Word, Count | Common-wrong-question | Top-10 word frequency in the free-text Error Subtype field, Hard tier only. |
| `rq3_prompt_comparison.csv` | Model, Difficulty, CoT Accuracy, CoT N, ToT Accuracy, ToT N, CoT-only Correct (b), ToT-only Correct (c), McNemar p-value, Significant, Direction | Balanced 40 | McNemar's Exact Test, paired per model. b/c are the discordant-pair counts. `Difficulty="All"` is the pooled test across all three tiers (the headline RQ3 result); Easy/Medium/Hard rows show the same test run separately per tier -- Easy and Medium mostly have zero discordant pairs (CoT and ToT rarely disagree there), so almost all of the pooled significance comes from the Hard-tier row. |
| `execution_time_summary.csv` | Model, Prompt, Difficulty, N, Mean (s), Median (s), Std (s) | Balanced 40 | |
| `step_count_summary.csv` | Model, Prompt, Difficulty, Correct, N, Mean Step Count, Median Step Count | Balanced 40 | |
| `hard_tier_hardest_questions.csv` | Question ID, Failures (of 6) | **Full 51** (deliberate exception) | Ranked by how many of the 6 model x prompt conditions got it wrong. |

`hard_tier_hardest_questions.csv` is the one deliberate exception to the
balanced-pool convention: it backs the correctness-matrix figures
(`correctness_heatmap_hard.png`), which are meant to show every Hard-tier
question, not a sample of them.
