# A Statistical Study of Mathematical Reasoning Errors in Large Language Models

## Overview

This repository contains the implementation and experimental results for the Master's dissertation:

**A Statistical Study of Mathematical Reasoning Errors in Large Language Models using Tree-of-Thought Reasoning**

The project investigates the mathematical reasoning performance of Large Language Models (LLMs) using different prompting strategies on a curated dataset of 131 mathematical reasoning problems (AIME and Hendrycks MATH).

Three free-tier LLMs are evaluated, none of them dedicated frontier
reasoning models (e.g. o1/o3-style models with reasoning always on):

- **Gemini** (`gemini-flash-lite-latest`, via Google)
- **GPT-OSS-120B** (via Groq)
- **Mistral Large** (`mistral-large-latest`, via Mistral)

using two prompting strategies:

- **Chain-of-Thought (CoT)** — single-call reasoning
- **Tree-of-Thought (ToT)** — real multi-branch reasoning: several candidate solutions are generated independently, then a separate selection call picks the most sound one

Research questions:

- **RQ1** — What are the most common types of mathematical reasoning errors, and does the distribution differ across models?
- **RQ2** — Is there a descriptive relationship between step count and answer correctness, and does this pattern differ across models?
- **RQ3** — Does ToT reduce final-answer errors compared to CoT?

Error classification uses Gemini as the primary judge. An independent neutral-judge validation sample (to check for self-evaluation bias via Cohen's kappa) is supported in `src/evaluation/error_judge.py` but was not completed for the current results — the neutral judge model available on Groq's free tier hit its daily token quota well before finishing the validation sample; this is documented as a known limitation rather than silently omitted.

---

## Repository Structure

```text
LLM_Project
│
├── config/
│   └── config.py           # single source of truth: models, prompts, paths, thresholds
├── data/
│   ├── raw/                 # untouched extraction output
│   ├── curated/              # manually reviewed Include/Category decisions
│   ├── processed/            # final_dataset.xlsx (frozen once experiments start)
│   └── results/               # experiment_results.csv / .xlsx (canonical results)
├── src/
│   ├── dataset/              # dataset extraction, curation, freezing
│   ├── models/                # Gemini / Groq / Mistral / neutral-judge clients
│   ├── prompts/                # cot.py, tot.py (real multi-branch ToT)
│   ├── experiments/             # resumable, checkpointed experiment runner
│   ├── evaluation/               # answer evaluator, error judge + neutral validation
│   ├── analysis/                  # RQ1/RQ2/RQ3 statistical analysis + figures
│   ├── parser/                     # response parsing (final answer extraction)
│   └── utils/                       # canonical I/O layer, shared stats helpers
├── tests/                    # pytest unit tests (evaluator, parser)
├── docs/                     # methodology notes, pilot results
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── stats/                 # judge-agreement / inferential test results (unpopulated -- see note above)
│   └── interactive/
│       ├── dashboard.html     # self-contained interactive dashboard (open directly, no server)
│       └── build/              # dashboard source (template + data + vendored Chart.js) + build script
├── thesis/                   # LaTeX dissertation source (University of Limerick MSc template)
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AnushreeM2701/LLM_Project.git
cd LLM_Project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

---

## Running the Pipeline

Run the full experiment collection (resumable — already-completed rows are skipped):

```bash
python -m src.experiments.run_experiment
```

Classify errors on incorrect responses (primary judge, then neutral-judge validation sample):

```bash
python -m src.evaluation.error_judge
```

Run the statistical analysis (RQ1/RQ2/RQ3 + execution time), each writing to `outputs/tables/` and `outputs/figures/`:

```bash
python -m src.analysis.descriptive              # accuracy tables (Wilson CI), Model x Prompt x Difficulty and Model x Category
python -m src.analysis.error_taxonomy           # RQ1: Model x Error Type contingency table + chi-square/Fisher independence test
python -m src.analysis.error_type_distribution  # Error Type x Difficulty, per model/prompt, balanced 40/40/40 pool
python -m src.analysis.error_type_frequency     # Error Type frequency by model, Hard tier, common-wrong-question pool
python -m src.analysis.error_heatmap            # Model x Difficulty x top-6 Error Type heatmap, balanced pool
python -m src.analysis.step_count               # RQ2: step count vs correctness (descriptive only)
python -m src.analysis.prompt_comparison        # RQ3: McNemar's Exact Test, CoT vs ToT, balanced 40/40/40 pool
python -m src.analysis.execution_time           # summary table, balanced 40/40/40 pool
python -m src.analysis.question_execution_time  # per-question execution time figures
python -m src.analysis.question_correctness     # per-question correctness heatmaps + hardest-questions ranking (full Hard tier, all 51)
python -m src.analysis.error_subtype_words      # error-description word frequency, Hard tier, common-wrong-question pool
python -m src.analysis.dataset_composition      # dataset composition figure
```

`error_type_distribution`, `error_type_frequency`, `error_heatmap`, `error_subtype_words`, `question_execution_time`, `step_count`, and `dataset_composition` are the exact static (PNG) counterparts of the panels in the interactive dashboard below — same balanced question pools, same common-wrong-question intersections, same whole-number axes, cross-checked against the dashboard's live output.

Run tests:

```bash
pytest tests/
```

---

## Viewing the Interactive Dashboard

`outputs/interactive/dashboard.html` is a single, fully self-contained file (Chart.js and all result data are embedded directly in it — no fetch/XHR calls), so it never needs a local server or a Live Server extension. Open it directly:

```bash
open outputs/interactive/dashboard.html
```

(On Windows/Linux, double-click the file or use `start outputs/interactive/dashboard.html` / `xdg-open outputs/interactive/dashboard.html`.)

### Rebuilding it

The dashboard's source lives in `outputs/interactive/build/` (`dashboard_template.html` + `dashboard_data.json` + the vendored `chart.umd.min.js`). If any of those change, regenerate `outputs/interactive/dashboard.html` with:

```bash
python outputs/interactive/build/build_dashboard.py
```

---

## Author

**Anushree Mahesha**

MSc Thesis, MSc in Data Science and Statistical Learning

Department of Mathematics and Statistics

University of Limerick

Supervisor: Meghana Kshirsagar, Department of Computer Science and Information Systems

2026
