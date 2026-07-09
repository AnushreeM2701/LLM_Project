# Statistical Study of Mathematical Reasoning Errors in Large Language Models

## Overview

This project evaluates the mathematical reasoning ability of multiple Large Language Models (LLMs) on probability problems using different prompting techniques.

The study compares:

- Gemini
- Qwen
- GPT-OSS

Prompting techniques:

- Baseline Prompting
- Chain-of-Thought (CoT)
- Tree-of-Thought (ToT)

The project evaluates:

- Answer correctness
- Error classification
- Reasoning steps
- Execution time
- Statistical comparison between models

---

## Project Structure

app/
    dataset/
    evaluation/
    experiments/
    models/
    parser/
    prompts/
    utils/

data/
    raw/
    processed/
    results/

outputs/

---

## Installation

pip install -r requirements.txt

---

## Configure API Keys

Create a .env file

GOOGLE_API_KEY=xxxxxxxx

OPENROUTER_API_KEY=xxxxxxxx

---

## Run Dataset Preparation

python -m app.dataset.prepare_dataset

---

## Run Experiments

python -m app.experiments.run_experiment

---

## Outputs

The experiment generates

data/results/experiment_results.csv

which contains

- Question
- Model
- Prompt
- Ground Truth
- Model Response
- Final Answer
- Answer Correct
- Error Type
- Execution Time

---

## Future Work

- Statistical analysis
- Step counting
- Visualization
- Error taxonomy analysis