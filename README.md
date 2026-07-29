# A Statistical Study of Mathematical Reasoning Errors in Large Language Models

## Overview

This repository contains the implementation and experimental results for the Master's dissertation:

**A Statistical Study of Mathematical Reasoning Errors in Large Language Models**

The project investigates the mathematical reasoning performance of Large Language Models (LLMs) using different prompting strategies on a curated dataset of mathematical reasoning problems.

Three Large Language Models are evaluated:

- Gemini
- Groq
- Mistral

using two prompting strategies:

- Chain-of-Thought (CoT)
- Tree-of-Thought (ToT)

The study focuses on analysing:

- Mathematical reasoning accuracy
- Error classification
- Execution time
- Prompt effectiveness
- Model-wise performance comparison

---

# Repository Structure

```text
LLM_Project
│
├── app/
│   ├── analysis/          
│   ├── dataset/           
│   ├── evaluation/        
│   ├── experiments/       
│   ├── models/            
│   ├── parser/            
│   ├── prompts/           
│   └── utils/             
├── data/
│   ├── raw/               
│   ├── processed/         
│   ├── candidates/        
│   ├── results/           
│   ├── aime_questions/    
│   └── aime_answers/      
│
├── docs/
│
├── outputs/
│   ├── figures/           
│   ├── tables/            
│   └── logs/              
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/AnushreeM2701/LLM_Project.git
cd LLM_Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a `.env` file in the project root directory.

Example:

```text
GOOGLE_API_KEY=your_google_api_key
```

---

# Running the Experiments

Run the complete experiment pipeline:

```bash
python -m app.experiments.run_experiment
```

The experiment automatically evaluates every question across all configured models and prompting strategies.

---

# Output Files

Experimental results are saved in:

```text
data/results/
```

Generated figures are saved in:

```text
outputs/figures/
```

Generated statistical tables are saved in:

```text
outputs/tables/
```

---

# Experimental Evaluation

The current implementation evaluates:

- Answer correctness
- Mathematical error classification
- Execution time
- Model-wise performance
- Prompt-wise performance
- Difficulty-wise performance

The generated outputs include:

- Experiment result files
- Model accuracy summaries
- Error distribution analysis
- Execution time analysis
- Publication-ready figures and tables

---

# Technologies Used

- Python
- Google Gemini API
- Pandas
- NumPy
- Matplotlib
- OpenPyXL

---

# Current Limitations

The current implementation focuses on evaluating model-generated mathematical reasoning using a curated dataset.

The following components are reserved for future work:

- Automated AIME dataset generation
- Automated AIME question and solution scraping
- Ground-truth reasoning step extraction
- Reasoning step comparison between model responses and reference solutions
- Evaluation on additional mathematical reasoning benchmarks

---

# Author

**Anushree Mahesha**

Master's Dissertation

Department of Computer Science and Information Systems

University of Limerick

2026