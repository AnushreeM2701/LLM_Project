from dataclasses import dataclass, field
from typing import Callable, List

from config.config import TOT_BRANCH_COUNT, TOT_BRANCH_TEMPERATURE
from src.models.base import ModelResponse
from src.parser.response_parser import parse_selection_response


def build_branch_prompt(question: str, branch_index: int, branch_count: int) -> str:

    return f"""
Solve the following mathematics problem.

This is one independent solution attempt (attempt {branch_index + 1} of
{branch_count} that will later be compared against each other). Work through
your own reasoning path on its merits — do not hedge between multiple
methods, commit to one approach and carry it through completely.

Show your reasoning as clear, numbered steps.

Format your response exactly as:

Step 1: ...
Step 2: ...
Step 3: ...

Final Answer: <answer>

Problem:
{question}
"""


def build_selection_prompt(question: str, branches: List[str]) -> str:

    branch_blocks = "\n\n".join(
        f"--- Candidate Branch {i + 1} ---\n{text}"
        for i, text in enumerate(branches)
    )

    return f"""
You are evaluating independently generated candidate solutions to a
mathematics problem, as part of a Tree-of-Thought search.

Evaluate each candidate branch for mathematical soundness. Select the ONE
branch most likely to be correct and well-reasoned. If multiple branches
reach the same final answer through valid reasoning, prefer the clearest
derivation. If branches disagree, select based on which reasoning is most
rigorous and free of errors — do not simply pick the majority answer
without checking the reasoning.

==================================================
PROBLEM
==================================================

{question}

==================================================
CANDIDATE BRANCHES
==================================================

{branch_blocks}

==================================================
OUTPUT FORMAT
==================================================

Selected Branch: <branch number>
Justification: <one sentence, maximum 25 words>
"""


@dataclass
class ToTResult:
    final_response: str
    branches: List[str]
    selected_branch: int
    selection_justification: str
    selection_raw: str
    total_latency_s: float
    model_version: str
    call_count: int
    branch_model_versions: List[str] = field(default_factory=list)


def generate_branches(
    question: str,
    generate_fn: Callable[..., ModelResponse],
    branch_count: int = None,
) -> List[ModelResponse]:

    branch_count = branch_count or TOT_BRANCH_COUNT

    return [
        generate_fn(
            build_branch_prompt(question, i, branch_count),
            temperature=TOT_BRANCH_TEMPERATURE,
        )
        for i in range(branch_count)
    ]


_SELECTION_MAX_TOKENS = 300

def select_best(
    question: str,
    branch_results: List[ModelResponse],
    generate_fn: Callable[..., ModelResponse],
) -> ModelResponse:

    branch_texts = [r.text for r in branch_results]

    selection_prompt = build_selection_prompt(question, branch_texts)

    return generate_fn(selection_prompt, temperature=0, max_tokens=_SELECTION_MAX_TOKENS)


def generate_tot_response(
    question: str,
    generate_fn: Callable[..., ModelResponse],
    branch_count: int = None,
) -> ToTResult:

    branch_count = branch_count or TOT_BRANCH_COUNT

    branch_results = generate_branches(question, generate_fn, branch_count)

    selection_result = select_best(question, branch_results, generate_fn)

    parsed_selection = parse_selection_response(selection_result.text)

    selected_idx = parsed_selection["selected_branch"] - 1
    if not (0 <= selected_idx < len(branch_results)):
        selected_idx = 0  # malformed selection -> degrade to first branch, don't crash the run

    total_latency = sum(r.latency_s for r in branch_results) + selection_result.latency_s

    return ToTResult(
        final_response=branch_results[selected_idx].text,
        branches=branch_texts_from(branch_results),
        selected_branch=selected_idx + 1,
        selection_justification=parsed_selection["selection_justification"],
        selection_raw=selection_result.text,
        total_latency_s=total_latency,
        model_version=branch_results[0].model_version if branch_results else selection_result.model_version,
        call_count=branch_count + 1,
        branch_model_versions=[r.model_version for r in branch_results],
    )


def branch_texts_from(branch_results: List[ModelResponse]) -> List[str]:
    return [r.text for r in branch_results]
