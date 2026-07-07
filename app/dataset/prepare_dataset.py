import pandas as pd

# ---------- Configuration ----------

CATEGORY = "probability"

INPUT_FILE = f"data/processed/selected_{CATEGORY}_questions.csv"
OUTPUT_FILE = f"data/processed/final_{CATEGORY}_dataset.csv"

# -----------------------------------


def extract_ground_truth(solution):
    """
    Extract the content inside the LAST \\boxed{...}
    Handles nested braces such as:
    \\boxed{\\frac{17}{24}}
    """

    start = solution.rfind("\\boxed{")

    if start == -1:
        return "Not Found"

    start += len("\\boxed{")

    brace_count = 1
    answer = ""

    while start < len(solution):

        char = solution[start]

        if char == "{":
            brace_count += 1

        elif char == "}":
            brace_count -= 1

            if brace_count == 0:
                break

        answer += char
        start += 1

    return answer.strip()


def prepare_dataset():

    df = pd.read_csv(INPUT_FILE)

    final_df = pd.DataFrame()

    # -------------------------------
    # Question Information
    # -------------------------------

    final_df["Question Number"] = df["Question Number"]
    final_df["Question"] = df["Question"]
    final_df["Original Level"] = df["Original Level"]
    final_df["Difficulty"] = df["Difficulty"]

    # -------------------------------
    # Ground Truth
    # -------------------------------

    final_df["Ground Truth Solution"] = df["Solution"]

    final_df["Ground Truth Answer"] = (
        df["Solution"]
        .apply(extract_ground_truth)
    )

    final_df["Ground Truth Steps"] = ""

    # -------------------------------
    # Baseline
    # -------------------------------

    final_df["Baseline Output"] = ""
    final_df["Baseline Answer"] = ""
    final_df["Baseline Steps"] = ""

    # -------------------------------
    # Chain-of-Thought
    # -------------------------------

    final_df["CoT Output"] = ""
    final_df["CoT Answer"] = ""
    final_df["CoT Steps"] = ""

    # -------------------------------
    # Tree-of-Thought
    # -------------------------------

    final_df["ToT Output"] = ""
    final_df["ToT Answer"] = ""
    final_df["ToT Steps"] = ""

    # -------------------------------
    # Evaluation
    # -------------------------------

    final_df["Baseline Correct"] = ""
    final_df["CoT Correct"] = ""
    final_df["ToT Correct"] = ""

    final_df["Baseline Error Type"] = ""
    final_df["CoT Error Type"] = ""
    final_df["ToT Error Type"] = ""

    # -------------------------------
    # Save
    # -------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("Dataset Prepared Successfully")
    print("=" * 60)
    print(f"Saved to:\n{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()