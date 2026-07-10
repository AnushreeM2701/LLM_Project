import pandas as pd
import hashlib
# ==========================================================
# Configuration
# ==========================================================

CATEGORY = "probability"

INPUT_FILE = f"data/processed/selected_{CATEGORY}_questions.csv"
OUTPUT_FILE = f"data/processed/final_{CATEGORY}_dataset.csv"

# ==========================================================
# Generate Question ID
# ==========================================================

def generate_question_id(question):
    """
    Generate a unique ID from the question text.
    """

    return hashlib.sha256(
        str(question).encode("utf-8")
    ).hexdigest()

# ==========================================================
# Extract Ground Truth Answer
# ==========================================================

def extract_ground_truth(solution):
    """
    Extract the complete answer inside \\boxed{...}
    Handles nested braces correctly.
    """

    solution = str(solution)

    start = solution.rfind(r"\boxed{")

    if start == -1:
        return ""

    # Move to first character after \boxed{
    i = start + len(r"\boxed{")

    brace_count = 1

    answer = ""

    while i < len(solution):

        char = solution[i]

        if char == "{":
            brace_count += 1

        elif char == "}":
            brace_count -= 1

            if brace_count == 0:
                break

        answer += char

        i += 1

    return answer.strip()


# ==========================================================
# Prepare Final Dataset
# ==========================================================

def prepare_dataset():

    df = pd.read_csv(INPUT_FILE)

    final_df = pd.DataFrame()

    final_df["Question Number"] = df["Question Number"]
    
    final_df["Question ID"] = (
    df["Question"]
    .apply(generate_question_id)
    )

    final_df["Question"] = df["Question"]

    final_df["Original Level"] = df["Original Level"]

    final_df["Difficulty"] = df["Difficulty"]

    # Complete ground truth solution
    final_df["Ground Truth Solution"] = df["Solution"]

    # Extract final boxed answer
    final_df["Ground Truth Answer"] = (
        df["Solution"]
        .apply(extract_ground_truth)
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("Final Dataset Created Successfully")
    print("=" * 60)

    print(final_df)

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":

    prepare_dataset()