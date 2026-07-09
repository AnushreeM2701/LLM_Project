import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

CATEGORY = "probability"

INPUT_FILE = f"data/raw/{CATEGORY}_questions.csv"
OUTPUT_FILE = f"data/processed/selected_{CATEGORY}_questions.csv"

# Number of questions to select
QUESTION_SELECTION = {
    "Easy": 1,
    "Medium": 1,
    "Hard": 5
}

# Random seed for reproducibility
RANDOM_STATE = 42


# ==========================================================
# Select Questions
# ==========================================================

def select_questions():

    df = pd.read_csv(INPUT_FILE)

    selected_questions = []

    question_number = 1

    # Always select in this order
    difficulty_order = ["Easy", "Medium", "Hard"]

    for difficulty in difficulty_order:

        count = QUESTION_SELECTION[difficulty]

        difficulty_df = df[df["Difficulty"] == difficulty]

        if len(difficulty_df) < count:

            raise ValueError(
                f"Only {len(difficulty_df)} {difficulty} questions are available, "
                f"but {count} were requested."
            )

        sampled = difficulty_df.sample(
            n=count,
            random_state=RANDOM_STATE
        ).copy()

        sampled.insert(
            0,
            "Question Number",
            range(
                question_number,
                question_number + len(sampled)
            )
        )

        question_number += len(sampled)

        selected_questions.append(sampled)

    final_df = pd.concat(
        selected_questions,
        ignore_index=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("Question Selection Completed Successfully")
    print("=" * 60)

    print(
        final_df[
            [
                "Question Number",
                "Difficulty",
                "Original Level"
            ]
        ]
    )

    print("\nSelected Questions:")

    for _, row in final_df.iterrows():

        print(
            f"Q{row['Question Number']} | "
            f"{row['Difficulty']} | "
            f"{row['Original Level']}"
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":

    select_questions()