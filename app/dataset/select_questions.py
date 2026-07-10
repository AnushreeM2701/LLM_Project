import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

CATEGORY = "probability"

INPUT_FILE = f"data/raw/{CATEGORY}_questions.csv"
OUTPUT_FILE = f"data/processed/selected_{CATEGORY}_questions.csv"


# ==========================================================
# Select Questions
# ==========================================================

def select_questions():

    df = pd.read_csv(INPUT_FILE)

    # Keep only manually selected questions
    selected_df = df[
        df["Selected"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "y", "true", "1"])
    ].copy()

    if selected_df.empty:
        raise ValueError("No questions have been selected.")

    # Assign sequential question numbers
    selected_df.insert(
        0,
        "Question Number",
        range(1, len(selected_df) + 1)
    )

    selected_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("Question Selection Completed Successfully")
    print("=" * 60)

    print(
        selected_df[
            [
                "Question Number",
                "Difficulty",
                "Original Level"
            ]
        ]
    )

    print("\nSelected Questions:")

    for _, row in selected_df.iterrows():

        print(
            f"Q{row['Question Number']} | "
            f"{row['Difficulty']} | "
            f"{row['Original Level']}"
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    select_questions()