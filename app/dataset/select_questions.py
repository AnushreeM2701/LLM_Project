import pandas as pd

# ---------- Configuration ----------

CATEGORY = "probability"

INPUT_FILE = f"data/raw/{CATEGORY}_questions.csv"
OUTPUT_FILE = f"data/processed/selected_{CATEGORY}_questions.csv"

# -----------------------------------


def main():

    df = pd.read_csv(INPUT_FILE)

    # Keep only selected questions
    selected_df = df[df["Selected"].astype(str).str.upper() == "YES"].copy()

    # Add Question ID
    selected_df.insert(
        0,
        "Question Number",
        range(1, len(selected_df) + 1)
    )

    # Save
    selected_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 50)
    print(f"Selected Questions : {len(selected_df)}")
    print(f"Saved to : {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()