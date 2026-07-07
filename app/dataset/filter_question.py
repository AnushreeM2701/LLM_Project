import pandas as pd

from dataset_loader import load_probability_dataset


# Keywords to identify probability-related questions
KEYWORDS = [
    "probability",
    "random",
    "randomly",
    "odds",
    "chance",
    "without replacement",
    "with replacement",
    "independent",
    "dependent",
    "conditional"
]
OBJECT_KEYWORDS = [
    "coin",
    "coins",
    "dice",
    "die",
    "card",
    "cards",
    "deck",
    "urn",
    "spinner"
]


def map_difficulty(level):
    """
    Convert Hendrycks MATH levels
    into our experiment difficulty.
    """

    if level in ["Level 1", "Level 2"]:
        return "Easy"

    elif level == "Level 3":
        return "Medium"

    else:
        return "Hard"


def is_probability_question(question):

    question = question.lower()

    # Strong probability keywords
    if any(keyword in question for keyword in KEYWORDS):
        return True

    # Probability objects
    if any(keyword in question for keyword in OBJECT_KEYWORDS):

        # Only keep if the question also mentions randomness/probability
        if "random" in question or "probability" in question or "odds" in question:
            return True

    return False


def filter_questions():

    dataset = load_probability_dataset()

    filtered_questions = []

    # Combine train and test
    all_questions = list(dataset["train"]) + list(dataset["test"])

    for item in all_questions:

        question = item["problem"]

        if is_probability_question(question):

            filtered_questions.append({

                "Question": question,

                "Original Level": item["level"],

                "Difficulty": map_difficulty(item["level"]),

                "Solution": item["solution"]

            })

    df = pd.DataFrame(filtered_questions)

    print("=" * 60)
    print(f"Probability Questions Found : {len(df)}")
    print("=" * 60)

    print(df.head())

    df.to_csv(
        "data/raw/probability_questions.csv",
        index=False
    )

    print("\nSaved to data/raw/probability_questions.csv")


if __name__ == "__main__":

    filter_questions()