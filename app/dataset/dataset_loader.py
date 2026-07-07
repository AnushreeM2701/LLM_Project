from datasets import load_dataset

def load_probability_dataset():
    """
    Load the Counting & Probability subset
    from the Hendrycks MATH dataset.
    """
    dataset = load_dataset(
        "EleutherAI/hendrycks_math",
        "counting_and_probability"
    )
    return dataset

if __name__ == "__main__":
    dataset = load_probability_dataset()
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(dataset)

    print("\n")

    print("=" * 60)
    print("Training Questions:", len(dataset["train"]))
    print("Test Questions:", len(dataset["test"]))
    print("\nColumns Available:")
    print(dataset["train"].column_names)
    sample = dataset["train"][0]

    print("\nSample Question")
    print("-" * 40)

    for key, value in sample.items():

        print(f"{key}:\n{value}\n")