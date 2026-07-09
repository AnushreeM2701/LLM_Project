from app.dataset.dataset_loader import load_probability_dataset


def explore_dataset():
    dataset = load_probability_dataset()

    train_data = dataset["train"]

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Training Questions : {len(train_data)}")
    print()

    # Count questions by difficulty level
    level_counts = {}

    for question in train_data:
        level = question["level"]

        if level not in level_counts:
            level_counts[level] = 0

        level_counts[level] += 1

    print("Questions by Difficulty")
    print("-" * 30)

    for level in sorted(level_counts.keys()):
        print(f"{level}: {level_counts[level]}")

    print("\n")

    # Show one example from each level
    print("=" * 60)
    print("SAMPLE QUESTION FROM EACH LEVEL")
    print("=" * 60)

    shown_levels = set()

    for question in train_data:

        level = question["level"]

        if level not in shown_levels:

            print(f"\n{level}")
            print("-" * 30)
            print(question["problem"])
            print()

            shown_levels.add(level)

        if len(shown_levels) == len(level_counts):
            break


if __name__ == "__main__":
    explore_dataset()