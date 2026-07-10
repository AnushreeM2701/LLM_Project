import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_FILE = "data/results/experiment_results.csv"
OUTPUT_FOLDER = "outputs/figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(RESULT_FILE)

# ----------------------------------------
# Overall Accuracy
# ----------------------------------------

overall_accuracy = (
    df["Answer Correct"].mean() * 100
)

# ----------------------------------------
# Model Accuracy
# ----------------------------------------

model_accuracy = (
    df.groupby("Model")["Answer Correct"]
      .mean() * 100
)

print("\nOverall Accuracy")
print(f"{overall_accuracy:.2f}%")

print("\nModel-wise Accuracy")
print(model_accuracy)

# ----------------------------------------
# Plot
# ----------------------------------------

labels = ["Overall"] + list(model_accuracy.index)

values = [overall_accuracy] + list(model_accuracy.values)

plt.figure(figsize=(7,5))

plt.bar(labels, values)

plt.ylim(0,100)

plt.ylabel("Accuracy (%)")

plt.title("Overall and Model-wise Accuracy")

for i, v in enumerate(values):

    plt.text(
        i,
        v + 1,
        f"{v:.1f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "outputs/figures/model_accuracy.png",
    dpi=300
)

plt.show()