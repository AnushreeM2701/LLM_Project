import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_FILE = "data/results/experiment_results.csv"
OUTPUT_FOLDER = "outputs/figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(RESULT_FILE)

accuracy = (
    df.groupby("Prompt")["Answer Correct"]
      .mean()
      .mul(100)
)

accuracy = accuracy[
    ["baseline","cot","tot"]
]

plt.figure(figsize=(7,5))

bars = plt.bar(
    accuracy.index,
    accuracy.values
)

plt.ylim(0,100)

plt.ylabel("Accuracy (%)")

plt.title("Accuracy by Prompting Technique")

for bar in bars:

    y = bar.get_height()

    plt.text(

        bar.get_x()+bar.get_width()/2,

        y+1,

        f"{y:.1f}%",

        ha="center"

    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/prompt_accuracy.png",
    dpi=300
)

plt.show()