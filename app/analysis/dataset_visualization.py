import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# FILE PATHS
DATASET = "data/processed/final_dataset.xlsx"
OUTPUT_FOLDER = "outputs/figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LOAD DATA

df = pd.read_excel(DATASET)

summary = (
    df.groupby(["Category", "Difficulty"])
      .size()
      .unstack(fill_value=0)
)

# COLORS

colors = {
    "Algebra": "#A9C4E2",
    "Number Theory": "#C8E6B8",
    "Probability": "#FFD3B6",
    "Combinatorics": "#F4B6B6"
}

# FIGURE

fig, ax = plt.subplots(figsize=(10,10))

ax.set_xlim(-1.2,1.2)
ax.set_ylim(-1.2,1.2)

ax.set_aspect("equal")
ax.axis("off")

# DRAW COLOURED QUADRANTS

quadrants = [
    ("Algebra",90,180),
    ("Number Theory",0,90),
    ("Probability",180,270),
    ("Combinatorics",270,360)
]

for cat,start,end in quadrants:
    wedge = Wedge(
        center=(0,0),
        r=1,
        theta1=start,
        theta2=end,
        facecolor=colors[cat],
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(wedge)

# Divider Lines

ax.plot([0,0],[-1,1],color="black",linewidth=2)
ax.plot([-1,1],[0,0],color="black",linewidth=2)

# TEXT POSITIONS

positions = {
    "Algebra":(-0.48,0.48),
    "Number Theory":(0.48,0.48),
    "Probability":(-0.48,-0.48),
    "Combinatorics":(0.48,-0.48)
}

# WRITE TEXT

for category,(x,y) in positions.items():
    total = int(summary.loc[category].sum())
    easy = int(summary.loc[category]["Easy"])
    medium = int(summary.loc[category]["Medium"])
    hard = int(summary.loc[category]["Hard"])

    text = (
        f"{category}\n\n"
        f"Total : {total}\n\n"
        f"Easy      : {easy}\n"
        f"Medium : {medium}\n"
        f"Hard      : {hard}"
    )

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=13,
        weight="bold"
    )

# TITLE

plt.title(
    "Dataset Composition",
    fontsize=24,
    weight="bold",
    pad=25
)

# SAVE

output = os.path.join(
    OUTPUT_FOLDER,
    "dataset_composition.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("="*60)
print("Dataset Composition Figure Created")
print(output)
print("="*60)