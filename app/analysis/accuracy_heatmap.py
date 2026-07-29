
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS_FILE = "data/results/experiment_results copy.csv"
OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)

category_order = {"ALG":0,"NT":1,"PROB":2,"COMB":3}
df["CategoryPrefix"] = df["Question ID"].str.split("_").str[0]
df["QuestionInCategory"] = df["Question ID"].str.extract(r'_(\d+)$').astype(int)
df["Plot Question"] = df["CategoryPrefix"].map(category_order)*10 + df["QuestionInCategory"]
df["Correct"] = df["Answer Correct"].astype(bool)

GREEN = "#2ca02c"
RED = "#d62728"

pairs=[("gemini","cot"),("gemini","tot"),
       ("groq","cot"),("groq","tot"),
       ("mistral","cot"),("mistral","tot")]

fig = plt.figure(figsize=(12,12.5))
gs = fig.add_gridspec(
    3,2,
    width_ratios=[9.5,2],
    height_ratios=[1,1,1.45],
    hspace=0.22,
    wspace=0.06
)

for r,diff in enumerate(["Easy","Medium","Hard"]):

    ax = fig.add_subplot(gs[r,0])
    sub = df[df["Difficulty"]==diff]

    piv = sub.pivot_table(
        index="Plot Question",
        columns=["Model","Prompt"],
        values="Correct",
        aggfunc="first"
    )

    piv = piv.reindex(columns=pd.MultiIndex.from_tuples(pairs))
    piv = piv[~piv.all(axis=1)]

    nrows = len(piv)

    group_gap = 0.35
    xs = [0,1,2+group_gap,3+group_gap,4+2*group_gap,5+2*group_gap]

    ax.set_xlim(-0.6, xs[-1]+1.2)

    if diff=="Hard":
        cell_h = 0.55
        pad_y = 0.22
        label_size = 6
    else:
        cell_h = 0.78
        pad_y = 0.10
        label_size = 9

    ax.set_ylim(nrows+0.5,-2.2)

    # draw cells
    for y,(_,row) in enumerate(piv.iterrows()):
        for x,val in enumerate(row):
            color = GREEN if bool(val) else RED
            patch = FancyBboxPatch(
                (xs[x]+0.08,y+pad_y),
                0.84,
                cell_h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=0.8,
                edgecolor="white",
                facecolor=color
            )
            ax.add_patch(patch)

    ax.set_yticks([i+0.5 for i in range(nrows)])
    ax.set_yticklabels([f"Q{q}" for q in piv.index],fontsize=label_size)
    ax.set_xticks([])
    ax.tick_params(length=0)

    # Header positions
    if diff == "Hard":
        model_y = -3.1
        cot_y = -1.0
    else:
        model_y = -1.55
        cot_y = -0.65

    model_centers = [0.5, 2.85, 5.2]

    for c, m in zip(model_centers, ["Gemini", "Groq", "Mistral"]):
        ax.text(
            c,
            model_y,
            m,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold"
        )

    for x, l in zip(xs, ["CoT", "ToT"] * 3):
        ax.text(
            x + 0.5,
            cot_y,
            l,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(diff,loc="left",fontsize=14,fontweight="bold",pad=14)

    for s in ax.spines.values():
        s.set_visible(False)

    # -------- Accuracy Table ----------
    tax = fig.add_subplot(gs[r,1])
    tax.axis("off")

    text = "Accuracy (%)\n\n"
    text += f"{'Model':<9}{'CoT':>7}{'ToT':>8}\n"
    text += "-"*26 + "\n"

    for m in ["gemini","groq","mistral"]:
        cot = sub[(sub.Model.str.lower()==m)&(sub.Prompt.str.lower()=="cot")]["Correct"].mean()*100
        tot = sub[(sub.Model.str.lower()==m)&(sub.Prompt.str.lower()=="tot")]["Correct"].mean()*100
        text += f"{m.capitalize():<9}{cot:>6.1f}%{tot:>8.1f}%\n"

    tax.text(
        0.02,0.98,text,
        va="top",
        ha="left",
        fontsize=8.8,
        family="monospace",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="black",
            linewidth=1.4
        )
    )

fig.suptitle("Model Accuracy",fontsize=18,fontweight="bold",y=0.99)

plt.savefig(
    os.path.join(OUT,"model_accuracy_boxes.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved:",os.path.join(OUT,"model_accuracy_boxes.png"))
