import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Final experiment\\optimised essemble model experiment #1 fix\\alpha_grid_search_results.xlsx")

plt.figure(figsize=(8,5))

plt.plot(
    df["alpha"],
    df["f1_macro"],
    "-o",
    linewidth=2.8,
    markersize=6
)

best = df.loc[df["f1_macro"].idxmax()]

plt.scatter(
    best["alpha"],
    best["f1_macro"],
    s=500,
    marker="*",
    color="red",
    edgecolors="black",
    zorder=5,
    label="Optimal α",
)

plt.annotate(
    f'({best["alpha"]:.2f}, {best["f1_macro"]:.4f})',
    xy=(best["alpha"], best["f1_macro"]),
    xytext=(8,12),
    textcoords="offset points",
    fontsize=20
)

plt.xlabel("Alpha Weight", fontsize=20)
plt.ylabel("Macro F1-Score", fontsize=20)
plt.title("Effect of Alpha Weight on Ensemble Performance", fontsize=30, fontweight="bold", pad=20)
plt.xticks(df["alpha"], fontsize=15)
plt.yticks(fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("Alpha_Grid_Search_Result.png", dpi=600, bbox_inches="tight")
plt.show()