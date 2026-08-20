import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Data (from provided table)
# ==============================
data = {
    "Model": ["CNN", "GRU", "LSTM", "RNN"],
    "Accuracy": [0.548551093035079, 0.57956278596848, 0.550584646670056, 0.492628368073208],
    "Precision-Macro": [0.542476924690389, 0.57168687637886, 0.548969195309183, 0.488749504876666],
    "Recall-Macro": [0.544199460147757, 0.567421323086511, 0.555277671448342, 0.493114424796087],
    "F1-Macro": [0.542078864405431, 0.568995024150377, 0.547143637568309, 0.488646901984606],
    # REMOVED - two stale arrays used to sit here:
    #   "F1-Micro":    [0.537366548, 0.571428571, 0.565836299, 0.493645145]
    #   "F1-Weighted": [0.53036245,  0.573980474, 0.568079316, 0.49437447]
    # In single-label multiclass, micro-F1 is identically accuracy. Those
    # values disagree with the "Accuracy" row above, which proves they came
    # from a different, older run than the numbers in keras_word_*.xlsx.
    # The correct weighted-F1 values are 0.55053610, 0.57818636, 0.55075283,
    # 0.49474001 - read them from the workbooks, do not retype them.
}

df = pd.DataFrame(data)

# ==============================
# Metrics to plot
# ==============================
metrics = [
    "Accuracy",
    "Precision-Macro",
    "Recall-Macro",
    "F1-Macro",
    # "F1-Micro",
    # "F1-Weighted"
]

models = df["Model"]
x = np.arange(len(models))
width = 0.13

# ==============================
# Plot
# ==============================
plt.figure(figsize=(14, 7))

for i, metric in enumerate(metrics):
    bars = plt.bar(
        x + i * width,
        df[metric],
        width,
        label=metric
    )

    # Value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

# ==============================
# Formatting
# ==============================
plt.xlabel("Model", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Metric Comparison – Keras Word Tokenizer", fontsize=14)
plt.xticks(x + width * (len(metrics) - 1) / 2, models)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
