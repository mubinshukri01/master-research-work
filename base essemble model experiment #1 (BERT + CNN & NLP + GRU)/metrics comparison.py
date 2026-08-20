import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Create DataFrame (from your data)
# ==============================
data = {
    "Model": ["CNN", "LSTM", "GRU", "RNN", "mBERT"],
    "Accuracy": [0.587697, 0.554652, 0.56482, 0.513981, 0.573971],
    "Precision": [0.59121, 0.545723, 0.560584, 0.517431, 0.565251],
    "Recall": [0.557815, 0.544183, 0.562217, 0.498411, 0.568107],
    "F1": [0.564732, 0.544832, 0.559618, 0.485532, 0.565765]
}

df = pd.DataFrame(data)

# ==============================
# Metrics to plot
# ==============================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
models = df["Model"]

x = np.arange(len(models))
width = 0.2

# ==============================
# Plot
# ==============================
plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    bars = plt.bar(
        x + i * width,
        df[metric],
        width,
        label=metric
    )

    # Add value labels
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
plt.title("Performance Metric Comparison (BERT Tokenizer)", fontsize=14)
plt.xticks(x + width * (len(metrics) - 1) / 2, models)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
