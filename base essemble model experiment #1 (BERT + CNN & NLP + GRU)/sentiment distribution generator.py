import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "Final experiment\\Mubin Master Finalize experiment.xlsx"

# Read the correct sheet
df = pd.read_excel(file_path, sheet_name="Sentiment distribution ensemble")

# Keep only the relevant columns
df = df.iloc[0:6, 0:4]

# Rename columns (ensures correct labels)
df.columns = ["Model", "Positive", "Negative", "Neutral"]

# Convert data types
df["Model"] = df["Model"].astype(str)
df["Positive"] = pd.to_numeric(df["Positive"])
df["Negative"] = pd.to_numeric(df["Negative"])
df["Neutral"] = pd.to_numeric(df["Neutral"])

models = df["Model"]
positive = df["Positive"]
negative = df["Negative"]
neutral = df["Neutral"]

# Calculate totals
totals = positive + negative + neutral

# Convert to percentages
pos_pct = positive / totals * 100
neg_pct = negative / totals * 100
neu_pct = neutral / totals * 100

x = np.arange(len(models))

plt.figure(figsize=(10,6))

# Stacked bars
plt.bar(x, pos_pct, color="green", label="Positive")
plt.bar(x, neg_pct, bottom=pos_pct, color="red", label="Negative")
plt.bar(x, neu_pct, bottom=pos_pct+neg_pct, color="gray", label="Neutral")

# Add labels inside bars
for i in range(len(models)):

    plt.text(x[i], pos_pct[i]/2,
             f"{pos_pct[i]:.1f}%\n({positive[i]})",
             ha="center", va="center", color="black", fontsize=20)

    plt.text(x[i], pos_pct[i] + neg_pct[i]/2,
             f"{neg_pct[i]:.1f}%\n({negative[i]})",
             ha="center", va="center", color="black", fontsize=20)

    plt.text(x[i], pos_pct[i] + neg_pct[i] + neu_pct[i]/2,
             f"{neu_pct[i]:.1f}%\n({neutral[i]})",
             ha="center", va="center", color="black", fontsize=20)

plt.xticks(x, models, fontsize=20)

plt.ylabel("Percentage of Instances (%)", fontsize=22)
plt.title("Sentiment Distribution Across Deep Learning Models", fontsize=30, fontweight='bold')

plt.ylim(0,100)

plt.legend(
    fontsize=20,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.2),
    ncol=3,
)
plt.tight_layout()
plt.show()