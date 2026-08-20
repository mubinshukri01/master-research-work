import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP Ensemble Results
data = {
    'Model': ['CNN', 'GRU', 'LSTM', 'RNN'],
    'Accuracy': [0.548551093, 0.579562786, 0.550584647, 0.492628368],
    'Precision': [0.542476925, 0.571686876, 0.548969195, 0.488749505],
    'Recall': [0.544199460, 0.567421323, 0.555277671, 0.493114425],
    'F1': [0.542078864, 0.568995024, 0.547143638, 0.488646902]
}

df = pd.DataFrame(data)

# Radar chart setup
categories = ['Accuracy', 'Precision', 'Recall', 'F1']
N = len(categories)

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

for _, row in df.iterrows():
    values = row[categories].tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2.5,
        marker='o',
        markersize=6,
        label=row['Model']
    )
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=20,)
ax.tick_params(axis='x',pad=20)

ax.set_ylim(0.45, 0.60)
ax.set_yticks([0.45, 0.50, 0.55, 0.60])
ax.set_yticklabels([0.45, 0.50, 0.55, 0.60], fontsize=18)

plt.title(
    'Performance Comparison of NLP-Based Baseline Models',
    fontsize=30,
    fontweight='bold',
    pad=25
)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.10),
    ncol=2,
    fontsize=20
)

plt.tight_layout()
plt.savefig('NLP_Ensemble_SpiderChart.png', dpi=600, bbox_inches='tight')
plt.show()