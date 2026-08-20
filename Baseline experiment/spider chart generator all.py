import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
data = {
    'Model': [
        'NLP + CNN',
        'NLP + GRU',
        'NLP + LSTM',
        'NLP + RNN',
        # These four use the BERT *tokenizer* over a randomly-initialised
        # nn.Embedding - there is no BERT encoder in them (see
        # BERT/BERT + CNN.py:20 and :116, whose own header says
        # "# BERT TOKENIZER + CNN"). Labelling them "BERT + X" overclaims.
        'BERT Tokenizer + CNN',
        'BERT Tokenizer + GRU',
        'BERT Tokenizer + LSTM',
        'BERT Tokenizer + RNN'
    ],
    'Accuracy': [
        0.548551093,
        0.579562786,
        0.550584647,
        0.492628368,
        0.594814438,
        0.543975597,
        0.565327911,
        0.524656838
    ],
    'Precision-Macro': [
        0.542476925,
        0.571686876,
        0.548969195,
        0.488749505,
        0.592650886,
        0.552038383,
        0.555420290,
        0.529587436
    ],
    'Recall-Macro': [
        0.544199460,
        0.567421323,
        0.555277671,
        0.493114425,
        0.565520826,
        0.521697290,
        0.552309080,
        0.518812122
    ],
    'F1-Macro': [
        0.542078864,
        0.568995024,
        0.547143638,
        0.488646902,
        0.570844352,
        0.527637442,
        0.550323216,
        0.515491197
    ]
}

df = pd.DataFrame(data)

# Radar chart categories
categories = ['Accuracy', 'Precision-Macro', 'Recall-Macro', 'F1-Macro']
N = len(categories)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Plot each model
for _, row in df.iterrows():
    values = row[categories].tolist()
    values += values[:1]  # close the polygon

    ax.plot(angles, values, linewidth=2, label=row['Model'])
    ax.fill(angles, values, alpha=0.05)

# Configure chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)

ax.set_ylim(0.45, 0.65)
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65])
ax.set_yticklabels(['0.45', '0.50', '0.55', '0.60', '0.65'])

plt.title(
    'Performance Comparison of Ensemble Models',
    size=16,
    weight='bold',
    pad=20
)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.10),
    ncol=4,
    fontsize=9
)

plt.tight_layout()
plt.show()