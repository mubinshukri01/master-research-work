import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# BERT-TOKENIZER baseline results (single models, not an ensemble).
#
# Two caveats on these numbers:
#  1. They are BERT *tokenizer* + head. The scripts import BertTokenizer
#     only and train a randomly-initialised nn.Embedding - there is no
#     BERT encoder (BERT + CNN.py:20, :116).
#  2. They have no on-disk provenance: the export blocks in all four
#     BERT + *.py scripts are commented out (:295-314 and siblings), so
#     these literals are the only record. Superseded by
#     "Baseline experiment/canonical rerun/results/".
data = {
    'Model': ['CNN', 'GRU', 'LSTM', 'RNN'],
    'Accuracy': [0.594814438, 0.543975597, 0.565327911, 0.524656838],
    'Precision': [0.592650886, 0.552038383, 0.555420290, 0.529587436],
    'Recall': [0.565520826, 0.521697290, 0.552309080, 0.518812122],
    'F1': [0.570844352, 0.527637442, 0.550323216, 0.515491197]
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
ax.set_xticklabels(categories, fontsize=12)

ax.set_ylim(0.50, 0.62)
ax.set_yticks([0.50, 0.53, 0.56, 0.59, 0.62])

plt.title(
    'Performance Comparison of BERT-Tokenizer Baseline Models',
    fontsize=14,
    fontweight='bold',
    pad=25
)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.10),
    ncol=2
)

plt.tight_layout()
plt.savefig('BERT_Ensemble_SpiderChart.png', dpi=600, bbox_inches='tight')
plt.show()