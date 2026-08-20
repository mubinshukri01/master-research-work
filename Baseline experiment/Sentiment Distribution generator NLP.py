import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# NLP MODELS
# ============================================================

models = [
    "Human + Annotation",
    "CNN",
    "GRU",
    "LSTM",
    "RNN"
]

negative = [42.65378749, 40.16268429, 44.58566345, 33.55363498, 37.26487036]
neutral  = [26.13116421, 30.40162684, 23.58922217, 28.01220132, 31.87595323]
positive = [31.21504830, 29.43568887, 31.82511439, 38.43416370, 30.85917641]

neg_count = [839, 790, 877, 660, 733]
neu_count = [514, 598, 464, 551, 627]
pos_count = [614, 579, 626, 756, 607]

# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))

ax.bar(x, negative, color='red', label='Negative')
ax.bar(x, neutral, bottom=negative, color='gray', label='Neutral')
ax.bar(
    x,
    positive,
    bottom=np.array(negative)+np.array(neutral),
    color='green',
    label='Positive'
)

# Labels
for i in range(len(models)):

    ax.text(
        x[i],
        negative[i]/2,
        f"{negative[i]:.1f}%\n({neg_count[i]})",
        ha='center',
        va='center',
        color='Black',
        fontsize=20,
    )

    ax.text(
        x[i],
        negative[i]+neutral[i]/2,
        f"{neutral[i]:.1f}%\n({neu_count[i]})",
        ha='center',
        va='center',
        color='Black',
        fontsize=20,
    )

    ax.text(
        x[i],
        negative[i]+neutral[i]+positive[i]/2,
        f"{positive[i]:.1f}%\n({pos_count[i]})",
        ha='center',
        va='center',
        color='Black',
        fontsize=20,
    )

ax.set_title(
    "Sentiment Distribution for NLP Tokenizer Models",
    fontsize=30,
    pad=10,
)

ax.set_ylabel("Percentage (%)", fontsize=22)
ax.set_ylim(0, 100)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.2),
    ncol=3,
    fontsize=20,
)

ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(
    "NLP_Sentiment_Distribution.png",
    dpi=600,
    bbox_inches='tight'
)

plt.show()