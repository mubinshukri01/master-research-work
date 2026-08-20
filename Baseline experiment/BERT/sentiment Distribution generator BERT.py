import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# BERT MODELS
# ============================================================

models = [
    "Human + Annotation",
    "CNN",
    "GRU",
    "LSTM",
    "RNN"
]

negative = [42.65378749, 54.2450, 41.4337, 51.3472, 42.2471]
neutral  = [26.13116421, 19.0137, 20.9964, 27.6055, 35.6380]
positive = [31.21504830, 26.7412, 37.5699, 21.0473, 22.1149]

neg_count = [839, 1067, 815, 1010, 831]
neu_count = [514, 374, 413, 543, 701]
pos_count = [614, 526, 739, 414, 435]

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
        fontsize=30,
    )

    ax.text(
        x[i],
        negative[i]+neutral[i]/2,
        f"{neutral[i]:.1f}%\n({neu_count[i]})",
        ha='center',
        va='center',
        color='Black',
        fontsize=30,
    )

    ax.text(
        x[i],
        negative[i]+neutral[i]+positive[i]/2,
        f"{positive[i]:.1f}%\n({pos_count[i]})",
        ha='center',
        va='center',
        color='Black',
        fontsize=30,
    )

ax.set_title(
    "Sentiment Distribution for BERT Tokenizer Models",
    fontsize=30
)

ax.set_ylabel("Percentage (%)", fontsize=22)
ax.set_ylim(0, 100)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5,1.20),
    ncol=3,
    fontsize=18,
)

ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(
    "BERT_Sentiment_Distribution.png",
    dpi=600,
    bbox_inches='tight'
)

plt.show()