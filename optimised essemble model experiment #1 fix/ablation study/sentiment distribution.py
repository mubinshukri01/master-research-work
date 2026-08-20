import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Labels
# =====================================================
labels = [
    "Human\n(True Label)",
    "Epoch\n5",
    "Epoch\n6",
    "Epoch\n7",
    "Epoch\n8",
    "Epoch\n9",
    "Epoch\n10"
]

# =====================================================
# Human Label
# =====================================================
human_positive = 307
human_neutral  = 257
human_negative = 420

# =====================================================
# 3-Way Split
# =====================================================
positive_3 = np.array([307,315,294,317,297,336,288])
neutral_3  = np.array([257,243,260,250,270,225,259])
negative_3 = np.array([420,426,430,417,417,423,437])

# =====================================================
# 4-Way Split
# =====================================================
positive_4 = np.array([307,312,290,311,290,336,276])
neutral_4  = np.array([257,249,269,241,271,226,272])
negative_4 = np.array([420,423,425,432,423,422,436])

# =====================================================
# Convert to Percentage
# =====================================================
total3 = positive_3 + neutral_3 + negative_3
total4 = positive_4 + neutral_4 + negative_4

pos3 = positive_3 / total3 * 100
neu3 = neutral_3 / total3 * 100
neg3 = negative_3 / total3 * 100

pos4 = positive_4 / total4 * 100
neu4 = neutral_4 / total4 * 100
neg4 = negative_4 / total4 * 100

# =====================================================
# Plot
# =====================================================
fig, ax = plt.subplots(1,2, figsize=(18,7), sharey=True)

# -----------------------------
# 3-Way
# -----------------------------
ax[0].bar(labels, neg3, color='red', label='Negative')
ax[0].bar(labels, neu3, bottom=neg3, color='gray', label='Neutral')
ax[0].bar(labels, pos3, bottom=neg3+neu3, color='green', label='Positive')

ax[0].set_title("3-Way Split", fontsize=20, weight='bold')
ax[0].set_ylabel("Percentage (%)", fontsize=20, weight='bold')
ax[0].set_ylim(0,100)

for i in range(len(labels)):
    ax[0].text(i, neg3[i]/2,
               f"{neg3[i]:.1f}%\n({negative_3[i]})",
               ha='center', va='center',
               fontsize=20, fontweight='bold')

    ax[0].text(i,
               neg3[i]+neu3[i]/2,
               f"{neu3[i]:.1f}%\n({neutral_3[i]})",
               ha='center', va='center',
               fontsize=20, fontweight='bold')

    ax[0].text(i,
               neg3[i]+neu3[i]+pos3[i]/2,
               f"{pos3[i]:.1f}%\n({positive_3[i]})",
               ha='center', va='center',
               fontsize=20, fontweight='bold')

# -----------------------------
# 4-Way
# -----------------------------
ax[1].bar(labels, neg4, color='red')
ax[1].bar(labels, neu4, bottom=neg4, color='gray')
ax[1].bar(labels, pos4, bottom=neg4+neu4, color='green')

ax[1].set_title("4-Way Split", fontsize=20, weight='bold')
ax[1].set_ylim(0,100)

for i in range(len(labels)):
    ax[1].text(i, neg4[i]/2,
               f"{neg4[i]:.1f}%\n({negative_4[i]})",
               ha='center', va='center',
               fontsize=20, fontweight='bold')

    ax[1].text(i,
               neg4[i]+neu4[i]/2,
               f"{neu4[i]:.1f}%\n({neutral_4[i]})",
               ha='center', va='center',
               fontsize=20, fontweight='bold')

    ax[1].text(i,
               neg4[i]+neu4[i]+pos4[i]/2,
               f"{pos4[i]:.1f}%\n({positive_4[i]})",
               ha='center', va='center',
               fontsize=20, fontweight='bold')

# =====================================================
# Overall Title
# =====================================================
fig.suptitle(
    "Sentiment Distribution Comparison Across 3-Way and 4-Way Splits",
    fontsize=18,
    weight='bold'
)

fig.legend(
    ['Negative','Neutral','Positive'],
    loc='upper center',
    ncol=3,
    bbox_to_anchor=(0.5,0.96),
    fontsize=20,
)

plt.tight_layout(rect=[0,0,1,0.93])

plt.savefig(
    "Sentiment_Distribution_3Way_vs_4Way.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()