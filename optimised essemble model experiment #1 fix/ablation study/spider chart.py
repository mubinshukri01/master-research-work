import numpy as np
import matplotlib.pyplot as plt



# ===========================
# Performance Metrics
# ===========================

categories = ['Accuracy', 'Precision', 'Recall', 'F1']
N = len(categories)

# Best 3-way (Epoch 8)
three_way = [
    0.861788618,   # Accuracy
    0.859640369,   # Precision
    0.862014637,   # Recall
    0.860572267    # F1
]

# Best 4-way (Epoch 10)
four_way = [
    0.870934959,   # Accuracy
    0.873351022,   # Precision
    0.869840686,   # Recall
    0.870440507    # F1
]

# Close the polygon
three_way += three_way[:1]
four_way += four_way[:1]

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ===========================
# Plot
# ===========================

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)

# Start at top
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Category labels
plt.xticks(angles[:-1], categories, fontsize=20)

# Radial limits
ax.set_ylim(0.84, 0.88)

# Radial ticks
ax.set_yticks([0.84,0.85,0.86,0.87,0.88])
ax.set_yticklabels(
    ['0.84','0.85','0.86','0.87','0.88'],
    fontsize=20
)

# Plot lines
ax.plot(angles, three_way,
        linewidth=2,
        color='tab:blue',
        marker='o',
        label='3-way (60-30-10) epoch 8')

ax.fill(angles, three_way,
        color='tab:blue',
        alpha=0.20)

ax.plot(angles, four_way,
        linewidth=2,
        color='tab:orange',
        marker='o',
        label='4-way (60-20-10-10) epoch 10')

ax.fill(angles, four_way,
        color='tab:orange',
        alpha=0.20)

# Grid
ax.grid(True, linestyle='--', alpha=0.6)

# Legend
plt.legend(loc='lower center',
           bbox_to_anchor=(0.5,-0.18),
           ncol=2,
           frameon=True,
           fontsize=20)

plt.title("Ablation Study: 3-way vs 4-way Split Performance comparison", pad=20, fontsize=20, weight='bold')
plt.tight_layout()
plt.show()