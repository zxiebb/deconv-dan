import numpy as np
import matplotlib.pyplot as plt

# Categories and their CCC/RMSE scores
categories = ['Breast Cancer', 'Breast Cancer Subtype', 'Murine Cell Line']
methods = ['Deconv-DAN', 'scpDeconv', 'Scaden']

ccc_scores = {
    'Deconv-DAN': [0.926, 0.596, 0.892],
    'scpDeconv':  [0.944, 0.092, 0.669],
    'Scaden':     [0.961, 0.554, 0.268],
}
rmse_scores = {
    'Deconv-DAN': [0.038, 0.031, 0.096],
    'scpDeconv':  [0.030, 0.035, 0.114],
    'Scaden':     [0.026, 0.030, 0.156],
}

# Color map (first three of tab10)
cmap    = plt.get_cmap('tab10')
colors  = {m: cmap(i) for i, m in enumerate(methods)}

# Compute y-limits with 20% headroom
ymax_ccc  = max(v for vals in ccc_scores.values()  for v in vals) * 1.2
ymax_rmse = max(v for vals in rmse_scores.values() for v in vals) * 1.2

# Create a 2×3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.subplots_adjust(hspace=0.4, bottom=0.1)

for col, category in enumerate(categories):
    # ----- Top row: CCC -----
    ax = axes[0, col]
    for i, m in enumerate(methods):
        val = ccc_scores[m][col]
        ax.bar(i, val, color=colors[m], edgecolor='black')
        ax.text(i, val + ymax_ccc*0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(category, fontsize=14, weight='bold')
    ax.set_ylim(0, ymax_ccc)
    # draw baseline and top line
    ax.axhline(0,         color='black', linewidth=1)
    ax.axhline(ymax_ccc,  color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    if col == 0:
        ax.set_ylabel('CCC', fontsize=12)

    # ----- Bottom row: RMSE -----
    ax = axes[1, col]
    for i, m in enumerate(methods):
        val = rmse_scores[m][col]
        ax.bar(i, val, color=colors[m], edgecolor='black')
        ax.text(i, val + ymax_rmse*0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, ymax_rmse)
    # draw baseline and top line
    ax.axhline(0,          color='black', linewidth=1)
    ax.axhline(ymax_rmse,  color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    if col == 0:
        ax.set_ylabel('RMSE', fontsize=12)

plt.tight_layout()
plt.show()
