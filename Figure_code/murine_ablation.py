
import numpy as np
import matplotlib.pyplot as plt

# Methods and data
methods = ['Deconv-DAN', 'scpDeconv', 'Scaden']
directions = ['Melanoma→Monocyte', 'Monocyte→Melanoma']

ccc_scores = {
    'Deconv-DAN': [0.096, 0.0859],
    'scpDeconv': [0.0720, 0.1550],
    'Scaden': [0.1878, 0.0616],
}
rmse_scores = {
    'Deconv-DAN': [0.096, 0.0860],
    'scpDeconv': [0.0850, 0.300],
    'Scaden': [0.1010, 0.2150],
}

# Color map – first three of tab10
cmap = plt.get_cmap('tab10')
colors = {m: cmap(i) for i, m in enumerate(methods)}

# Compute y-limits with 20% headroom
ymax_rmse = max(v for vals in rmse_scores.values() for v in vals) * 1.2

# Prepare 1×4 subplots with increased space between axes
fig, axes = plt.subplots(
    1, 4,
    figsize=(18, 5),
    sharey=False,
    gridspec_kw={'wspace': 0.3}  # more horizontal spacing
)

configs = [
    ('CCC',  ccc_scores,  ymax_rmse,  0),
    ('CCC',  ccc_scores,  ymax_rmse,  1),
    ('RMSE', rmse_scores, ymax_rmse, 0),
    ('RMSE', rmse_scores, ymax_rmse, 1),
]

for ax, (metric, scores, y_max, dir_idx) in zip(axes, configs):
    # Draw baseline and top border
    # ax.axhline(0,     color='black', linewidth=1)
    # ax.axhline(y_max, color='black', linewidth=1)
    # # Draw right border
    # ax.axvline(x=len(methods)-0.5, color='black', linewidth=1)

    # Plot bars
    bars = ax.bar(
        np.arange(len(methods)),
        [scores[m][dir_idx] for m in methods],
        color=[colors[m] for m in methods],
        edgecolor='black'
    )

    # Annotate values and method names
    for bar, m in zip(bars, methods):
        val = bar.get_height()
        xpos = bar.get_x() + bar.get_width()/2
        ax.text(xpos, val + y_max*0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(xpos, -y_max*0.02,
                m, ha='center', va='top', rotation=45, fontsize=8)

    # Styling
    title = f"{metric} ({directions[dir_idx]})"
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_ylim(0, y_max)
    ax.yaxis.grid(True, alpha=0.3)

    # Remove default spines (we've drawn our own)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # Keep bottom spine so the x-axis line remains

    # Remove x-ticks
    ax.set_xticks([])

    # Y-label only on first subplot
    if ax is axes[0]:
        ax.set_ylabel('Score', fontsize=12)

plt.tight_layout()
plt.show()

