import numpy as np
import matplotlib.pyplot as plt

# 1. Only the 'Murine Cell Line' category remains
categories = ['Murine Cell Line']
x = np.arange(len(categories))  # just [0]
methods = ['Deconv-DAN', 'scpDeconv', 'Scaden']

# 2. RMSE data for that single category in each panel
panel1 = [[0.085], [0.217], [0.207]]    # Nonuniform Proportions
panel2 = [[0.094], [0.146], [0.162]]   # Mix-Up Size Sensitivity
panel3 = [[0.086], [0.257], [0.293]]   # High-Purity Targets

panels = [
    ('Nonuniform Proportions', panel1),
    ('Mix-Up Size Sensitivity', panel2),
    ('High-Purity Targets', panel3),
]

#  Compute global y-max + 20% headroom
all_vals = [v for _, mat in panels for vals in mat for v in vals]
y_max = max(all_vals) * 1.2

# 3. Create a 1×3 row of plots, share Y-axis
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Bar positions
width = 0.2
offsets = (np.arange(len(methods)) - 1) * width  # [-0.2, 0, +0.2]

# Reuse first three colors from 'tab10'
cmap = plt.get_cmap('tab10')
base_colors = cmap(np.arange(6))
colors = {
    'Deconv-DAN': base_colors[0],
    'scpDeconv':  base_colors[1],
    'Scaden':     base_colors[2],
}

for ax, (title, data_matrix) in zip(axes, panels):
    # draw bars & annotate
    for i, method in enumerate(methods):
        val = data_matrix[i][0]
        ax.bar(x + offsets[i], val, width,
               color=colors[method], edgecolor='black')
        ax.text(x + offsets[i],
                val + y_max * 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    # styling
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xticks([])                   # no x-ticks since only one category
    ax.set_ylim(0, y_max)               # 20% headroom
    ax.axhline(y=y_max, color='black', linewidth=1)  # top horizontal line
    ax.grid(axis='y', alpha=0.3)

    # hide top spine, keep right spine for the vertical border
    ax.spines['top'].set_visible(False)
    # ax.spines['right'] remains visible

# y-axis label on the leftmost plot
axes[0].set_ylabel('RMSE', fontsize=12)

# Legend below all subplots
fig.legend(
    methods,
    loc='lower center',
    ncol=3,
    fontsize=12,
    bbox_to_anchor=(0.5, -0.12),
    frameon=False
)

# Super-title and layout adjustments
fig.suptitle(
    'RMSE of Deconv-DAN, scpDeconv, and Scaden\non Complex In Silico Tasks',
    fontsize=16, weight='bold'
)
fig.subplots_adjust(bottom=0.18, wspace=0.4)

plt.tight_layout()
plt.show()
