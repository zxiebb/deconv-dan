import numpy as np
import matplotlib.pyplot as plt

# Methods and their RMSE scores on each platform
methods = ['Deconv-DAN', 'EMeth_normal', 'MethylResolver', 'Iced-t', 'FARDEEP', 'DCQ']
rmse_scores = {
    'Deconv-DAN':    [0.0520, 0.0330, 0.0402],
    'EMeth_normal':  [0.0483, 0.14687,0.148],
    'MethylResolver':[0.0410, 0.0179, 0.041],
    'Iced-t':        [0.0490, 0.0136, 0.0414],
    'FARDEEP':       [0.0530, 0.0180, 0.0335],
    'DCQ':           [0.0660, 0.0422, 0.091],
}

# Color map and dictionary
cmap = plt.get_cmap('tab10')
base_colors = cmap(np.arange(len(methods)))
colors = {m: base_colors[i] for i, m in enumerate(methods)}

# Platforms
platforms = ['450K', '850K', 'WGBS']

# Compute global y-max with 20% headroom
all_vals = [v for vals in rmse_scores.values() for v in vals]
y_max = max(all_vals) * 1.2

# Create 1×3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, ax in enumerate(axes):
    x = np.arange(len(methods))
    values = [rmse_scores[m][idx] for m in methods]

    # Draw bars and annotate
    for i, m in enumerate(methods):
        val = values[i]
        ax.bar(x[i], val, color=colors[m], edgecolor='black')
        ax.text(x[i], val + y_max*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Styling
    ax.set_title(platforms[idx], fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, y_max)
    ax.axhline(y=y_max, color='black', linewidth=1)  # top horizontal line
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)              # hide top border

    if idx == 0:
        ax.set_ylabel('RMSE', fontsize=12)

plt.tight_layout()
plt.show()
