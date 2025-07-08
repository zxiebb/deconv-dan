import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Deconv-DAN', 'EMeth_normal', 'MethylResolver', 'Iced-t', 'FARDEEP', 'DCQ']
rmse = [0.0388, 0.14,   0.0756, 0.0756, 0.0689, 0.103]
times = [23,     175.8, 228,    5250,   208,    14]

# Colors
cmap = plt.get_cmap('tab10')
colors = cmap(np.arange(len(methods)))

# Compute y‐axis limits with 20% headroom
y_max_rmse = max(rmse) * 1.2
y_max_time = max(times) * 1.2

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharex=False)

x = np.arange(len(methods))

# ---- RMSE plot ----
bars1 = ax1.bar(x, rmse, color=colors, edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.set_ylabel('RMSE')
ax1.set_title('B cell RMSE in WGBS Dataset')

# Annotate
for rect, val in zip(bars1, rmse):
    ax1.text(rect.get_x() + rect.get_width()/2,
             val + y_max_rmse * 0.02,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=8)

# Style: limits, top line, grid, spine
ax1.set_ylim(0, y_max_rmse)
ax1.axhline(y=y_max_rmse, color='black', linewidth=1)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)

# ---- Time plot ----
bars2 = ax2.bar(x, times, color=colors, edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=45, ha='right')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Time for 1000 Samples')

# Annotate
for rect, val in zip(bars2, times):
    ax2.text(rect.get_x() + rect.get_width()/2,
             val + y_max_time * 0.02,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=8)

# Style: limits, top line, grid, spine
ax2.set_ylim(0, y_max_time)
ax2.axhline(y=y_max_time, color='black', linewidth=1)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
