import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
celltype_num=3
np.random.seed(42)
'''
scpDeconv
'''
def mix_up_scp():
    fracs = np.random.rand(celltype_num)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return float(fracs[0])
data=[]
for i in range(4000):
        data.append(mix_up_scp())
data=np.array(data)
# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(data, bins=30, alpha=0.8)

# Add mean line and annotation
mean_val = data.mean()
ax.axvline(mean_val, linestyle='--', linewidth=1.5)
ax.text(mean_val, ax.get_ylim()[1] * 0.9, f'Mean: {mean_val:.2f}',
        ha='center', va='center')

# Styling
ax.set_title('Distribution of Cell Type 0 Proportions', fontsize=16, weight='bold')
ax.set_xlabel('Proportion of Cell Type 0', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.grid(axis='y', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
