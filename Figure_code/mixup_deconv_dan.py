import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
celltype_num=3
'''
new method
'''
np.random.seed(42)

def mix_up_new():
    frac1 = np.random.uniform(low=0.0, high=1.0, size=1)
    frac1 = float(frac1[0])
    fracs = np.random.uniform(low=0.0, high=1.0, size=(celltype_num - 1,))
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    fracs = fracs * (1 - frac1)
    fracs = list(fracs)
    fracs.append(frac1)
    fracs = np.array(fracs)
    np.random.shuffle(fracs)
    return float(fracs[0])
data=[]
for i in range (4000):
    data.append(mix_up_new())
data=np.array(data)
# data = generate_simulated_data(3)[:, 0]

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
