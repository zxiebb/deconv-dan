import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
celltype_num=3
np.random.seed(42)

'''
scaden
'''
def create_subsample(celltypes, sparse=False):
    available_celltypes = celltypes
    if sparse:
        no_keep = np.random.randint(1, len(available_celltypes))
        keep = np.random.choice(
            list(range(len(available_celltypes))), size=no_keep, replace=False
        )
        available_celltypes = [available_celltypes[i] for i in keep]
    fracs = np.random.rand(len(available_celltypes))
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    fracs_complete = [0] * len(celltypes)
    for i, act in enumerate(available_celltypes):
        idx = celltypes.index(act)
        fracs_complete[idx] = fracs[i]
    return fracs_complete
result=[]
for i in range (2000):
    result.append(create_subsample(["A","B","C"],sparse=False))
for i in range(2000):
    result.append(create_subsample(["A", "B", "C"], sparse=True))
result=np.array(result)
result=result[:,0]
count=0
for i in range(4000):
    if 0.8<result[i]<1:
        count=count+1
print(count)
data=result
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
