import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
celltype_num=3
'''
new method
'''
# def mix_up_new():
#     frac1 = np.random.uniform(low=0.0, high=1.0, size=1)
#     frac1 = float(frac1[0])
#     fracs = np.random.uniform(low=0.0, high=1.0, size=(celltype_num - 1,))
#     fracs_sum = np.sum(fracs)
#     fracs = np.divide(fracs, fracs_sum)
#     fracs = fracs * (1 - frac1)
#     fracs = list(fracs)
#     fracs.append(frac1)
#     fracs = np.array(fracs)
#     np.random.shuffle(fracs)
#     return float(fracs[0])
# result=[]
# for i in range (4000):
#     result.append(mix_up_new())
'''
dirichlet distribution
'''
# result=np.random.dirichlet(np.ones(celltype_num), 4000)
# result=np.array(result[:,0])
'''
scpDeconv
'''
# def mix_up_scp():
#     fracs = np.random.rand(celltype_num)
#     fracs_sum = np.sum(fracs)
#     fracs = np.divide(fracs, fracs_sum)
#     return float(fracs[0])
'''
scaden
'''
# def create_subsample(celltypes, sparse=False):
#     available_celltypes = celltypes
#     if sparse:
#         no_keep = np.random.randint(1, len(available_celltypes))
#         keep = np.random.choice(
#             list(range(len(available_celltypes))), size=no_keep, replace=False
#         )
#         available_celltypes = [available_celltypes[i] for i in keep]
#     fracs = np.random.rand(len(available_celltypes))
#     fracs_sum = np.sum(fracs)
#     fracs = np.divide(fracs, fracs_sum)
#     fracs_complete = [0] * len(celltypes)
#     for i, act in enumerate(available_celltypes):
#         idx = celltypes.index(act)
#         fracs_complete[idx] = fracs[i]
#     return fracs_complete
# result=[]
# for i in range (2000):
#     result.append(create_subsample(["A","B","C"],sparse=False))
# for i in range(2000):
#     result.append(create_subsample(["A", "B", "C"], sparse=True))
# result=np.array(result)
# result=result[:,0]
# count=0
# for i in range(4000):
#     if 0.8<result[i]<1:
#         count=count+1
# print(count)
'''
beta distribution
'''
# result=np.random.beta(1, 2, size=4000)
# result=np.array(result)
'''
TAPE
https://github.com/poseidonchan/TAPE/blob/main/TAPE/simulation.py
'''
import numpy as np
import matplotlib.pyplot as plt

# def generate_simulated_data(num_celltype, samplenum=4000, sparse=True, sparse_prob=0.5, rare=False, rare_percentage=0.4):
#     prop = np.random.dirichlet(np.ones(num_celltype), samplenum)
#     prop = prop / np.sum(prop, axis=1, keepdims=True)
#     if sparse:
#         for i in range(int(prop.shape[0] * sparse_prob)):
#             indices = np.random.choice(prop.shape[1], size=int(prop.shape[1] * sparse_prob), replace=False)
#             prop[i, indices] = 0
#         prop = prop / np.sum(prop, axis=1, keepdims=True)
#     if rare:
#         indices = np.random.choice(prop.shape[1], size=int(prop.shape[1] * rare_percentage), replace=False)
#         prop = prop / np.sum(prop, axis=1, keepdims=True)
#         for i in range(int(0.5 * prop.shape[0]) + int(rare_percentage * 0.5 * prop.shape[0])):
#             prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
#             buf = prop[i, indices].copy()
#             prop[i, indices] = 0
#             prop[i] = (1 - buf.sum()) * prop[i] / prop[i].sum()
#             prop[i, indices] = buf
#     return prop

# Generate and select data for cell type 0
data = generate_simulated_data(3)[:, 0]

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
