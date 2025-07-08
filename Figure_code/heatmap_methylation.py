import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
target=pd.read_csv("D:/Data/Methylation/sequencing_simulated/target/target_sequencing.csv",index_col=0)
reference=pd.read_csv("D:/Data/Methylation/sequencing_simulated/reference/reference_sequencing_mean_value.csv",index_col=0)
plt.figure(figsize=(10, 10))
fig, ax = plt.subplots()
reference=reference.fillna(0)
reference=reference.to_numpy()
target=target.fillna(0)
target=target.to_numpy()
all_corr_matrix=np.corrcoef(reference, target)
all_corr_matrix=all_corr_matrix[0:6,:]
all_corr_matrix=all_corr_matrix[:,6:]
print(all_corr_matrix)
mask=all_corr_matrix<0.9
sns.heatmap(all_corr_matrix,vmin=0.9, vmax=1.0,mask=mask,annot=True)
x=np.arange(12)
labels=["B","B","CD4","CD4","CD8","CD8","mono","mono","neu","neu","NK","NK"]
ax.set_xticks(x, labels, rotation='vertical')
y=np.arange(6)
labels=["B","CD4","CD8","mono","neu","NK"]
ax.set_yticks(y, labels, rotation='vertical')
plt.show()