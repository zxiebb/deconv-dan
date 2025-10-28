import pandas as pd
import os
import numpy as np
import argparse
import random
data_dir = 'Download/'
random_type = "CellType"
train_sample_num = 100
target_sample_num = 1000
SaveResultsDir = ''
# if not os.path.isdir(SaveResultsDir):
#     os.mkdir(SaveResultsDir)
sample_size=100
parser = argparse.ArgumentParser()
parser.add_argument("--cudaindex", type=str, default='3', help='The index of GPU')
args = parser.parse_args()
index =args.cudaindex
os.environ['CUDA_VISIBLE_DEVICES'] = index
target_dataset_name = 'murine_nanoPOTS_SCP_exp.csv'
target_metadata_name = 'murine_nanoPOTS_SCP_meta.csv'
type_list = ['C10', 'SVEC', 'RAW']
celltype_num = len(type_list)
def mixup():
    target_data_x, target_data_y = mixup_dataset(target_dataset_name, target_metadata_name,target_sample_num)  
    return target_data_x,target_data_y
def mixup_dataset(dataset, metadata, sample_num):
    sim_data_x = []
    sim_data_y = []
    train_data_x, train_data_y = load_dataset(dataset, metadata)
    for _ in range(int(sample_num)):
        sample, label = mixup_cells(train_data_x, train_data_y, type_list)
        sim_data_x.append(sample)
        sim_data_y.append(label)
    sim_data_x = pd.concat(sim_data_x, axis=1).T
    sim_data_x=sim_data_x/sample_size
    sim_data_y = pd.DataFrame(sim_data_y, columns=type_list)
    return sim_data_x, sim_data_y
def load_dataset(dataset, metadata):
    filename = data_dir+dataset
    data_x = pd.read_csv(filename, header=0, index_col=0)
    data_x = data_x.fillna(0)
    metadata_filename =data_dir+metadata
    data_y = pd.read_csv(metadata_filename, header=0, index_col=0)
    return data_x, data_y
def mixup_fraction_test(celltype_num):
    while True:
        fracs = np.random.rand(celltype_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)
        if fracs[0]>0.5:
            break
    return fracs
def mixup_cells(x, y, celltypes):
    celltype_num = len(celltypes)
    fracs = mixup_fraction_test(celltype_num)
    samp_fracs = np.multiply(fracs, sample_size)
    samp_fracs = [round(i) for i in samp_fracs]
    artificial_samples = []
    for i in range(celltype_num):
        ct = celltypes[i]
        cells_sub = x.loc[np.array(y[random_type] == ct), :]
        cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
        cells_sub = cells_sub.iloc[cells_fraction, :]
        artificial_samples.append(cells_sub)
    df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = df_samp.sum(axis=0)
    return df_samp, fracs
for good in range (1):
    random.seed(good)
    np.random.seed(good)
    target_data_x,target_data_y= mixup()
    target_data_x.to_csv(SaveResultsDir+"murine_C10_at_least_50_test_x_"+str(good)+".csv")
    target_data_y.to_csv(SaveResultsDir+"murine_C10_at_least_50_test_y_"+str(good)+".csv")