import pandas as pd
import os
from sklearn import preprocessing as pp
import numpy as np
import argparse
import random
dataset = 'murine_cellline'
data_dir = ''
random_type = "celltype"
ref_sample_num = 100
target_sample_num = 1000
SaveResultsDir = 'target_murine_cellline/'
sample_size=100
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='murine_cellline', help='The name of benchmarking datasets')
parser.add_argument("--cudaindex", type=str, default='3', help='The index of GPU')
args = parser.parse_args()
dataset = args.dataset
index =args.cudaindex
os.environ['CUDA_VISIBLE_DEVICES'] = index
if dataset == 'breast':
    ref_dataset_name = "breast_cellprotein_reference.csv"
    ref_metadata_name = "breast_annotation_reference.csv"
    target_dataset_name = "breast_cellprotein_target.csv"
    target_metadata_name = "breast_annotation_target.csv"
    type_list = ['VA', 'AV', 'BA', 'FI', 'HS', 'IM']
elif dataset == 'breast_cluster':
    ref_dataset_name = "reference_breast_cellprotein_cluster.csv"
    ref_metadata_name = "reference_breast_annotation_cluster.csv"
    target_dataset_name = "target_breast_cellprotein_cluster.csv"
    target_metadata_name = "target_breast_annotation_cluster.csv"
    type_list = ['AV1','AV2','AV3','BA','FI1','FI2','HS','IM1','IM2','IM3','IM4','PE1','PE2','VA1','VA2','VA3']
elif dataset == 'breast_cluster_epithelial':
    ref_dataset_name = "reference_breast_cellprotein_epithelial.csv"
    ref_metadata_name = "reference_breast_annotation_epithelial.csv"
    target_dataset_name = "target_breast_cellprotein_epithelial.csv"
    target_metadata_name = "target_breast_annotation_epithelial.csv"
    type_list = ['AP1','AP2','AP3','AP4','BA1','BA2','BL1','BL2','HS1','HS2']
elif dataset == 'murine_cellline_simulated':
    ref_dataset_name = 'murine_cellprotein_N2.csv'
    ref_metadata_name = 'murine_annotation_N2.csv'
    target_dataset_name = 'murine_cellprotein_nanoPOTS.csv'
    target_metadata_name = 'murine_annotation_nanoPOTS.csv'
    type_list = ['C10', 'SVEC', 'RAW']
elif dataset == 'murine_cellline_simulated_reverse_direction':
    ref_dataset_name ='murine_cellprotein_nanoPOTS.csv'
    ref_metadata_name ='murine_annotation_nanoPOTS.csv'
    target_dataset_name =  'murine_cellprotein_N2.csv'
    target_metadata_name =  'murine_annotation_N2.csv'
    type_list = ['C10', 'SVEC', 'RAW']
elif dataset == 'integrated':
    ref_dataset_name = 'integrated_reference_cellprotein.csv'
    ref_metadata_name = 'integrated_reference_annotation.csv'
    target_dataset_name = 'integrated_target_cellprotein.csv'
    target_metadata_name = 'integrated_target_annotation.csv'
    type_list = ['U','M','P']
elif dataset == 'CDC_mel_to_mon':
    ref_dataset_name = 'melanoma.csv'
    ref_metadata_name = 'annotation_CDC.csv'
    target_dataset_name = 'monocyte.csv'
    target_metadata_name = 'annotation_CDC.csv'
    type_list = ['G1','S','G2']
elif dataset == 'CDC_mon_to_mel':
    ref_dataset_name = 'monocyte.csv'
    ref_metadata_name = 'annotation_CDC.csv'
    target_dataset_name = 'melanoma.csv'
    target_metadata_name = 'annotation_CDC.csv'
    type_list = ['G1','S','G2']
elif dataset == 'human_cellline':
    ref_dataset_name = 'glioma_cellprotein_reference_clean.csv'
    ref_metadata_name = 'glioma_annotation_reference_clean.csv'
    target_dataset_name = 'glioma_cellprotein_target.csv'
    type_list = ['CT', 'IT', 'LE', 'PAN', 'MVP']
elif dataset == '450K_simulated':
    ref_dataset_name= 'reference_450K.csv'
    ref_metadata_name = 'reference_450K_annotation.csv'
    target_dataset_name= 'target_450K.csv'
    target_metadata_name= "target_450K_annotation.csv"
    type_list = ['neu', 'nk', 'cd4','cd8','b','mono']
elif dataset == 'EPIC_simulated':
    ref_dataset_name= 'reference_EPIC.csv'
    ref_metadata_name = 'reference_EPIC_annotation.csv'
    target_dataset_name= 'target_EPIC.csv'
    target_metadata_name= "target_EPIC_annotation.csv"
    type_list = ['neu', 'nk', 'cd4','cd8','b','mono']
elif dataset == 'sequencing':
    ref_dataset_name= 'reference_sequencing.csv'
    ref_metadata_name = 'reference_sequencing_annotation.csv'
    target_dataset_name= 'target_sequencing.csv'
    target_metadata_name= "target_sequencing_annotation.csv"
    type_list = ['neu', 'nk', 'cd4','cd8','b','mono']
celltype_num = len(type_list)
feature_num=0
def sample_normalize(data):
    mm = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    data = mm.fit_transform(data.T).T
    return data
def mixup():
    global feature_num
    train_data_x, train_data_y = mixup_dataset(ref_dataset_name, ref_metadata_name,ref_sample_num)
    target_data_x, target_data_y = mixup_dataset(target_dataset_name, target_metadata_name,target_sample_num)  
    used_features = list(set(train_data_x.columns.tolist()).intersection(set(target_data_x.columns.tolist())))
    feature_num = len(used_features)
    train_data_x=train_data_x[used_features]
    target_data_x = target_data_x[used_features]
    return target_data_x,target_data_y
def mixup_dataset(dataset, metadata, sample_num):
    sim_data_x = []
    sim_data_y = []
    ref_data_x, ref_data_y = load_dataset(dataset, metadata)
    for i in range(int(sample_num)):
        sample, label = mixup_cells(ref_data_x, ref_data_y, type_list)
        sim_data_x.append(sample)
        sim_data_y.append(label)
    sim_data_x = pd.concat(sim_data_x, axis=1).T
    sim_data_y = pd.DataFrame(sim_data_y, columns=type_list)
    sim_data_x_scale = sample_normalize(sim_data_x)
    sim_data_x_scale = pd.DataFrame(sim_data_x_scale, columns=sim_data_x.columns)
    sim_data_x = sim_data_x_scale
    return sim_data_x, sim_data_y
def load_dataset(dataset, metadata):
    filename = data_dir+dataset
    data_x = pd.read_csv(filename, header=0, index_col=0)
    data_x = data_x.fillna(0)
    metadata_filename =data_dir+metadata
    data_y = pd.read_csv(metadata_filename, header=0, index_col=0)
    return data_x, data_y
def mixup_fraction_test(celltype_num):
    fracs = np.random.rand(celltype_num)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs
def mixup_cells(x, y, celltypes):
    available_celltypes = celltypes
    celltype_num = len(available_celltypes)
    fracs = mixup_fraction_test(celltype_num)
    samp_fracs = np.multiply(fracs, sample_size)
    samp_fracs = list(map(round, samp_fracs))
    artificial_samples = []
    for i in range(celltype_num):
        ct = available_celltypes[i]
        cells_sub = x.loc[np.array(y[random_type] == ct), :]
        cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
        cells_sub = cells_sub.iloc[cells_fraction, :]
        artificial_samples.append(cells_sub)
    df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = df_samp.sum(axis=0)
    return df_samp, fracs
for good in range (20):
    random.seed(good)
    np.random.seed(good)
    target_data_x,target_data_y= mixup()
    target_data_x.to_csv(SaveResultsDir+str(dataset)+"_test_x_"+str(good)+".csv")
    target_data_y.to_csv(SaveResultsDir+str(dataset)+"_test_y_"+str(good)+".csv")