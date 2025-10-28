import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
import random
from sklearn.neighbors import KNeighborsClassifier
DATA_DIRECTORY = ''
TRAIN_SAMPLE_NUM = 4000
TAREGT_SAMPLE_NUM = 1000
BATCH_SIZE = 100
EPOCHS = 30
LEARNING_RATE = 0.00001
RESULT_DIRECTORY= 'Result_glioma/'
SAMPLE_SIZE=100
TARGET_TYPE="real"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='murine_cellline', help='The name of benchmarking datasets')
parser.add_argument("--cudaindex", type=str, default='3', help='The index of GPU')
args = parser.parse_args()
DATASET = args.dataset
INDEX=args.cudaindex
os.environ['CUDA_VISIBLE_DEVICES'] = INDEX
TRAIN_DATASET_NAME= 'glioma_reference_cellprotein.csv'
TRAIN_ANNOTATION_NAME = 'glioma_annotation_reference.csv'
TARGET_DATASET_NAME = None
TARGET_ANNOTATION_NAME=None
TYPE_LIST = ['CT','IT','LE','PAN','MVP']
CCC_LIST=[]
RMSE_LIST=[]
CORR_LIST=[]
CELLTYPE_NUM = len(TYPE_LIST)
feature=None
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)
def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    loss = torch.mean(XX + YY - XY - YX)
    return loss
def mse_loss(predictions,targets):
    diff = predictions - targets
    return diff.pow(2).reshape(-1).mean()
def concordance_corrcoef(a,b):
    corr = np.corrcoef(a, b)[0, 1]
    std_a, std_b = a.std(), b.std()
    mean_a, mean_b = a.mean(), b.mean()
    num = 2 * corr * std_a * std_b
    den = a.var() + b.var() + (mean_a - mean_b) ** 2
    return num / den
def compute_metrics(preds_df,gt_df):
    gt_aligned = gt_df[preds_df.columns]
    x = preds_df.values.flatten()
    y = gt_aligned.values.flatten()
    return (
        concordance_corrcoef(x, y),
        sqrt(mean_squared_error(x, y)),
        pearsonr(x,y)[0]
    )
def normalize(matrix):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(matrix.T).T
def plot_loss_histories(histories,labels):
    if TARGET_TYPE=="simulated":
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    elif TARGET_TYPE=="real":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes_flat = axes.flat
    for ax, hist, lbl in zip(axes.flat, histories, labels):
        ax.plot(hist)
        ax.set_title(lbl, loc='center')
    for ax in axes_flat[len(histories):]:
        fig.delaxes(ax)
    out_path = RESULT_DIRECTORY+"loss_histories"+str(good)+".png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
def plot_prediction_scatter(preds,gt):
    ncols = len(TYPE_LIST) + 1
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    def single_plot(ax, x, y, title):
        ccc_val = concordance_corrcoef(y, x)
        rmse_val = sqrt(mean_squared_error(y, x))
        corr_val = pearsonr(y, x)[0]
        ax.scatter(x, y, s=2)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, linestyle='--')
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color='orange')
        txt = (
            f"CCC={ccc_val:.3f}\n"
            f"RMSE={rmse_val:.3f}\n"
            f"Corr={corr_val:.3f}"
        )
        ax.text(
            0.05, 
            0.95 * np.max(y),
            txt,
            fontsize=8,
            verticalalignment='top'
        )
        ax.set_title(title)
        ax.set_xlim(0, np.max(y))
        ax.set_ylim(0, np.max(y))
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")
    xs = preds.values.flatten()
    ys = gt[preds.columns].values.flatten()
    single_plot(axs[0], xs, ys, "All samples")
    for idx, key in enumerate(TYPE_LIST, start=1):
        x_t = preds[key].values
        y_t = gt[key].values
        single_plot(axs[idx], x_t, y_t, key)
    out_file =RESULT_DIRECTORY+"prediction_scatter_plot"+str(good)+".jpg"
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
def mixup():
    global feature
    train_data_x, train_data_y = mixup_dataset(TRAIN_DATASET_NAME, TRAIN_ANNOTATION_NAME,TRAIN_SAMPLE_NUM,"train")
    if TARGET_TYPE == "simulated":
        target_data_x, target_data_y = mixup_dataset(TARGET_DATASET_NAME, TARGET_ANNOTATION_NAME,TAREGT_SAMPLE_NUM,"test")  
    elif TARGET_TYPE == "real":
        target_data_x = load_real_data(TARGET_DATASET_NAME)
        target_data_y=None
    used_features = sorted(set(train_data_x.columns) & set(target_data_x.columns))
    feature = used_features
    train_data_x = train_data_x[used_features]
    target_data_x = target_data_x[used_features]
    return train_data_x,train_data_y,target_data_x,target_data_y
def mixup_dataset(dataset, metadata, sample_num,labels):
    sim_data_x = []
    sim_data_y = []
    data_x, data_y = load_dataset(dataset, metadata)
    for _ in range(int(sample_num)):
        sample, label = mixup_cells(data_x, data_y, TYPE_LIST,labels)
        sim_data_x.append(sample)
        sim_data_y.append(label)
    sim_data_x = pd.concat(sim_data_x, axis=1).T
    sim_data_y = pd.DataFrame(sim_data_y, columns=TYPE_LIST)
    sim_data_x_normalized = normalize(sim_data_x)
    sim_data_x_normalized = pd.DataFrame(sim_data_x_normalized, columns=sim_data_x.columns)
    sim_data_x = sim_data_x_normalized
    return sim_data_x, sim_data_y
def load_dataset(dataset, metadata):
    filename = DATA_DIRECTORY+dataset
    data_x = pd.read_csv(filename, header=0, index_col=0)
    data_x = data_x.fillna(0)
    metadata_filename =DATA_DIRECTORY+metadata
    data_y = pd.read_csv(metadata_filename, header=0, index_col=0)
    return data_x, data_y
def load_real_data(dataset):
    filename =DATA_DIRECTORY+dataset
    data_x = pd.read_csv(filename, header=0, index_col=0)
    data_x = data_x.fillna(0)
    row=data_x.index
    column=data_x.columns
    data_x = normalize(data_x)
    data_x=pd.DataFrame(data_x,index=row,columns=column)
    return data_x
def mixup_fraction_train(celltype_num):  
    type_range=np.arange(celltype_num)
    max_type=np.random.choice(type_range,p=EXPERT_PROPORTION)
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
    max_index=np.argmax(fracs)
    if max_index!=max_type:
        temp=fracs[max_index]
        fracs[max_index]=fracs[max_type]
        fracs[max_type]=temp
    return fracs
def mixup_fraction_test(celltype_num):
    fracs = np.random.rand(celltype_num)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs
def mixup_cells(x, y, celltypes,label):
    celltype_num = len(celltypes)
    if label=="train":
        fracs = mixup_fraction_train(celltype_num)
    elif label=="test":
        fracs = mixup_fraction_test(celltype_num)
    samp_fracs = np.multiply(fracs, SAMPLE_SIZE)
    samp_fracs = [round(i) for i in samp_fracs]
    artificial_samples = []
    for i in range(celltype_num):
        ct = celltypes[i]
        cells_sub = x.loc[np.array(y["celltype"] == ct), :]
        cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
        cells_sub = cells_sub.iloc[cells_fraction, :]
        artificial_samples.append(cells_sub)
    df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = df_samp.sum(axis=0)
    return df_samp, fracs
def train(train_data_x,train_data_y,target_data_x,target_data_y):
    train_data_x = train_data_x.to_numpy()
    train_data_y = train_data_y.to_numpy()
    train_data = torch.FloatTensor(train_data_x)
    train_labels = torch.FloatTensor(train_data_y)
    train_dataset = Data.TensorDataset(train_data, train_labels)
    train_train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,worker_init_fn=worker_init_fn,generator=g)
    target_data_x = target_data_x.to_numpy()
    if TARGET_TYPE == "simulated":
        target_data_y = target_data_y.to_numpy()
    elif TARGET_TYPE == "real":
        target_data_y = np.random.rand(target_data_x.shape[0], CELLTYPE_NUM)
    target_data = torch.FloatTensor(target_data_x)
    target_labels = torch.FloatTensor(target_data_y)
    target_dataset = Data.TensorDataset(target_data, target_labels)
    train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=BATCH_SIZE, shuffle=True,worker_init_fn=worker_init_fn,generator=g)
    test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=BATCH_SIZE, shuffle=False,worker_init_fn=worker_init_fn,generator=g)
    model_extract = nn.Sequential(
            nn.Linear(len(feature), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.2, inplace=False)
        )
    predictor = nn.Sequential(
            nn.Linear(1024, CELLTYPE_NUM),
            nn.Softmax(dim=1))
    for layer in model_extract:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.constant_(layer.bias, 0)
    for layer in predictor:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.constant_(layer.bias, 0)
    predictor.cuda()
    model_extract.cuda()
    optimizer = torch.optim.Adam([{'params': model_extract.parameters()},{'params': predictor.parameters()}],lr=LEARNING_RATE)
    pred_loss_list=[]
    mmd_loss_list=[]
    ccc_list=[]
    rmse_list=[]
    corr_list=[]
    for epoch in range(EPOCHS):
        model_extract.train()
        predictor.train()
        train_target_iterator = iter(train_target_loader)
        pred_loss_epoch, mmd_loss_epoch = 0., 0.
        for batch_idx, (source_x, source_y) in enumerate(train_train_loader):
            try:
                target_x, _ = next(train_target_iterator)
            except StopIteration:
                train_target_iterator = iter(train_target_loader)
                target_x, _ = next(train_target_iterator)
            source_x = source_x.cuda()
            target_x = target_x.cuda()
            source_y=source_y.cuda()
            input=torch.cat((source_x,target_x),0)
            embedding=model_extract(input)
            result= predictor(embedding)
            pred_loss = mse_loss(result.narrow(0, 0, source_x.size(0)), source_y)
            pred_loss_epoch += pred_loss.data.item()
            transfer_loss=DAN(embedding.narrow(0, 0, source_x.size(0)), embedding.narrow(0, source_x.size(0), target_x.size(0)))
            mmd_loss_epoch +=transfer_loss.data           
            total_loss = transfer_loss + pred_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
        pred_loss_epoch = pred_loss_epoch
        pred_loss_list.append(pred_loss_epoch)
        mmd_loss_epoch = mmd_loss_epoch / (batch_idx + 1)
        mmd_loss_epoch=mmd_loss_epoch.cpu()
        mmd_loss_list.append(mmd_loss_epoch)
        if TARGET_TYPE == "simulated":
            preds,gt=None, None
            for batch_idx, (x, y) in enumerate(test_target_loader):
                model_extract.eval()
                predictor.eval()
                logits = predictor(model_extract(x.cuda())).detach().cpu().numpy()
                frac = y.detach().cpu().numpy()
                preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
                gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
            target_preds = pd.DataFrame(preds, columns=TYPE_LIST)
            ground_truth = pd.DataFrame(gt, columns=TYPE_LIST)
            epoch_ccc, epoch_rmse, epoch_corr = compute_metrics(target_preds, ground_truth)
            ccc_list.append(epoch_ccc)
            rmse_list.append(epoch_rmse)
            corr_list.append(epoch_corr)
    result_list=[]
    result_list.append(pred_loss_list)
    result_list.append(mmd_loss_list)
    if TARGET_TYPE == "simulated":
        result_list.append(ccc_list)
        result_list.append(rmse_list)
        result_list.append(corr_list)
        plot_loss_histories(result_list,['pred_loss', 'mmd_loss', 'target_ccc', 'target_rmse','target_corr'])
    elif TARGET_TYPE=="real":
        plot_loss_histories(result_list,['pred_loss', 'mmd_loss'])
    preds, gt = None, None
    for batch_idx, (x, y) in enumerate(test_target_loader):
        model_extract.eval()
        predictor.eval()
        logits = predictor(model_extract(x.cuda())).detach().cpu().numpy()
        frac = y.detach().cpu().numpy()
        preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
        gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
    target_preds = pd.DataFrame(preds, columns=TYPE_LIST)
    ground_truth = pd.DataFrame(gt, columns=TYPE_LIST)
    model_extract=None
    predictor=None
    return target_preds,ground_truth
for good in range (20):
    EXPERT_PROPORTION=np.zeros(CELLTYPE_NUM)
    seed=good+100
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnndeterministic =True
    torch.backends.cudnn.benchmark =False        
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    g = torch.Generator()
    g.manual_seed(seed)
    num_of_neighbors=5
    TARGET_DATASET_NAME = 'glioma_target_cellprotein.csv'
    ref_data_x= load_real_data(TRAIN_DATASET_NAME)
    _, ref_data_y=load_dataset(TRAIN_DATASET_NAME,TRAIN_ANNOTATION_NAME)
    ref_data_y=ref_data_y.to_numpy().flatten()
    target_data_x = load_real_data(TARGET_DATASET_NAME)
    used_features = list(set(ref_data_x.columns.tolist()).intersection(set(target_data_x.columns.tolist())))
    ref_data_x=ref_data_x[used_features]
    target_data_x = target_data_x[used_features]
    clf = KNeighborsClassifier(n_neighbors=num_of_neighbors,metric='euclidean')
    prediction=clf.fit(ref_data_x, ref_data_y).predict(target_data_x)
    for j in range(len(prediction)):
        for i in range(len(TYPE_LIST)):
            if prediction[j]==TYPE_LIST[i]:
                EXPERT_PROPORTION[i]=EXPERT_PROPORTION[i]+1
    EXPERT_PROPORTION=EXPERT_PROPORTION/np.sum(EXPERT_PROPORTION)
    train_data_x,train_data_y,target_data_x,target_data_y= mixup()
    final_preds_target, ground_truth_target=train(train_data_x,train_data_y,target_data_x,target_data_y)
    if TARGET_TYPE == "simulated":
        plot_prediction_scatter(final_preds_target, ground_truth_target)
        final_preds_target.to_csv(RESULT_DIRECTORY + "target_predicted_fractions" + str(good) + ".csv")
        ccc,rmse,correlation=compute_metrics(final_preds_target,ground_truth_target)
        CCC_LIST.append(ccc)
        RMSE_LIST.append(rmse)
        CORR_LIST.append(correlation)
    elif TARGET_TYPE == "real":
        final_preds_target.to_csv(RESULT_DIRECTORY + 'target_predicted_fraction' + str(good) + '.csv')
if TARGET_TYPE=="simulated":
    result_dataframe=pd.DataFrame({"CCC":CCC_LIST,"RMSE":RMSE_LIST,"CORR":CORR_LIST})
    result_dataframe.to_csv(RESULT_DIRECTORY + "target_result_all.csv")