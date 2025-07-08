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
DATASET = 'EPIC'
DATA_DIRECTORY = 'D:/Data/Methylation/450K_simulated/target_450K_simulated/'
RESULT_DIRECTORY= 'D:/Data/Methylation/450K_simulated/target_450K_simulated/DCQ/'
TYPE_LIST=['neu', 'NK', 'CD4','CD8','B','mono']
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
        concordance_corrcoef(y, x),
        sqrt(mean_squared_error(y, x)),
        pearsonr(y, x)[0]
    )
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
CCC_LIST=[]
RMSE_LIST=[]
CORR_LIST=[]
for good in range (20):
    prediction=pd.read_csv(RESULT_DIRECTORY+"dcq"+str(good)+".csv",index_col=0,header=0)
    ground_truth=pd.read_csv(DATA_DIRECTORY+"450K_simulated_test_y_"+str(good)+".csv",index_col=0,header=0)
    plot_prediction_scatter(prediction,ground_truth)
    ccc,rmse,correlation=compute_metrics(prediction,ground_truth)
    CCC_LIST.append(ccc)
    RMSE_LIST.append(rmse)
    CORR_LIST.append(correlation)
    result_dataframe=pd.DataFrame({"CCC":CCC_LIST,"RMSE":RMSE_LIST,"CORR":CORR_LIST})
    result_dataframe.to_csv(RESULT_DIRECTORY + "target_result_all.csv")