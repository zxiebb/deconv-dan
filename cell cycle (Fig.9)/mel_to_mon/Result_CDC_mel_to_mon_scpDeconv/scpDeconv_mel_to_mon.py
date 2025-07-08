import scipy
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import sys
import argparse
option_list = defaultdict(list)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def get_option_list(good):
    option_list['data_dir'] = 'target_CDC_mel_to_mon/'
    option_list['ref_dataset_name'] = "melanoma.csv"
    option_list['ref_metadata_name'] ="annotation_CDC.csv"
    option_list['target_dataset_name'] = "CDC_mel_to_mon_test_x_"+str(good)+".csv"
    option_list['target_metadata_name'] = None
    option_list['random_type'] = "celltype"
    option_list['type_list'] = ['G1', 'S', 'G2']
    option_list['ref_sample_num'] = 4000
    option_list['sample_size'] = 100
    option_list['HVP_num'] = 500
    option_list['target_type'] = "real"
    option_list['target_sample_num'] = 1000
    option_list['batch_size'] = 50
    option_list['epochs'] = 30
    option_list['learning_rate'] = 0.0001
    option_list['SaveResultsDir'] = "Result_CDC_mel_to_mon_scpDeconv/"
    return option_list

def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss
def Recon_loss(recon_data, input_data):
    loss_rec_fn = nn.MSELoss().cuda()
    loss = loss_rec_fn(recon_data, input_data)
    return loss
def ccc(preds, gt):
    numerator = 2 * np.corrcoef(gt, preds)[0][1] * np.std(gt) * np.std(preds)
    denominator = np.var(gt) + np.var(preds) + (np.mean(gt) - np.mean(preds)) ** 2
    ccc_value = numerator / denominator
    return ccc_value
def compute_metrics(preds, gt):
    gt = gt[preds.columns]  # Align pred order and gt order
    x = pd.melt(preds)['value']
    y = pd.melt(gt)['value']
    CCC = ccc(x, y)
    RMSE = sqrt(mean_squared_error(x, y))
    Corr = pearsonr(x, y)[0]
    return CCC, RMSE, Corr
def sample_normalize(data, normalize_method='min_max'):
    # Normalize data
    mm = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    if normalize_method == 'min_max':
        # it scales features so transpose is needed
        data = mm.fit_transform(data.T).T
    elif normalize_method == 'z_score':
        # Z score normalization
        data = (data - data.mean(0)) / (data.std(0) + (1e-10))
    return data
def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    for i in range(len(loss_type)):
        plt.subplot(2, 3, i + 1)
        plt.plot(metric_logger[loss_type[i]])
        plt.title(loss_type[i], x=0.5, y=0.5)
    imgName = os.path.join(SavePath, output_prex +str(good)+ '.png')
    plt.savefig(imgName)
    plt.close()
def SavePredPlot(SavePath, target_preds, ground_truth):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    celltypes = list(target_preds.columns)
    plt.figure(figsize=(5 * (len(celltypes) + 1), 5))
    eval_metric = []
    x = pd.melt(target_preds)['value']
    y = pd.melt(ground_truth)['value']
    eval_metric.append(ccc(x, y))
    eval_metric.append(sqrt(mean_squared_error(x, y)))
    eval_metric.append(pearsonr(x, y)[0])
    plt.subplot(1, len(celltypes) + 1, 1)
    plt.xlim(0, max(y))
    plt.ylim(0, max(y))
    plt.scatter(x, y, s=2)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")
    text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
    plt.text(0.05, max(y) - 0.05, text, fontsize=8, verticalalignment='top')
    plt.title('All samples')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    for i in range(len(celltypes)):
        eval_metric = []
        x = target_preds[celltypes[i]]
        y = ground_truth[celltypes[i]]
        eval_metric.append(ccc(x, y))
        eval_metric.append(sqrt(mean_squared_error(x, y)))
        eval_metric.append(pearsonr(x, y)[0])
        plt.subplot(1, len(celltypes) + 1, i + 2)
        plt.xlim(0, max(y))
        plt.ylim(0, max(y))
        plt.scatter(x, y, s=2)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--")
        text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
        plt.text(0.05, max(y) - 0.05, text, fontsize=8, verticalalignment='top')
        plt.title(celltypes[i])
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
    imgName = SavePath+ 'pred_fraction_target_scatter'+str(good)+'.jpg'
    plt.savefig(imgName)
    plt.close()
class ReferMixup(object):
    def __init__(self, option_list):  
        self.data_path = option_list['data_dir']
        self.ref_dataset_name = option_list['ref_dataset_name']
        self.ref_metadata_name = option_list['ref_metadata_name']
        self.target_dataset_name = option_list['target_dataset_name']
        self.target_metadata_name = option_list['target_metadata_name']
        self.random_type = option_list['random_type']
        self.type_list = option_list['type_list']
        self.train_sample_num = option_list['ref_sample_num']
        self.sample_size = option_list['sample_size']
        self.HVP_num = option_list['HVP_num']
        self.target_type = option_list['target_type']
        self.target_sample_num = option_list['target_sample_num']
        self.outdir = option_list['SaveResultsDir']
        self.normalize = 'min_max'
    def mixup(self):
        # mixup reference datasets and simulate pseudo-bulk train data
        train_data_x, train_data_y = self.mixup_dataset(self.ref_dataset_name, self.ref_metadata_name, self.train_sample_num)
        # mixup to simulate pseudo target data or get real target data
        if self.target_type == "simulated":
            target_data_x, target_data_y = self.mixup_dataset(self.target_dataset_name, self.target_metadata_name, self.target_sample_num)
            target_data = ad.AnnData(X=target_data_x.to_numpy(), obs=target_data_y)
            target_data.var_names = target_data_x.columns
        elif self.target_type == "real":
            target_data = self.load_real_data(self.target_dataset_name)
        # find protein list as used features by integrating train and target 
        used_features = self.align_features(train_data_x, target_data)
        # prepare train data and target data with aligned features
        train_data = self.align_dataset(train_data_x, train_data_y, used_features)
        target_data = target_data[:,used_features]
        # SavetSNEPlot(self.outdir, train_data, output_prex='Pseudo_Bulk_Source_'+str(self.train_sample_num))
        # SavetSNEPlot(self.outdir, target_data, output_prex='Pseudo_Bulk_Target_'+str(self.target_sample_num))
        return train_data, target_data
    def align_features(self, train_data_x, target_data):
        used_features = set(train_data_x.columns.tolist()).intersection(set(target_data.var_names.tolist())) # overlapped features between reference and target
        if self.HVP_num == 0:
            used_features = list(used_features)
        elif self.HVP_num > 0:
            sc.pp.highly_variable_genes(target_data, n_top_genes=self.HVP_num)
            HVPs = set(target_data.var[target_data.var.highly_variable].index)
            used_features = list(used_features.union(HVPs))
        return used_features
    def align_dataset(self, sim_data_x, sim_data_y, used_features):
        missing_features = [feature for feature in used_features if feature not in list(sim_data_x.columns)]
        if len(missing_features) > 0:
            missing_data_x = pd.DataFrame(np.zeros((sim_data_x.shape[0],len(missing_features))), columns=missing_features, index=sim_data_x.index)
            sim_data_x = pd.concat([sim_data_x, missing_data_x], axis=1)
        sim_data_x = sim_data_x[used_features]
        sim_data = ad.AnnData(
            X=sim_data_x.to_numpy(),
            obs=sim_data_y
        )
        sim_data.uns["cell_types"] = self.type_list
        sim_data.var_names = used_features
        return sim_data
    def mixup_dataset(self, dataset, metadata, sample_num):
        sim_data_x = []
        sim_data_y = []
        ref_data_x, ref_data_y = self.load_ref_dataset(dataset, metadata)
        for i in range(int(sample_num)):
            sample, label = self.mixup_cells(ref_data_x, ref_data_y, self.type_list)
            sim_data_x.append(sample)
            sim_data_y.append(label)
        sim_data_x = pd.concat(sim_data_x, axis=1).T
        sim_data_y = pd.DataFrame(sim_data_y, columns=self.type_list)
        # Scale pseudo-bulk data
        if self.normalize:
            sim_data_x_scale = sample_normalize(sim_data_x, normalize_method=self.normalize)
            sim_data_x_scale = pd.DataFrame(sim_data_x_scale, columns=sim_data_x.columns)
            sim_data_x = sim_data_x_scale
        return sim_data_x, sim_data_y
    def load_ref_dataset(self, dataset, metadata):
        if ".h5ad" in dataset:
            filename = os.path.join(self.data_path, dataset)
            try:
                data_h5ad = ad.read_h5ad(filename)
                # Extract celltypes
                if self.type_list == None:
                    self.type_list = list(set(data_h5ad.obs[self.random_type].tolist()))
                data_h5ad = data_h5ad[data_h5ad.obs[self.random_type].isin(self.type_list)]
            except FileNotFoundError as e:
                print(f"No such h5ad file found for [cyan]{dataset}")
                sys.exit(e)
            try:
                data_y = pd.DataFrame(data_h5ad.obs[self.random_type])
                data_y.reset_index(inplace=True, drop=True)
            except Exception as e:
                print(f"Celltype attribute not found for [cyan]{dataset}")
                sys.exit(e)
            if scipy.sparse.issparse(data_h5ad.X):
                data_x = pd.DataFrame(data_h5ad.X.todense())
            else:
                data_x = pd.DataFrame(data_h5ad.X)

            data_x = data_x.fillna(0) # fill na with 0    
            data_x.index = data_h5ad.obs_names
            data_x.columns = data_h5ad.var_names

            return data_x, data_y

        elif ".csv" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_x = pd.read_csv(filename, header=0, index_col=0)
            except FileNotFoundError as e:
                print(f"No such expression csv file found for [cyan]{dataset}")
                sys.exit(e)
        
            data_x = data_x.fillna(0) # fill na with 0    
            
            if metadata is not None:
                metadata_filename = os.path.join(self.data_path, metadata)
                try:
                    data_y = pd.read_csv(metadata_filename, header=0, index_col=0)
                except Exception as e:
                    print(f"Celltype attribute not found for [cyan]{dataset}")
                    sys.exit(e)
            else:
                print(f"Metadata file is not provided for [cyan]{dataset}")
                sys.exit(1)

            return data_x, data_y

    def load_real_data(self, dataset):
        
        if ".h5ad" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_h5ad = ad.read_h5ad(filename)
            except FileNotFoundError as e:
                print(f"No such h5ad file found for [cyan]{dataset}.")
                sys.exit(e)
            
            if scipy.sparse.issparse(data_h5ad.X):
                data_h5ad.X = pd.DataFrame(data_h5ad.X.todense()).fillna(0)
            else:
                data_h5ad.X = pd.DataFrame(data_h5ad.X).fillna(0)
        
            if self.normalize:
                data_h5ad.X = sample_normalize(data_h5ad.X, normalize_method=self.normalize)

            return data_h5ad

        elif ".csv" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_x = pd.read_csv(filename, header=0, index_col=0)
            except FileNotFoundError as e:
                print(f"No such target expression csv file found for [cyan]{dataset}.")
                sys.exit(e)
            
            data_x = data_x.fillna(0) # fill na with 0    
            
            data_h5ad = ad.AnnData(X=data_x)
            data_h5ad.var_names = data_x.columns

            if self.normalize:
                data_h5ad.X = sample_normalize(data_h5ad.X, normalize_method=self.normalize)

            return data_h5ad

    def mixup_fraction(self, celltype_num):

        fracs = np.random.rand(celltype_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)

        return fracs

    def mixup_cells(self, x, y, celltypes):

        available_celltypes = celltypes
        
        celltype_num = len(available_celltypes)

        # Create fractions for available celltypes
        fracs = self.mixup_fraction(celltype_num)

        samp_fracs = np.multiply(fracs, self.sample_size)
        samp_fracs = list(map(round, samp_fracs))
        
        # Make complete fracions
        fracs_complete = [0] * len(celltypes)

        for i, act in enumerate(available_celltypes):
            idx = celltypes.index(act)
            fracs_complete[idx] = fracs[i]

        artificial_samples = []

        for i in range(celltype_num):
            ct = available_celltypes[i]
            cells_sub = x.loc[np.array(y[self.random_type] == ct), :]
            cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
            cells_sub = cells_sub.iloc[cells_fraction, :]
            artificial_samples.append(cells_sub)

        df_samp = pd.concat(artificial_samples, axis=0)
        df_samp = df_samp.sum(axis=0)

        return df_samp, fracs_complete


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out = self.layer(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out = self.layer(x)
        return out


class AEimpute(object):
    def __init__(self, option_list):
        self.num_epochs = 200
        self.batch_size = option_list['batch_size']
        self.learning_rate = option_list['learning_rate']
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021
        self.outdir = option_list['SaveResultsDir']

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def AEimpute_model(self, celltype_num):
        feature_num = len(self.used_features)

        self.encoder_im = nn.Sequential(EncoderBlock(feature_num, 512),
                                        EncoderBlock(512, 256))

        self.predictor_im = nn.Sequential(nn.Linear(256, celltype_num),
                                          nn.Softmax(dim=-1))

        self.decoder_im = nn.Sequential(DecoderBlock(256, 512),
                                        DecoderBlock(512, feature_num))

        model_im = nn.ModuleList([])
        model_im.append(self.encoder_im)
        model_im.append(self.predictor_im)
        model_im.append(self.decoder_im)
        return model_im

    def prepare_dataloader(self, ref_data, target_data, batch_size):
        ### Prepare data loader for training ###
        # ref dataset
        ref_ratios = [ref_data.obs[ctype] for ctype in ref_data.uns['cell_types']]
        self.ref_data_x = ref_data.X.astype(np.float32)
        self.ref_data_y = np.array(ref_ratios, dtype=np.float32).transpose()

        tr_data = torch.FloatTensor(self.ref_data_x)
        tr_labels = torch.FloatTensor(self.ref_data_y)
        ref_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_ref_loader = Data.DataLoader(dataset=ref_dataset, batch_size=batch_size, shuffle=True)
        self.test_ref_loader = Data.DataLoader(dataset=ref_dataset, batch_size=batch_size, shuffle=False)

        # Extract celltype and feature info
        self.labels = ref_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(ref_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def train(self, ref_data, target_data):
        ### prepare model structure ###
        self.prepare_dataloader(ref_data, target_data, self.batch_size)
        self.model_im = self.AEimpute_model(self.celltype_num).cuda()

        ### setup optimizer ###
        optimizer_im = torch.optim.Adam([{'params': self.encoder_im.parameters()},
                                         {'params': self.predictor_im.parameters()},
                                         {'params': self.decoder_im.parameters()}], lr=self.learning_rate)

        metric_logger = defaultdict(list)

        for epoch in range(self.num_epochs):
            self.model_im.train()

            train_target_iterator = iter(self.train_target_loader)
            loss_epoch, pred_loss_epoch, recon_loss_epoch = 0., 0., 0.
            for batch_idx, (ref_x, ref_y) in enumerate(self.train_ref_loader):
                # get batch item of target
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                X = torch.cat((ref_x, target_x))

                embedding = self.encoder_im(X.cuda())
                frac_pred = self.predictor_im(embedding)
                recon_X = self.decoder_im(embedding)

                # caculate loss
                pred_loss = L1_loss(frac_pred[range(self.batch_size),], ref_y.cuda())
                pred_loss_epoch += pred_loss
                rec_loss = Recon_loss(recon_X, X.cuda())
                recon_loss_epoch += rec_loss
                loss = rec_loss + pred_loss
                loss_epoch += loss

                # update weights
                optimizer_im.zero_grad()
                loss.backward()
                optimizer_im.step()

            loss_epoch = loss_epoch / (batch_idx + 1)
            loss_epoch=loss_epoch.detach().cpu().numpy()
            metric_logger['cAE_loss'].append(loss_epoch)
            pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
            pred_loss_epoch=pred_loss_epoch.detach().cpu().numpy()
            metric_logger['pred_loss'].append(pred_loss_epoch)
            recon_loss_epoch = recon_loss_epoch / (batch_idx + 1)
            recon_loss_epoch=recon_loss_epoch.detach().cpu().numpy()
            metric_logger['recon_loss'].append(recon_loss_epoch)
            if (epoch + 1) % 10 == 0:
                print('============= Epoch {:02d}/{:02d} in stage2 ============='.format(epoch + 1, self.num_epochs))
                print("cAE_loss=%f, pred_loss=%f, recon_loss=%f" % (loss_epoch, pred_loss_epoch, recon_loss_epoch))

        ### Plot loss ###
        SaveLossPlot(self.outdir, metric_logger, loss_type=['cAE_loss', 'pred_loss', 'recon_loss'],
                     output_prex='Loss_plot_stage2')

        ### Save reconstruction data of ref and target ###
        ref_recon_data = self.write_recon(ref_data)

        return ref_recon_data

    def write_recon(self, ref_data):

        self.model_im.eval()

        ref_recon, ref_label = None, None
        for batch_idx, (x, y) in enumerate(self.test_ref_loader):
            x_embedding = self.encoder_im(x.cuda())
            x_prediction = self.predictor_im(x_embedding)
            x_recon = self.decoder_im(x_embedding).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            ref_recon = x_recon if ref_recon is None else np.concatenate((ref_recon, x_recon), axis=0)
            ref_label = labels if ref_label is None else np.concatenate((ref_label, labels), axis=0)
        ref_recon = pd.DataFrame(ref_recon, columns=self.used_features)
        ref_label = pd.DataFrame(ref_label, columns=self.labels)

        ref_recon_data = ad.AnnData(X=ref_recon.to_numpy(), obs=ref_label)
        ref_recon_data.uns['cell_types'] = self.labels
        ref_recon_data.var_names = self.used_features

        ### Plot recon ref TSNE plot ###
        # SavetSNEPlot(self.outdir, ref_recon_data, output_prex='AE_Recon_ref')
        ### Plot recon ref TSNE plot using missing features ###
        # sc.pp.filter_genes(ref_data, min_cells=0)
        # missing_features = list(ref_data.var[ref_data.var['n_cells']==0].index)
        # if len(missing_features) > 0:
        #     Recon_ref_data_new = ref_recon_data[:,missing_features]
        #     SavetSNEPlot(self.outdir, Recon_ref_data_new, output_prex='AE_Recon_ref_missingfeature')

        return ref_recon_data


class EncoderBlockN(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlockN, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out


class DecoderBlockN(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlockN, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out


class DANN(object):
    def __init__(self, option_list):
        self.num_epochs = option_list['epochs']
        self.batch_size = option_list['batch_size']
        self.target_type = option_list['target_type']
        self.learning_rate = option_list['learning_rate']
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021
        self.outdir = option_list['SaveResultsDir']

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def DANN_model(self, celltype_num):
        feature_num = len(self.used_features)

        self.encoder_da = nn.Sequential(EncoderBlockN(feature_num, 512, 0),
                                        EncoderBlockN(512, 256, 0.3))

        self.predictor_da = nn.Sequential(EncoderBlockN(256, 128, 0.2),
                                          nn.Linear(128, celltype_num),
                                          nn.Softmax(dim=1))

        self.discriminator_da = nn.Sequential(EncoderBlockN(256, 128, 0.2),
                                              nn.Linear(128, 1),
                                              nn.Sigmoid())

        model_da = nn.ModuleList([])
        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)
        model_da.append(self.discriminator_da)
        return model_da

    def prepare_dataloader(self, source_data, target_data, batch_size):
        ### Prepare data loader for training ###
        # Source dataset
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()

        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn,generator=g)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        if self.target_type == "simulated":
            target_ratios = [target_data.obs[ctype] for ctype in self.labels]
            self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        elif self.target_type == "real":
            self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn,generator=g)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False,worker_init_fn=worker_init_fn,generator=g)

    def train(self, source_data, target_data):

        ### prepare model structure ###
        self.prepare_dataloader(source_data, target_data, self.batch_size)
        self.model_da = self.DANN_model(self.celltype_num).cuda()

        ### setup optimizer ###
        optimizer_da1 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.predictor_da.parameters()},
                                          {'params': self.discriminator_da.parameters()}], lr=self.learning_rate)
        optimizer_da2 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.discriminator_da.parameters()}], lr=self.learning_rate)

        criterion_da = nn.BCELoss().cuda()
        source_label = torch.ones(self.batch_size).unsqueeze(1).cuda()  # 定义source domain label为1
        target_label = torch.zeros(self.batch_size).unsqueeze(1).cuda()  # 定义target domain label为0

        metric_logger = defaultdict(list)

        for epoch in range(self.num_epochs):
            self.model_da.train()

            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch = 0., 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                # get batch item of target
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                embedding_source = self.encoder_da(source_x.cuda())
                embedding_target = self.encoder_da(target_x.cuda())
                frac_pred = self.predictor_da(embedding_source)
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)

                # caculate loss
                pred_loss = L1_loss(frac_pred, source_y.cuda())
                pred_loss_epoch += pred_loss.data.item()
                disc_loss = criterion_da(domain_pred_source,
                                         source_label[0:domain_pred_source.shape[0], ]) + criterion_da(
                    domain_pred_target, target_label[0:domain_pred_target.shape[0], ])
                disc_loss_epoch += disc_loss.data.item()
                loss = pred_loss + disc_loss

                # update weights
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_da1.step()

                embedding_source = self.encoder_da(source_x.cuda())
                embedding_target = self.encoder_da(target_x.cuda())
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)

                # caculate loss
                disc_loss_DA = criterion_da(domain_pred_target,
                                            source_label[0:domain_pred_target.shape[0], ]) + criterion_da(
                    domain_pred_source, target_label[0:domain_pred_source.shape[0], ])
                disc_loss_DA_epoch += disc_loss_DA.data.item()

                # update weights
                optimizer_da2.zero_grad()
                disc_loss_DA.backward(retain_graph=True)
                optimizer_da2.step()

            pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
            # pred_loss_epoch=pred_loss_epoch.cpu()
            metric_logger['pred_loss'].append(pred_loss_epoch)
            disc_loss_epoch = disc_loss_epoch / (batch_idx + 1)
            # disc_loss_epoch=disc_loss_epoch.detach().cpu().numpy()
            metric_logger['disc_loss'].append(disc_loss_epoch)
            disc_loss_DA_epoch = disc_loss_DA_epoch / (batch_idx + 1)
            # disc_loss_DA_epoch=disc_loss_DA_epoch.detach().cpu().numpy()
            metric_logger['disc_loss_DA'].append(disc_loss_DA_epoch)

            if (epoch + 1) % 1 == 0:
                print('============= Epoch {:02d}/{:02d} in stage3 ============='.format(epoch + 1, self.num_epochs))
                print("pred_loss=%f, disc_loss=%f, disc_loss_DA=%f" % (
                pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch))
                if self.target_type == "simulated":
                    ### model validation on target data ###
                    target_preds, ground_truth = self.prediction()
                    epoch_ccc, epoch_rmse, epoch_corr = compute_metrics(target_preds, ground_truth)
                    metric_logger['target_ccc'].append(epoch_ccc)
                    metric_logger['target_rmse'].append(epoch_rmse)
                    metric_logger['target_corr'].append(epoch_corr)

        if self.target_type == "simulated":
            SaveLossPlot(self.outdir, metric_logger,
                         loss_type=['pred_loss', 'disc_loss', 'disc_loss_DA', 'target_ccc', 'target_rmse',
                                    'target_corr'], output_prex='Loss_metric_plot_stage3')
        elif self.target_type == "real":
            SaveLossPlot(self.outdir, metric_logger, loss_type=['pred_loss', 'disc_loss', 'disc_loss_DA'],
                         output_prex='Loss_metric_plot_stage3')
    def prediction(self):
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            logits = self.predictor_da(self.encoder_da(x.cuda())).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='murine_cellline', help='The name of benchmarking datasets')
args = parser.parse_args()

def main():
    dataset = args.dataset
	### Start Running scpDeconv ###
    print("------Start Running scpDeconv------")
    opt = get_option_list(good)

	### Run Stage 1 ###
    print("------Start Running Stage 1 : Mixup reference------")
    model_mx = ReferMixup(opt)
    source_data, target_data = model_mx.mixup()
    print("The dim of source data is :")
    print(source_data.shape)
    print("The dim of target data is :")
    print(target_data.shape)
    print("Stage 1 : Mixup finished!")

	### Run Stage 2 ###
    print("------Start Running Stage 2 : Training AEimpute model------")
    model_im = AEimpute(opt)
    source_recon_data = model_im.train(source_data, target_data)
    print("Stage 2 : AEimpute model training finished!")

	### Run Stage 3 ###
    print("------Start Running Stage 3 : Training DANN model------")
    model_da = DANN(opt)
    model_da.train(source_recon_data, target_data)
    print("Stage 3 : DANN model training finished!")

	### Run Stage 4 ###
    print("------Start Running Stage 4 : Inference for target data------")
    if opt['target_type'] == "simulated":
        final_preds_target, ground_truth_target = model_da.prediction()
        SavePredPlot(opt['SaveResultsDir'], final_preds_target, ground_truth_target)
        final_preds_target.to_csv(os.path.join(opt['SaveResultsDir'], "target_predicted_fractions.csv"))

    elif opt['target_type'] == "real":
        final_preds_target, _ = model_da.prediction()
        final_preds_target.to_csv(opt['SaveResultsDir']+"target_predicted_fraction"+str(good)+".csv")
        print("Stage 4 : Inference for target data finished!")
for good in range(20):

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
    main()
