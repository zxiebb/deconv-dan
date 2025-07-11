# target sample contain at least 50 percent of one certain cell type. 
import numpy as np
def mixup_fraction_test(celltype_num):
    while True:
        fracs = np.random.rand(celltype_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)
        if fracs[0]>0.5:
            break
    return fracs
# whether the number of cells mixed-up to generate pseudo bulk samples would matter. 
sample_size=200
sample_size=20
# whether number of number of target samples would influence deconvolution performance
target_sample_num = 20
# with higher purity (one cell type make up more than 70 percent)
import numpy as np
def mixup_fraction_test(celltype_num):
    while True:
        fracs = np.random.rand(celltype_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)
        if np.any(fracs>0.7):
            break
    return fracs
# scpDeconv's mix-up
import pandas as pd
def mixup_fraction_train(celltype_num):
    fracs = np.random.rand(celltype_num)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
# Scaden's mix-up
def create_fractions(no_celltypes):
    fracs = np.random.rand(no_celltypes)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs
def mixup_dataset(x, y, celltypes):
    sim_x = []
    sim_y = []
    for i in range(target_sample_num//2):
        sample, label = create_subsample(x, y, celltypes)
        sim_x.append(sample)
        sim_y.append(label)
    for i in range(target_sample_num//2):
        sample, label = create_subsample(x, y, celltypes, sparse=True)
        sim_x.append(sample)
        sim_y.append(label)
    sim_x = pd.concat(sim_x, axis=1).T
    sim_y = pd.DataFrame(sim_y, columns=celltypes)
    return sim_x, sim_y
def create_subsample(x, y, celltypes, sparse=False):
    available_celltypes = celltypes
    if sparse:
        no_keep = np.random.randint(1, len(available_celltypes))
        keep = np.random.choice(list(range(len(available_celltypes))), size=no_keep, replace=False)
        available_celltypes = [available_celltypes[i] for i in keep]
    no_avail_cts = len(available_celltypes)
    fracs = create_fractions(no_celltypes=no_avail_cts)
    samp_fracs = np.multiply(fracs, sample_size)
    samp_fracs = list(map(int, samp_fracs))
    fracs_complete = [0] * len(celltypes)
    for i, act in enumerate(available_celltypes):
        idx = celltypes.index(act)
        fracs_complete[idx] = fracs[i]
    artificial_samples = []
    for i in range(no_avail_cts):
        ct = available_celltypes[i]
        cells_sub = x.loc[np.array(y["celltype"] == ct), :]
        cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
        cells_sub = cells_sub.iloc[cells_fraction, :]
        artificial_samples.append(cells_sub)
    df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = df_samp.sum(axis=0)
    return df_samp, fracs_complete
# Scaden's network
pred_loss.backward()
optimizer.step()
# scpDeconv's network
class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out
class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
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
        torch.manual_seed(self.seed)
        random.seed(self.seed)
    def DANN_model(self, celltype_num):
        feature_num = len(self.used_features)
        self.encoder_da = nn.Sequential(EncoderBlock(feature_num, 512, 0),
                                        EncoderBlock(512, 256, 0.3))
        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                          nn.Linear(128, celltype_num),
                                          nn.Softmax(dim=1))
        self.discriminator_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                              nn.Linear(128, 1),
                                              nn.Sigmoid())
        model_da = nn.ModuleList([])
        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)
        model_da.append(self.discriminator_da)
        return model_da
    def prepare_dataloader(self, source_data, target_data, batch_size):
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)
        self.target_data_x = target_data.X.astype(np.float32)
        if self.target_type == "simulated":
            target_ratios = [target_data.obs[ctype] for ctype in self.labels]
            self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        elif self.target_type == "real":
            self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)
            # random rand will not be used, just for making dataloaders
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)
    def train(self, source_data, target_data):
        self.prepare_dataloader(source_data, target_data, self.batch_size)
        self.model_da = self.DANN_model(self.celltype_num)
        optimizer_da1 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.predictor_da.parameters()},
                                          {'params': self.discriminator_da.parameters()}], lr=self.learning_rate)
        optimizer_da2 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.discriminator_da.parameters()}], lr=self.learning_rate)
        criterion_da = nn.BCELoss()
        source_label = torch.ones(self.batch_size).unsqueeze(1)
        target_label = torch.zeros(self.batch_size).unsqueeze(1)
        metric_logger = defaultdict(list)
        for epoch in range(self.num_epochs):
            self.model_da.train()
            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch = 0., 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)
                embedding_source = self.encoder_da(source_x)
                embedding_target = self.encoder_da(target_x)
                frac_pred = self.predictor_da(embedding_source)
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)
                pred_loss = L1_loss(frac_pred, source_y)
                pred_loss_epoch += pred_loss.data.item()
                disc_loss = criterion_da(domain_pred_source,
                                         source_label[0:domain_pred_source.shape[0], ]) + criterion_da(
                    domain_pred_target, target_label[0:domain_pred_target.shape[0], ])
                disc_loss_epoch += disc_loss.data.item()
                loss = pred_loss + disc_loss
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_da1.step()
                embedding_source = self.encoder_da(source_x)
                embedding_target = self.encoder_da(target_x)
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)
                disc_loss_DA = criterion_da(domain_pred_target,
                                            source_label[0:domain_pred_target.shape[0], ]) + criterion_da(
                    domain_pred_source, target_label[0:domain_pred_source.shape[0], ])
                disc_loss_DA_epoch += disc_loss_DA.data.item()
                optimizer_da2.zero_grad()
                disc_loss_DA.backward(retain_graph=True)
                optimizer_da2.step()
            pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
            metric_logger['pred_loss'].append(pred_loss_epoch)
            disc_loss_epoch = disc_loss_epoch / (batch_idx + 1)
            metric_logger['disc_loss'].append(disc_loss_epoch)
            disc_loss_DA_epoch = disc_loss_DA_epoch / (batch_idx + 1)
            metric_logger['disc_loss_DA'].append(disc_loss_DA_epoch)
            if (epoch + 1) % 1 == 0:
                print('============= Epoch {:02d}/{:02d} in stage3 ============='.format(epoch + 1, self.num_epochs))
                print("pred_loss=%f, disc_loss=%f, disc_loss_DA=%f" % (
                pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch))
                if self.target_type == "simulated":
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
            logits = self.predictor_da(self.encoder_da(x)).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)
        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth