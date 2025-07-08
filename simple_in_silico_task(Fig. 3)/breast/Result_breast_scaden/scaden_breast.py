import os
import pandas as pd
import anndata as ad
import numpy as np
from sklearn import preprocessing as pp
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
architectures = {'m256':    ([256, 128, 64, 32],    [0, 0, 0, 0]),
                 'm512':    ([512, 256, 128, 64],   [0, 0.3, 0.2, 0.1]),
                 'm1024':   ([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1])}
M256_HIDDEN_UNITS = architectures["m256"][0]
M512_HIDDEN_UNITS = architectures["m512"][0]
M1024_HIDDEN_UNITS = architectures["m1024"][0]
M256_DO_RATES = architectures["m256"][1]
M512_DO_RATES = architectures["m512"][1]
M1024_DO_RATES = architectures["m1024"][1]
def sample_scaling(x, scaling_option):
    if scaling_option == "log_min_max":
        # x = np.log2(x + 1)
        mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
        x = mms.fit_transform(x.T).T
    return x
def preprocess_h5ad_data(raw_input, scaling_option="log_min_max", sig_genes=None):
    raw_input = raw_input[:, sig_genes]
    raw_input.X = sample_scaling(raw_input.X, scaling_option)
    return raw_input
def get_signature_genes(input_path, sig_genes_complete, var_cutoff=0.1):
    data = pd.read_csv(input_path, header=0,index_col=0)
    data=data.T
    keep = data.var(axis=1) > var_cutoff
    data = data.loc[keep]
    available_genes = list(data.index)
    new_sig_genes = list(set(available_genes).intersection(sig_genes_complete))
    print(len(new_sig_genes))
    return new_sig_genes
def processing(data_path, raw_input, var_cutoff):
    sig_genes_complete = list(raw_input.var_names)
    sig_genes = get_signature_genes(input_path=data_path, sig_genes_complete=sig_genes_complete, var_cutoff=var_cutoff)
    result=preprocess_h5ad_data(raw_input=raw_input,sig_genes=sig_genes)
    return result
def create_fractions(no_celltypes):
    fracs = np.random.rand(no_celltypes)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs
class BulkSimulator(object):
    def __init__(
        self,
        sample_size=100,
        num_samples=1000
    ):
        self.sample_size = sample_size
        self.num_samples = num_samples // 2
    def simulate(self):
        dataset_counts = "reference_breast_cellprotein.csv"
        dataset_celltypes = "reference_annotation_breast.csv"
        data_x=pd.read_csv(dataset_counts, header=0, index_col=0)
        data_y=pd.read_csv(dataset_celltypes,header=0, index_col=0)
        celltypes = list(set(data_y["celltype"].tolist()))
        tmp_x, tmp_y = self.create_subsample_dataset(
            data_x, data_y, celltypes=celltypes
        )
        tmp_x = tmp_x.sort_index(axis=1)
        ratios = pd.DataFrame(tmp_y, columns=celltypes)
        print(ratios)
        ratios["ds"] = pd.Series(np.repeat("A", tmp_y.shape[0]), index=ratios.index)
        ann_data = ad.AnnData(
            X=tmp_x.to_numpy(),
            obs=ratios,
            var=pd.DataFrame(columns=[], index=list(tmp_x)),
        )
        ann_data.uns["cell_types"] = celltypes
        return ann_data
    def create_subsample_dataset(self, x, y, celltypes):
        sim_x = []
        sim_y = []
        for i in range(self.num_samples):
            sample, label = self.create_subsample(x, y, celltypes)
            sim_x.append(sample)
            sim_y.append(label)
        for i in range(self.num_samples):
            sample, label = self.create_subsample(x, y, celltypes, sparse=True)
            sim_x.append(sample)
            sim_y.append(label)
        sim_x = pd.concat(sim_x, axis=1).T
        sim_y = pd.DataFrame(sim_y, columns=celltypes)
        return sim_x, sim_y
    def create_subsample(self, x, y, celltypes, sparse=False):
        available_celltypes = celltypes
        if sparse:
            no_keep = np.random.randint(1, len(available_celltypes))
            keep = np.random.choice(
                list(range(len(available_celltypes))), size=no_keep, replace=False
            )
            available_celltypes = [available_celltypes[i] for i in keep]
        no_avail_cts = len(available_celltypes)
        fracs = create_fractions(no_celltypes=no_avail_cts)
        samp_fracs = np.multiply(fracs, self.sample_size)
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
class Scaden(object):
    def __init__(
        self,
        batch_size=128,
        learning_rate=0.0001,
        num_steps=1000,
        seed=0,
        hidden_units=[256, 128, 64, 32],
        do_rates=[0, 0, 0, 0],
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data = None
        self.n_classes = None
        self.labels = None
        self.x = None
        self.y = None
        self.num_steps = num_steps
        self.scaling = "log_min_max"
        self.sig_genes = None
        self.hidden_units = hidden_units
        self.do_rates = do_rates
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        np.random.seed(seed)
    def scaden_model(self, n_classes):
        model = nn.Sequential(
            nn.Linear(self.x_data.shape[1], self.hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(p=self.do_rates[0], inplace=False),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(p=self.do_rates[1], inplace=False),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.ReLU(),
            nn.Dropout(p=self.do_rates[2], inplace=False),
            nn.Linear(self.hidden_units[2], self.hidden_units[3]),
            nn.ReLU(),
            nn.Dropout(p=self.do_rates[3], inplace=False),
            nn.Linear(self.hidden_units[3], n_classes),
            nn.Softmax(dim=1)
        )
        return model
    def compute_loss(self, logits, targets):
        loss = torch.mean(torch.reshape(torch.square(logits - targets), (-1,)))
        return loss
    def load_h5ad_file(self, raw_input, batch_size):
        ratios = [raw_input.obs[ctype] for ctype in raw_input.uns["cell_types"]]
        self.x_data = raw_input.X.astype(np.float32)
        self.y_data = np.array(ratios, dtype=np.float32).transpose()
        self.x_data=torch.tensor(self.x_data)
        self.y_data=torch.tensor(self.y_data)
        self.data = Data.TensorDataset(self.x_data, self.y_data)
        self.data=Data.DataLoader(dataset=self.data, batch_size=batch_size, shuffle=True,generator=g)
        self.data_iter = iter(self.data)
        self.labels = raw_input.uns["cell_types"]
        self.sig_genes = list(raw_input.var_names)
    def build_model(self, input_path):
        self.load_h5ad_file(input_path,batch_size=self.batch_size)
        self.n_classes = len(self.labels)
        self.model = self.scaden_model(n_classes=self.n_classes)
        self.model.cuda()
    def train(self, input_path):
        self.build_model(input_path=input_path)
        optimizer = torch.optim.Adam(self.model.parameters(),self.learning_rate)
        self.model.train()
        for step in range(self.num_steps):
            try:
                x, y = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.data)
                x, y = next(self.data_iter)
            x=x.cuda()
            y=y.cuda()
            self.logits = self.model(x)
            loss = self.compute_loss(self.logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    def predict(self):
        self.model.train(False)
        data = pd.read_csv(PREDICTION_DATA, header=0, index_col=0)
        data=data.T
        self.sample_names = list(data.columns)
        data_index = list(data.index)
        if not (len(data_index) == len(set(data_index))):
            data = data.loc[~data.index.duplicated(keep="first")]
        data = data.loc[self.sig_genes]
        data = data.T
        if self.scaling:
            data = sample_scaling(data, scaling_option=self.scaling)
        self.data = torch.tensor(data,dtype=torch.float)
        self.data=self.data.cuda()
        print(self.data.dtype)
        predictions = self.model(self.data)
        pred_df = pd.DataFrame(
            predictions.detach().cpu().numpy(), columns=self.labels, index=self.sample_names
        )
        return pred_df
def simulation(sample_size, num_samples):
    bulk_simulator = BulkSimulator(sample_size=sample_size,
                                       num_samples=num_samples)
    result = bulk_simulator.simulate()
    return result
def training(
    data, batch_size, learning_rate, num_steps, seed=0
):
    cdn256 = Scaden(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
        hidden_units=M256_HIDDEN_UNITS,
        do_rates=M256_DO_RATES,
    )
    cdn256.train(input_path=data)
    preds_256 = cdn256.predict()
    del cdn256
    cdn512 = Scaden(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
        hidden_units=M512_HIDDEN_UNITS,
        do_rates=M512_DO_RATES,
    )
    cdn512.train(input_path=data)
    preds_512 = cdn512.predict()
    del cdn512
    cdn1024 = Scaden(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
        hidden_units=M1024_HIDDEN_UNITS,
        do_rates=M1024_DO_RATES,
    )
    cdn1024.train(input_path=data)
    preds_1024 = cdn1024.predict()
    del cdn1024
    preds = (preds_256 + preds_512 + preds_1024) / 3
    preds.to_csv(DATASET+"prediction"+str(i)+".csv")
batch_size=128
learning_rate=0.0001
steps=5000
seed=0
var_cutoff=0
cells=100
n_samples=5000
DATASET="Result_breast_scaden/"
for i in range(20):
    seed=i+100
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
    PREDICTION_DATA="target/breast_simulated_test_x_"+str(i)+".csv"
    data_simulated=simulation(cells, n_samples)
    processed_data=processing(PREDICTION_DATA,data_simulated,var_cutoff=var_cutoff)
    training(processed_data, batch_size, learning_rate, steps, seed)