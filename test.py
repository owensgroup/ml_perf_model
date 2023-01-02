import pandas as pd
import numpy as np
import torch, os, json, copy
import torch.nn as nn
from torch.nn import functional as F
from analysis.inference import preprocess, preprocess_fbgemm, abs_err, gmae

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Table feature extraction
        self.t_fc_0 = nn.Linear(21, 128)
        self.t_fc_1 = nn.Linear(128, 32)

        # Final MLPs of cost estimation head
        self.c_fc_0 = nn.Linear(self.t_fc_1.out_features, 64)
        self.c_fc_1 = nn.Linear(64, 1)

    def forward(self, X, y):
        X_len = torch.tensor([x.shape[0] for x in X])
        B = X_len.shape[0]

        X = torch.cat(X, dim=0)
        X = F.relu(self.t_fc_0(X))
        X = F.relu(self.t_fc_1(X))

        ind = torch.repeat_interleave(torch.arange(len(X_len)), X_len)
        tmp = torch.zeros((X_len.shape[0], X.shape[1]))
        tmp.index_add_(0, ind, X)
        X = tmp

        cost = F.relu(self.c_fc_0(X))
        cost = self.c_fc_1(cost)
        cost = cost.view(B)

        return cost


def transform(data, table_configs):
    Xs = []
    for _, x in data.iterrows():
        # Table indices are under 'num_embeddings'
        _X = torch.tensor([[table_configs[int(t_idx)][key] for key in [
            "dim",
            "row",
            "pooling_factor",
            "size",
            "bin_0",
            "bin_1",
            "bin_2",
            "bin_3",
            "bin_4",
            "bin_5",
            "bin_6",
            "bin_7",
            "bin_8",
            "bin_9",
            "bin_10",
            "bin_11",
            "bin_12",
            "bin_13",
            "bin_14",
            "bin_15",
            "bin_16",
        ]] for t_idx in x['num_embeddings'].split('-')], dtype=torch.float32)
        Xs.append(_X)
    ys = data["kernel_runtime"].tolist()

    return Xs, ys


def test(pth_file, Xs, ys):
    net = Net()
    optimizer = torch.optim.Adam(
        list(net.parameters()),
        lr=0.001
    )

    checkpoint = torch.load(pth_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    net.eval()
    with torch.no_grad():
        pred_y = net.forward(Xs, ys)

    return pred_y

if __name__ == '__main__':
    # Load table configs
    table_config_path = os.path.join("../experimental/sharding/data/dlrm_datasets/table_configs.json")
    with open(table_config_path) as f:
        table_configs = json.load(f)["tables"]
        for i in range(len(table_configs)):
            table_configs[i]["size"] = table_configs[i]["dim"] * table_configs[i]["row"]

    # Normalize the features
    table_configs = copy.deepcopy(table_configs)
    features = ["row", "dim", "size", "pooling_factor"]
    for feature in features:
        vals = [table_config[feature] for table_config in table_configs]
        mean = np.mean(vals)
        std = np.std(vals)
        for i in range(len(table_configs)):
            table_configs[i][feature] = (table_configs[i][feature] - mean) / std

    data = pd.read_csv('data/A100/kernel/embedding_lookup_1_fbgemm_dlrm_datasets.csv')
    data = preprocess(data)
    data = preprocess_fbgemm(data)
    # data.insert(0, 'kernel_name', ['dummy'] * len(data))

    Xs, ys = transform(data, table_configs)
    time = test(
        '/home/m092926/wario/Documents/ml_perf_model/pts/multi_table_cost_model_fw_dim_512.pth',
        Xs, ys
    )
    error = abs_err(time, data['kernel_runtime'])
    print("ELF flexible: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
