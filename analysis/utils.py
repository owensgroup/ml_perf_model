# BSD 3-Clause License
#
# Copyright (c) 2021, The Regents of the University of California, Davis
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import torch
import argparse, json, GPUtil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .memory_bw_utils import *
from param_bench.train.compute.python.tools.eg_replay_utils import *

PM_HOME = os.environ.get('PM_HOME')
if PM_HOME is None:
    PM_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Call dirname twice to get the upper director

SUPPORTED_GPUS = {"A100", "V100", "P100", "Xp"}
def get_gpu_name():
    gpus = GPUtil.getGPUs()
    assert len(gpus) > 0, "No GPUs detected!"
    assert len(set([x.name for x in gpus])) == 1, "Platforms with hybrid GPUs not supported for now!"
    assert any(x in gpus[0].name for x in SUPPORTED_GPUS), "GPU not supported!"

    for gpu in SUPPORTED_GPUS:
        if gpu in gpus[0].name:
            return len(gpus), gpu
    return 0, None
GPU_COUNT, GPU_NAME = get_gpu_name()


HW_PARAMS = {
    "A100": {
        "peak_throughput": 12410.474,
        "peak_PCIe_BW": 8.1, # Roughly the per direction of PCIe 3.0 x16 (16 GB/s)
        "peak_DRAM_BW": 1283.578,
        "DRAM_BW_param": {
            "mul_factor": MUL_FACTOR_FUNCS["others"](1),
            "mem_ch": {
                'ln_p': 16, 
                'sats_p': 27, 
                'max_bw': 816.953003, 
                'overhead': 4.83328223
            },
            "sigmoid_param": (6.53478964, 11.78536754, 0.19557855, -3.4045424),
        }, # Use V100 param as a placeholder. TODO: Fix this.
        "peak_L2_BW": 1811.562,
        "peak_SMEM_BW": 2903.956,
        "num_SM": 108,
        "L2_size": 40 * 1024 * 1024,
        "SMEM_size": 160 * 1024,
    },
    "V100": {
        "peak_throughput": 15441.524,
        "peak_PCIe_BW": 8.1, # Roughly the per direction of PCIe 3.0 x16 (16 GB/s)
        "peak_DRAM_BW": 816.953,
        "DRAM_BW_param": {
            "mul_factor": MUL_FACTOR_FUNCS["others"](1),
            "mem_ch": {
                'ln_p': 16, 
                'sats_p': 27, 
                'max_bw': 816.953003, 
                'overhead': 4.83328223
            },
            "sigmoid_param": (6.53478964, 11.78536754, 0.19557855, -3.4045424),
        },
        "peak_L2_BW": 2847.457,
        "peak_SMEM_BW": 3918.911,
        "num_SM": 80,
        "L2_size": 6 * 1024 * 1024,
        "SMEM_size": 64 * 1024,
    },
    "Xp": {
        "peak_throughput": 10768.622,
        "peak_PCIe_BW": 3.43, # Roughly the per direction of PCIe 3.0 x8 (8 GB/s)
        "peak_DRAM_BW": 438.699,
        "DRAM_BW_param": {
            "mul_factor": MUL_FACTOR_FUNCS["others"](1),
            "mem_ch": {
                'ln_p': 16, 
                'sats_p': 27, 
                'max_bw': 816.953003, 
                'overhead': 4.83328223
            },
            "sigmoid_param": (6.53478964, 11.78536754, 0.19557855, -3.4045424),
        }, # Use V100 param as a placeholder. TODO: Fix this.
        "peak_L2_BW": 1406.454,
        "peak_SMEM_BW": 1831.258,
        "num_SM": 30,
        "L2_size": 3 * 1024 * 1024,
        "SMEM_size": 48 * 1024,
    },
    "P100": {
        "peak_throughput": 9343.711,
        "peak_PCIe_BW": 7.62, # Roughly the per direction of PCIe 3.0 x16 (16 GB/s)
        "peak_DRAM_BW": 543.406,
        "DRAM_BW_param": {
            "mul_factor": MUL_FACTOR_FUNCS["others"](1),
            "mem_ch": {
                'ln_p': 16, 
                'sats_p': 27, 
                'max_bw': 816.953003, 
                'overhead': 4.83328223
            },
            "sigmoid_param": (6.53478964, 11.78536754, 0.19557855, -3.4045424),
        }, # Use V100 param as a placeholder. TODO: Fix this.
        "peak_L2_BW": 1591.259,
        "peak_SMEM_BW": 2384.979,
        "num_SM": 56,
        "L2_size": 4 * 1024 * 1024,
        "SMEM_size": 64 * 1024,
    }
}
GPU_PARAMS = HW_PARAMS[GPU_NAME]
GPU_PARAMS["DRAM_BW_func"] = lambda x: predict_bw(x, **GPU_PARAMS["DRAM_BW_param"])
GPU_PARAMS["DRAM_BW_time"] = lambda x: predict_data_movement_time(x, **GPU_PARAMS["DRAM_BW_param"])


CPU_EVENT_OVERHEAD = 2
GPU_EVENT_OVERHEAD = 4
KERNEL_LAUNCH_LENGTH = 10


COMPUTE_STREAM = 0
MEMORY_STREAM = 1
ADDITIONAL = 2


# TODO: Distinguish conv1d and conv2d for ConvolutionBackward
CONSIDER = [
    "aten::linear", "AddmmBackward", \
    "aten::bmm", "BmmBackward", \
    "aten::matmul", "MmBackward", \
    "aten::conv1d", "ConvolutionBackward", \
    "aten::conv2d", "CudnnConvolutionBackward", \
    "LookupFunction", "LookupFunctionBackward", \
    "aten::batch_norm", "CudnnBatchNormBackward", \
    "aten::index", "IndexBackward", \
    "aten::relu", "aten::relu_", "ReluBackward", \
    "aten::sigmoid", "SigmoidBackward", \
    "aten::binary_cross_entropy", "BinaryCrossEntropyBackward", \
    "aten::mse_loss", "MseLossBackward", \
    "aten::avg_pool2d", "AvgPool2D", \
    "aten::max_pool2d", "MaxPool2DWithIndicesBackward", \
    "aten::add", "aten::add_", "aten::__and__", "aten::sub", "aten::mul", "MulBackward", \
    "aten::cat", "aten::sum", "aten::to", "aten::ones_like", \
    "autograd::engine::evaluate_function: torch::autograd::CppNode<SplitLookupFunction_sgd_Op>", \
    "torch::autograd::AccumulateGrad", \
    "torch.distributed.ddp.reducer::copy_bucket_to_grad", \
    "Optimizer.step#SGD.step", "Optimizer.zero_grad#SGD.zero_grad"
]


SKIP = [    "SliceBackward",
            "FusedDropoutBackward",
            "## Distribute emb data ##"
] # Temporary solution for ops occur during skipped intervals (see trace analysis code)
# FusedDropoutBackward somehow occurs in DeepFM exgrs


# Dummy shapes
DUMMY_SHAPES = (((-1,),), ((-1,),))


def has_comm_collective(op):
    if op.name == "record_param_comms":
        return True
    return op.get_child_by_name("record_param_comms") is not None


def is_wait_collective(op):
    return op.name in ["All2All_Wait", "All2All_Pooled_Wait"] or "All2All_ReqBackward" in op.name


def depends_on_collective_output(op, collective_output):
    return op.search_input_tensor(collective_output)


def is_all2all(op):
    return op.name == "nccl:all_to_all"


def is_allreduce(op):
    return op.name == "nccl:all_reduce"


def is_all2all_parent(op):
    return op.get_child_by_name("nccl:all_to_all") is not None


def is_allreduce_parent(op):
    return op.get_child_by_name("nccl:all_reduce") is not None


# Simple heuristic to infer if an op will schedule a kernel in a new stream
def infer_multi_stream(op):
    if has_comm_collective(op):
        return MEMORY_STREAM
    return COMPUTE_STREAM


def op_name_in_list(op, lst):
    if op.name in lst:
        return True
    if "autograd::engine::evaluate_function: " in op.name:
        bw_truncated_name = op.name.split("autograd::engine::evaluate_function: ")[-1]
        return bw_truncated_name in lst or \
                bw_truncated_name[:-1] in lst # Truncate trailing 0/1
    return False


def to_consider(op):
    return op_name_in_list(op, CONSIDER) or is_fbgemm(op)


def to_skip(op):
    return op_name_in_list(op, SKIP)


# Currently only support All2All and AllReduce
COMMS = ["nccl:all_to_all", "nccl:all_reduce"]


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )
    return value


# Utility functions
def abs_err(pred, real):
    return abs((pred - real) / real)


def err(pred, real):
    return (pred - real) / real


def geomean(x):
    return np.exp(np.log(x).mean())


def gmae(x):
    return np.exp(np.log(abs(x)).mean())


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y


def get_sigmoid_bw(s, sigmoid_param):
    return 10 ** sigmoid(s, *sigmoid_param) # L, x0, k, b


def histogram(df, perc=True, is_abs=True, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]):
    count = len(df)
    ret = {}
    if not is_abs: # Show actual error instead of abs error
        tmp_bins = []
        for i in range(0, len(bins) - 1):
            tmp_bins.append(-bins[len(bins) - 1 - i])
        for b in bins:
            tmp_bins.append(b)
        bins = tmp_bins
    for idx, b in enumerate(bins):
        if idx == 0:
            continue
        ret[(bins[idx-1], bins[idx])] = 0
    for x in df:
        xx = abs(x) if is_abs else x
        for idx, b in enumerate(bins):
            if idx == 0:
                continue
            if xx >= bins[idx-1] and xx < bins[idx]:
                ret[(bins[idx-1], bins[idx])] += 1
                break
    for b, c in sorted(ret.items(), key=lambda x: x[0]):
        if perc:
            print("{:.0f}% - {:.0f}%: {:.2f}%".format(b[0] * 100, b[1] * 100, c / count * 100))
        else:
            print("{:.2f} - {:.2f}: {:.2f}%".format(b[0], b[1], c / count * 100))
    return ret


def strip_unit(x):
    for col in ['dram_read_throughput', 'dram_write_throughput', 'gld_requested_throughput', 'gld_throughput',\
               'gst_requested_throughput', 'gst_throughput', 'l2_read_throughput', 'l2_write_throughput', \
                'shared_load_throughput', 'shared_store_throughput']:
        if col in x.keys():
            if x[col].endswith('GB/s'):
                x[col] = float(x[col].rstrip('GB/s'))
            elif x[col].endswith('MB/s'):
                x[col] = float(x[col].rstrip('MB/s')) / 1e3
            elif x[col].endswith('KB/s'):
                x[col] = float(x[col].rstrip('KB/s')) / 1e6
            elif x[col].endswith('B/s'):
                x[col] = float(x[col].rstrip('B/s')) / 1e9
            else:
                raise Exception("Unrecognizable unit!")
    return x


def p2f(x):
    for col in ['flop_dp_efficiency', 'flop_sp_efficiency', 'gld_efficiency', 'gst_efficiency', \
                'shared_efficiency', 'sm_efficiency', 'warp_execution_efficiency']:
        if col in x.keys():
            x[col] = float(str(x[col]).rstrip('%')) / 100.0
    return x


def strip_parenthesis(x):
    for col in ['dram_utilization', 'l2_utilization', 'tex_utilization']:
        if col in x.keys():
            x[col] = x[col].strip('(').strip(')')
    return x


def process_smem(x):
    # To bytes
    if 'smem' in x.keys():
        if x['smem'].endswith('MB'):
            x['smem'] = int(float(x['smem'].rstrip('MB')) * 1024 * 1024)
        elif x['smem'].endswith('KB'):
            x['smem'] = int(float(x['smem'].rstrip('KB')) * 1024)
        elif x['smem'].endswith('B'):
            x['smem'] = int(x['smem'].rstrip('B'))
        else:
            raise Exception("Unrecognizable unit!")
    return x


def preprocess(df):
    df.dropna(inplace=True)
    df = df.apply(func=p2f, axis=1)
    df = df.apply(func=strip_unit, axis=1)
    df = df.apply(func=strip_parenthesis, axis=1)
    df = df.apply(func=process_smem, axis=1)
    return df


def preprocess_fbgemm(data, backward=False):
    # Filter all irrelavant kernels
    data = data[~(
        (data["kernel_name"].str.contains('elementwise')) | \
        (data["kernel_name"].str.contains('Accessor'))
    )]
    # # Keep compute kernels only
    # data = data[(data['kernel_name'].str.contains('kernel')) & (data['kernel_name'].str.contains('embedding'))]
    if backward:
        data = data[~(data['kernel_name'].str.contains('forward'))]

    # Sum up all related kernels
    if 'dataset_path' in data.columns:
        size_columns = data.columns[:7].to_list()
    else:
        size_columns = data.columns[:6].to_list()
    data = data[size_columns + ['kernel_runtime']].groupby(size_columns[1:], as_index=False).sum()
    return data


def div_round_up(x, y):
    return int((x + y - 1) / y)


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


# def gmae_loss(output, target):
#     x = abs((torch.exp(output) - torch.exp(target)) / torch.exp(target))
#     loss = torch.exp(torch.mean(torch.log(x)))
#     return loss


# def abs_mean_loss(output, target):
#     x = abs((output - target) / target)
#     loss = torch.mean(x)
#     return loss


def remove_outliers(data):
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    return [x for x in data if x >= Q1 - 1.5 * IQR and x <= Q3 + 1.5 * IQR]


def get_pretrained_net(op_type, backward=False):
    suffix = "{}_{}".format(op_type, 1 if not backward else 0)
    with open("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix), "r") as f:
        best_config = json.load(f)
        n_hidden = [best_config["size"]] * best_config["num_layers"]
    if op_type == "embedding_lookup":
        n_feature = 21
    elif op_type == "fully_connected":
        n_feature = 4
    elif op_type == "conv2d":
        n_feature = 9
    elif op_type == "conv1d":
        n_feature = 5
    elif op_type == "transpose":
        n_feature = 3
    elif op_type == "bn":
        n_feature = 3
    else: # tril
        n_feature = 4
    net = MLP(n_feature=n_feature, n_hidden=n_hidden, n_output=1)
    net.load_state_dict(torch.load("{}/analysis/ml_predictors/{}/predictor_{}.pth".format(PM_HOME, GPU_NAME, suffix)))
    return net


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        if not isinstance(n_hidden, list):
            n_hidden = list(n_hidden)
        prv = n_feature
        self.layers = torch.nn.ModuleList()
        for nxt in n_hidden:
            self.layers.append(torch.nn.Linear(prv, nxt))
            self.layers.append(torch.nn.ReLU())
            prv = nxt
        self.layers.append(torch.nn.Linear(prv, n_output))
    def forward(self, X, fbgemm=False):
        if fbgemm:
            device = X[0].device
            X_len = torch.tensor([x.shape[0] for x in X]).to(device)

            X = torch.cat(X, dim=0)
            for l in self.layers[:-3]: # Hold for the last two FCs and one sigmoid
                X = l(X)

            ind = torch.repeat_interleave(torch.arange(len(X_len)).to(device), X_len).to(device)
            tmp = torch.zeros((X_len.shape[0], X.shape[1])).to(device)
            tmp.index_add_(0, ind, X)
            X = tmp

            for l in self.layers[-3:]:
                X = l(X)
        else:
            for l in self.layers:
                X = l(X)

        return X.view(-1)


def transform_emb_data(data, table_configs, test_frac=0.2):
    train_data, test_data = (data, None) if test_frac == 1.0 else train_test_split(data, test_size=test_frac)

    def f(x):
        return torch.tensor([
            ([
                np.log(x['batch_size']),
                np.log(table_configs[x['dataset_path']][int(t_idx)]['num_embeddings']),
                np.log(float(L) if float(L) != 0.0 else 1e-3), # Avoid L = 0
                np.log(float(D)),
            ] + [
                float(rf) for rf in rfs.split('-') # All in range (0, 1)
            ]) for t_idx, L, D, rfs in zip(
                x['num_embeddings'].split('-'),
                x['bag_size'].split('-'),
                x['embedding_dim'].split('-'),
                x['reuse_factors'].split('_'),
            )
        ], dtype=torch.float32)

    train_x = [x for x in train_data.apply(f, axis=1).tolist()]
    train_y = torch.tensor(np.log(train_data["kernel_runtime"].values), dtype=torch.float32)
    if test_frac == 1.0: # Use train data as test data if only testing
        return None, None, train_x, train_y

    test_x = [x for x in test_data.apply(f, axis=1).tolist()]
    test_y = torch.tensor(np.log(test_data["kernel_runtime"].values), dtype=torch.float32)

    return train_x, train_y, test_x, test_y


def get_emb_train_test_data(backward=False, test_frac=0.2, **kwargs):
    suffix = ('_' + kwargs['suffix']) if 'suffix' in kwargs.keys() else ''
    data_path = '{}/data/{}/kernel/embedding_lookup_{}{}.csv'.format(
        PM_HOME,
        GPU_NAME,
        '1' if not backward else '0_sgd',
        suffix,
    )
    data = pd.read_csv(data_path, delimiter=',')
    data = preprocess_fbgemm(data, backward=backward)

    rf_file_path = '/'.join(data_path.split('/')[:-1] + ['embedding_lookup_{}_rf.csv'.format(kwargs['suffix'])])
    rf = pd.read_csv(rf_file_path, delimiter=',')
    # print(data.shape, rf.shape)
    # print(data.shape)#drop_duplicates(subset=['num_embeddings'], keep='last').shape)
    data = pd.merge(data, rf, on=data.columns[:6].tolist()) # kernel_name & BETLD
    # print(data.shape, data.drop_duplicates(subset=data.columns, keep='last'))
    # print(data.iloc[6])
    # print(data.columns, rf.columns)
    # exit()

    dataset_paths = data["dataset_path"].unique().tolist()
    table_config_paths = [(os.path.splitext(x)[0] + '_configs.json') for x in dataset_paths]
    table_configs = {}
    for dp, cp in zip(dataset_paths, table_config_paths):
        with open(cp) as f:
            table_config = json.load(f)["tables"]
        table_configs[dp] = table_config

    train_x, train_y, test_x, test_y = transform_emb_data(data, table_configs, test_frac=test_frac)
    return test_x[0].shape[1], train_x, train_y, test_x, test_y


def get_train_test_data(op_type, backward=False, test_frac=0.2, **kwargs):
    if op_type == "embedding_lookup":
        return get_emb_train_test_data(backward=backward, test_frac=test_frac, **kwargs)

    data = pd.read_csv('{}/data/{}/kernel/{}_{}.csv'.format(PM_HOME, GPU_NAME, op_type, 1 if not backward else 0), delimiter=',')
    data = preprocess(data)
    if op_type == 'fully_connected':
        data = data[data['kernel_name'].str.contains("sgemm")] # Train on samples with 'sgemm' in kernel name
        df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'K': np.log(data['K']),
            'kernel_runtime': np.log(data['kernel_runtime']),
        })
    elif op_type == 'conv2d':
        data = data[~(data['kernel_name'].str.contains("at::"))] # Exclude ATen kernels for training
        data = data[['batch_size', 'H', 'W', 'IC', 'OC', 'stride', 'dilation', 'FH', 'FW', 'is_dw', 'kernel_runtime']].groupby(['batch_size', 'H', 'W', 'IC', 'OC', 'stride', 'dilation', 'FH', 'FW', 'is_dw'], as_index=False).sum() # Sum up all kernels
        df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'H': np.log(data['H']),
            'IC': np.log(data['IC']),
            'OC': np.log(data['OC']),
            'stride': data['stride'],
            'dilation': data['dilation'],
            'FH': data['FH'],
            'FW': data['FW'],
            'is_dw': data['is_dw'],
            'kernel_runtime': np.log(data['kernel_runtime']),
        })
    elif op_type == 'conv1d':
        data = data[~((data['kernel_name'].str.contains("at::")) & (~(data['kernel_name'].str.contains("conv"))))] # Exclude ATen kernels that are not conv for training
        data = data[['batch_size', 'L', 'IC', 'OC', 'groups', 'kernel_runtime']].groupby(['batch_size', 'L', 'IC', 'OC', 'groups'], as_index=False).sum() # Sum up all kernels
        df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'L': np.log(data['L']),
            'IC': np.log(data['IC']),
            'OC': np.log(data['OC']),
            'groups': data['groups'],
            'kernel_runtime': np.log(data['kernel_runtime']),
        })
    elif op_type == 'transpose':
        df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'kernel_runtime': np.log(data['kernel_runtime']),
        })
    elif op_type == 'bn':
        data = data[data['kernel_name'].str.contains("bn")] # Train on samples with 'bn' in kernel name
        df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'H': np.log(data['H']),
            'OC': np.log(data['OC']),
            'kernel_runtime': np.log(data['kernel_runtime']),
        })
    else: # tril
        if backward:
            data = data[['batch_size', 'M', 'N', 'diag', 'kernel_runtime']].groupby(['batch_size', 'M', 'N', 'diag'], as_index=False).sum() # Sum up all kernels
        df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'diag': data['diag'],
            'kernel_runtime': np.log(data['kernel_runtime']),
        })

    train_data, test_data = train_test_split(df, test_size=test_frac)
    train_x = torch.tensor(train_data[train_data.columns[:-1]].values).float()
    train_y = torch.tensor(train_data[train_data.columns[-1]].values).float()
    test_x = torch.tensor(test_data[test_data.columns[:-1]].values).float()
    test_y = torch.tensor(test_data[test_data.columns[-1]].values).float()

    return train_x.shape[1], train_x, train_y, test_x, test_y
