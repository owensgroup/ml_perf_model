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

import torch
import argparse, json, os, GPUtil
import numpy as np
import pandas as pd

PM_HOME = os.environ.get('PM_HOME')
if PM_HOME is None:
    PM_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Call dirname twice to get the upper director


def get_gpu_name():
    gpus = GPUtil.getGPUs()
    assert len(gpus) > 0, "No GPUs detected!"

    for gpu in gpus:
        if "A100" in gpu.name:
            return "A100"
        if "V100" in gpu.name:
            return "V100"
        if "P100" in gpu.name:
            return "P100"
        if "Xp" in gpu.name:
            return "Xp"
        if "1080" in gpu.name:
            return "1080"
    return None
GPU_NAME = get_gpu_name()


HW_PARAMS = {
    "A100": {
        "peak_throughput": 12410.474,
        "peak_PCIe_BW": 8.1, # Roughly the per direction of PCIe 3.0 x16 (16 GB/s)
        "peak_DRAM_BW": 1283.578,
        "peak_L2_BW": 1811.562,
        "peak_SMEM_BW": 2903.956,
        "num_SM": 108,
        "L2_size": 40 * 1024 * 1024,
        "SMEM_size": 160 * 1024,
    },
    "V100": {
        "peak_throughput": 15441.524,
        "peak_PCIe_BW": 8.1, # Roughly the per direction of PCIe 3.0 x16 (16 GB/s)
        "peak_DRAM_BW": 804.497,
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
        "peak_L2_BW": 1591.259,
        "peak_SMEM_BW": 2384.979,
        "num_SM": 56,
        "L2_size": 4 * 1024 * 1024,
        "SMEM_size": 64 * 1024,
    }
}
GPU_PARAMS = HW_PARAMS[GPU_NAME]


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
                "torch::autograd::AccumulateGrad", "torch.distributed.ddp.reducer::copy_bucket_to_grad", \
                "Optimizer.step#SGD.step", "Optimizer.zero_grad#SGD.zero_grad"
]


SKIP = [    "SliceBackward",
            "FusedDropoutBackward",
            "DLRM distribute emb data"
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
    return op_name_in_list(op, CONSIDER)


def to_skip(op):
    return op_name_in_list(op, SKIP)


# alg_bw -> bus_bw for multi-gpu collectives
MUL_FACTOR_FUNCS = {
    'all_to_all': lambda n: (n-1) / n,
    'all_to_allv': lambda n: (n-1) / n,
    'all_reduce': lambda n: 2 * (n-1) / n,
    'all_gather': lambda n: (n-1) / n,
    'all_gather_base': lambda n: (n-1) / n,
    'reduce': lambda n: 1,
    'reduce_scatter': lambda n: (n-1) / n
}


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


def preprocessing(df):
    df.dropna(inplace=True)
    df = df.apply(func=p2f, axis=1)
    df = df.apply(func=strip_unit, axis=1)
    df = df.apply(func=strip_parenthesis, axis=1)
    df = df.apply(func=process_smem, axis=1)
    return df


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
    if op_type == "fully_connected":
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
            self.layers.append(torch.nn.Sigmoid())
            prv = nxt
        self.layers.append(torch.nn.Linear(prv, n_output))
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def get_data(op_type, backward=False, gpu=True):
    data = pd.read_csv('{}/data/{}/kernel/{}_{}.csv'.format(PM_HOME, GPU_NAME, op_type, 1 if not backward else 0), delimiter=',')
    data = preprocessing(data)

    if op_type == 'fully_connected':
        data = data[data['kernel_name'].str.contains("sgemm")] # Train on samples with 'sgemm' in kernel name
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'K': np.log(data['K'])
        })
    elif op_type == 'conv2d':
        data = data[~(data['kernel_name'].str.contains("at::"))] # Exclude ATen kernels for training
        data = data[['batch_size', 'H', 'W', 'IC', 'OC', 'stride', 'dilation', 'FH', 'FW', 'is_dw', 'kernel_runtime']].groupby(['batch_size', 'H', 'W', 'IC', 'OC', 'stride', 'dilation', 'FH', 'FW', 'is_dw'], as_index=False).sum() # Sum up all kernels
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'H': np.log(data['H']),
            'IC': np.log(data['IC']),
            'OC': np.log(data['OC']),
            'stride': data['stride'],
            'dilation': data['dilation'],
            'FH': data['FH'],
            'FW': data['FW'],
            'is_dw': data['is_dw'],
        })
    elif op_type == 'conv1d':
        data = data[~((data['kernel_name'].str.contains("at::")) & (~(data['kernel_name'].str.contains("conv"))))] # Exclude ATen kernels that are not conv for training
        data = data[['batch_size', 'L', 'IC', 'OC', 'groups', 'kernel_runtime']].groupby(['batch_size', 'L', 'IC', 'OC', 'groups'], as_index=False).sum() # Sum up all kernels
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'L': np.log(data['L']),
            'IC': np.log(data['IC']),
            'OC': np.log(data['OC']),
            'groups': data['groups'],
        })
    elif op_type == 'transpose':
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N'])
        })
    elif op_type == 'bn':
        data = data[data['kernel_name'].str.contains("cudnn")] # Train on samples with 'cudnn' in kernel name
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'H': np.log(data['H']),
            'OC': np.log(data['OC']),
        })
    else: # tril
        if backward:
            data = data[['batch_size', 'M', 'N', 'diag', 'kernel_runtime']].groupby(['batch_size', 'M', 'N', 'diag'], as_index=False).sum() # Sum up all kernels
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'diag': data['diag']
        })

    x = torch.tensor(input_df.values).float()

    output_df = pd.DataFrame({
        'kernel_runtime': np.log(data['kernel_runtime'])
    })
    y = torch.tensor(output_df.values).float()

    if gpu:
        x, y = x.to('cuda:0'), y.to('cuda:0')

    return x.shape[1], x, y
