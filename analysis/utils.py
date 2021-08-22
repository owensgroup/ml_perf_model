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
    "V100": {
        "peak_throughput": 15441.524,
        "peak_PCIe_BW": 10.83, # 16
        "peak_DRAM_BW": 804.497,
        "peak_L2_BW": 2847.457,
        "peak_SMEM_BW": 3918.911,
        "num_SM": 80,
        "L2_size": 6 * 1024 * 1024,
    },
    "Xp": {
        "peak_throughput": 13422.779,
        "peak_PCIe_BW": 3.63, # 16
        "peak_DRAM_BW": 438.699,
        "peak_L2_BW": 1406.454,
        "peak_SMEM_BW": 1831.258,
        "num_SM": 60,
        "L2_size": 3 * 1024 * 1024,
    },
    "P100": {
        "peak_throughput": 9343.711,
        "peak_PCIe_BW": 10.33, # 16
        "peak_DRAM_BW": 547.798,
        "peak_L2_BW": 1591.259,
        "peak_SMEM_BW": 2384.979,
        "num_SM": 56,
        "L2_size": 4 * 1024 * 1024,
    },
    "1080": {
        "peak_throughput": 9494.746,
        "peak_PCIe_BW": 3.04,
        "peak_DRAM_BW": 246.890,
        "peak_L2_BW": 1747.627,
        "peak_SMEM_BW": 1200.361,
        "num_SM": 20,
        "L2_size": 2 * 1024 * 1024,
    }
}
GPU_PARAMS = HW_PARAMS[GPU_NAME]


CPU_EVENT_OVERHEAD = 3.8
GPU_EVENT_OVERHEAD = 4
KERNEL_LAUNCH_LENGTH = 10


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


def gmae(x):
    return np.exp(np.log(abs(x)).mean())


def histogram(df, perc=True, is_abs=False, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]):
    count = len(df)
    ret = {}
    if is_abs:
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
        for idx, b in enumerate(bins):
            if idx == 0:
                continue
            if x >= bins[idx-1] and x < bins[idx]:
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


def get_pretrained_net(op_type, backward=False):
    suffix = "{}_{}".format(op_type, 1 if not backward else 0)
    with open("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix), "r") as f:
        best_config = json.load(f)
        n_hidden = [best_config["size"]] * best_config["num_layers"]
    if op_type == "fully_connected":
        n_feature = 4
    if op_type == "conv":
        n_feature = 9
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
    elif op_type == 'conv':
        data = data[data['kernel_name'].str.contains("at::") == False] # Exclude ATen kernels for training
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
