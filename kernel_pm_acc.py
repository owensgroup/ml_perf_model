from analysis.ml_predictors.mlp import inference
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import argparse, json, os
import torch

# Utility functions
def abs_err(pred, real):
    return abs((pred - real) / real)

def err(pred, real):
    return (pred - real) / real

def gmae(x):
    return np.exp(np.log(abs(x)).mean())

def histogram(df, perc=True, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]):
    count = len(df)
    ret = {}
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
    df = df[(df['kernel_name'] != 'gemv2T_kernel') & (df['kernel_name'] != 'splitKreduce_kernel')]
    return df

def div_round_up(x, y):
    return int((x + y - 1) / y)

# V100
peak_throughput = 15441.524
peak_PCIe_BW = 5.9 # 16
peak_DRAM_BW = 804.497
peak_L2_BW = 2847.457
peak_SMEM_BW = 3918.911
num_SM = 80
L2_size = 6 * 1024 * 1024
SMEM_size = 96 * 1024
regs_size = 64 * 1024 * 4
maximum_CTA_per_SM = 32
maximum_warps_per_SM = 64
frequency = 876 # MHz, for memory

# Titan XP
# peak_throughput = 13422.779
# peak_PCIe_BW = 5.9 # 16
# peak_DRAM_BW = 438.699
# peak_L2_BW = 1406.454
# peak_SMEM_BW = 1831.258
# num_SM = 60
# L2_size = 3 * 1024 * 1024
# SMEM_size = 48 * 1024
# path_prefix = '~/mario/Documents/ml_perf_model'

pm_home = os.environ.get('PM_HOME')
if pm_home is None:
    pm_home = "/home/m092926/daisy/Documents/ml_perf_model"
dir_prefix = "{}/analysis/ml_predictors".format(pm_home)

def get_data(op_type, backward=False):
    data = pd.read_csv('{}/../../data/{}_{}.csv'.format(dir_prefix, op_type, 1 if not backward else 0), delimiter=',')
    data = preprocessing(data)

    if op_type == 'fully_connected':
        data = data[data['kernel_name'].str.contains("sgemm")] # Train on samples with 'sgemm' in kernel name
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'K': np.log(data['K'])
        })
    elif op_type == 'transpose':
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N'])
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

    return x.shape[1], x.to('cuda:0'), y.to('cuda:0')


def infer_concat():
    concat_data = pd.read_csv('{}/data/concat_1.csv'.format(pm_home), delimiter=',')
    concat_data = preprocessing(concat_data)
    concat_data = concat_data[concat_data["batch_size"] > 1]
    A_size = concat_data["batch_size"] * concat_data["M"] * concat_data["K"]
    B_size = concat_data["batch_size"] * concat_data["N"] * concat_data["K"]
    actual_peak_DRAM_BW = 0.79 * peak_DRAM_BW
    concat_traffic = 2 * (A_size + B_size) * 4
    estimated_time = concat_traffic / actual_peak_DRAM_BW / 1000
    error = abs_err(estimated_time, concat_data['kernel_runtime'])
    print("Concat: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))


def infer_memcpy():
    memcpy_data = pd.read_csv('{}/data/memcpy_1.csv'.format(pm_home), delimiter=',')
    memcpy_data = preprocessing(memcpy_data)
    memcpy_traffic = memcpy_data['batch_size'] * memcpy_data['M'] * memcpy_data['N'] * 4
    estimated_time = memcpy_traffic / 10.83 / 1000
    error = abs_err(estimated_time, memcpy_data['kernel_runtime'])
    print("Memcpy: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))


def embedding_forward_simple(**kwargs):
    B = kwargs["batch_size"]
    E = kwargs["num_embeddings"]
    T = kwargs["num_tables"]
    L = kwargs["bag_size"]
    D = kwargs["embedding_dim"]

    table_offsets_t = 32
    offsets_t = 64
    indices_t = div_round_up(4 * L, 32) * 32
    weights_t = L * div_round_up(4 * D, 32) * 32
    output_t = div_round_up(4 * D, 32) * 32
    total_t = B * T * (table_offsets_t + offsets_t + indices_t + weights_t + output_t)
    return total_t / peak_DRAM_BW / 1000


def embedding_forward_predictor(**kwargs):
    # hit_rate = C(X, L) / C(E, L), X = avg_num_rows_per_table
    def hit_rate(X, E, L):
        ret = 1.0
        e = E
        x = X
        for idx in range(L):
            ret *= x / e
            x -= 1
            e -= 1
        return ret

    # Average number of rows per table in L2
    y = kwargs
    num_total_warps = y["batch_size"] * y["num_tables"] # Total warp number of the kernel
    num_warps_per_sm = y["rows_per_block"] # Number of warps per sm
    num_warps_simul = num_SM * num_warps_per_sm # Total number of warps simultaneously running on the device
    num_tables_simul = (num_warps_simul + y["batch_size"] - 1) // y["batch_size"] # Number of tables simultaneously being accessed on the device
    avg_table_size = min(L2_size // num_tables_simul, y["num_embeddings"] * y["embedding_dim"] * 4) # Average table size that reside on the device
    indices_size = 0
    avg_num_rows_per_table = (avg_table_size - indices_size) // 4 // y["embedding_dim"]

    # Hit rate
    hr = hit_rate(avg_num_rows_per_table, y["num_embeddings"], y["bag_size"])

    # Traffics
    table_offsets_traffic = 32
    offsets_traffic = 64
    indices_dram_traffic = div_round_up(y["bag_size"] * 4, 32) * 32
    indices_l2_traffic = 0
    table_traffic = y["bag_size"] * (div_round_up(y["embedding_dim"] * 4, 32) * 32)
    output_traffic = (div_round_up(y["embedding_dim"] * 4, 32) * 32)

    total_l2_traffic = ((table_offsets_traffic + offsets_traffic + indices_l2_traffic) * y["batch_size"] + \
                        hr * (table_traffic * y["batch_size"] - avg_table_size)) * y["num_tables"]
    total_dram_traffic = ((indices_dram_traffic + output_traffic) * y["batch_size"] + \
                          (1 - hr) * (table_traffic * y["batch_size"] - avg_table_size) + avg_table_size) * y["num_tables"]

    return max(total_dram_traffic / peak_DRAM_BW / 1000.0, total_l2_traffic / peak_L2_BW / 1000.0)


def embedding_backward_sgd_simple(**kwargs):
    y = kwargs
    indices_traffic = div_round_up(y["bag_size"] * 4, 32) * 32
    grad_output_traffic = div_round_up(y["embedding_dim"] * 4, 32) * 32

    # Traffic per warp = t_offsets + t_table_offsets + t_indices + t_weights + t_grad_outputs
    total_traffic_per_warp = 32 + \
                            64 + \
                            indices_traffic + \
                            2 * y["bag_size"] * (div_round_up(y["embedding_dim"] * 4, 32) * 32) + \
                            grad_output_traffic

    # Traffic = warp * traffic per warp
    total_traffic = y["batch_size"] * y["num_tables"] * total_traffic_per_warp

    # Total compute throughput
    mac_per_warp = y["bag_size"] * 4 * (y["embedding_dim"] // 4)
    total_mac = y["batch_size"] * y["num_tables"] * mac_per_warp

    return max(total_traffic / peak_DRAM_BW / 1000, total_mac / peak_throughput / 1000)


def embedding_backward_sgd_predictor(**kwargs):
    # hit_rate = C(X, L) / C(E, L), X = avg_num_rows_per_table
    def hit_rate(X, E, L):
        ret = 1.0
        e = E
        x = X
        for idx in range(L):
            ret *= x / e
            x -= 1
            e -= 1
        return ret

    # Average number of rows per table in L2
    y = kwargs
    num_total_warps = y["batch_size"] * y["num_tables"] # Total warp number of the kernel
    num_warps_per_sm = y["rows_per_block"] # Number of warps per sm
    num_warps_simul = num_SM * num_warps_per_sm # Total number of warps simultaneously running on the device
    num_tables_simul = (num_warps_simul + y["batch_size"] - 1) // y["batch_size"] # Number of tables simultaneously being accessed on the device
    avg_table_size = min(L2_size // num_tables_simul, y["num_embeddings"] * y["embedding_dim"] * 4) # Average table size that reside on the device
    indices_size = 0
    avg_num_rows_per_table = (avg_table_size - indices_size) // 4 // y["embedding_dim"]

    # Hit rate
    hr = hit_rate(avg_num_rows_per_table, y["num_embeddings"], y["bag_size"])

    # Traffics
    table_offsets_traffic = 32
    offsets_traffic = 64
    indices_dram_traffic = div_round_up(y["bag_size"] * 4, 32) * 32
    indices_l2_traffic = 0
    table_traffic = 2 * y["bag_size"] * (div_round_up(y["embedding_dim"] * 4, 32) * 32)
    output_traffic = (div_round_up(y["embedding_dim"] * 4, 32) * 32)

    total_l2_traffic = ((table_offsets_traffic + offsets_traffic + indices_l2_traffic) * y["batch_size"] + \
                        hr * (table_traffic * y["batch_size"] - avg_table_size)) * y["num_tables"]
    total_dram_traffic = ((indices_dram_traffic + output_traffic) * y["batch_size"] + \
                          (1 - hr) * (table_traffic * y["batch_size"] - avg_table_size) + avg_table_size) * y["num_tables"]

    # Total compute throughput
    mac_per_warp = y["bag_size"] * 4 * (y["embedding_dim"] // 4)
    total_mac = y["batch_size"] * y["num_tables"] * mac_per_warp

    return max(total_dram_traffic / peak_DRAM_BW / 1000.0, total_l2_traffic / peak_L2_BW / 1000.0, total_mac / peak_throughput / 1000)


def infer_elf():
    data = pd.read_csv('{}/data/embedding_lookup_1_shmem.csv'.format(pm_home), delimiter=',')
    data = preprocessing(data)
    data = data[data["kernel_name"].str.contains("batched_embedding")]
    data = data[data['batch_size'] > 1]

    time_all = data.apply(lambda x: embedding_forward_simple(**x[1:6]), axis=1)
    error_all = abs_err(time_all, data['kernel_runtime'])
    print("ELF all sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_all) * 100.0, error_all.mean() * 100.0, error_all.std() * 100.0))
    time_big = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_simple(**x[1:6]), axis=1)
    error_big = abs_err(time_big, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
    print("ELF big sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_big) * 100.0, error_big.mean() * 100.0, error_big.std() * 100.0))

    time_all = data.apply(lambda x: embedding_forward_predictor(**x[1:7]), axis=1)
    error_all = abs_err(time_all, data['kernel_runtime'])
    print("ELF all sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_all) * 100.0, error_all.mean() * 100.0, error_all.std() * 100.0))
    time_big = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_predictor(**x[1:7]), axis=1)
    error_big = abs_err(time_big, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
    print("ELF big sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_big) * 100.0, error_big.mean() * 100.0, error_big.std() * 100.0))


def infer_elb():
    data = pd.read_csv('{}/data/embedding_lookup_0_sgd_shmem.csv'.format(pm_home), delimiter=',')
    data = preprocessing(data)
    data = data[data["kernel_name"].str.contains("batched_embedding")]
    data = data[data['batch_size'] > 1]

    time_all = data.apply(lambda x: embedding_backward_sgd_simple(**x[1:6]), axis=1)
    error_all = abs_err(time_all, data['kernel_runtime'])
    print("ELB all sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_all) * 100.0, error_all.mean() * 100.0, error_all.std() * 100.0))
    time_big = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_simple(**x[1:6]), axis=1)
    error_big = abs_err(time_big, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
    print("ELB big sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_big) * 100.0, error_big.mean() * 100.0, error_big.std() * 100.0))

    time_all = data.apply(lambda x: embedding_backward_sgd_predictor(**x[1:7]), axis=1)
    error_all = abs_err(time_all, data['kernel_runtime'])
    print("ELBH all sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_all) * 100.0, error_all.mean() * 100.0, error_all.std() * 100.0))
    time_big = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_predictor(**x[1:7]), axis=1)
    error_big = abs_err(time_big, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
    print("ELBH big sizes: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error_big) * 100.0, error_big.mean() * 100.0, error_big.std() * 100.0))


def infer_el(p):
    if p == "forward":
        infer_elf()
    else:
        infer_elb()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training MLP performance model for FC, transpose, and tril.")
    parser.add_argument("--op-type", type=str, default="all")
    parser.add_argument("--backward", action="store_true", default=False)
    args = parser.parse_args()

    if args.op_type == "all":
        for t in ['embedding_lookup', "fully_connected", "concat", "memcpy", "transpose", "tril"]:
            for p in ["forward", "backward"]:
                if (t == "fully_connected" or t == "transpose" or t == "concat" or t == "memcpy") and p == "backward":
                    continue
                if t == "fully_connected" or t == "transpose" or t == "tril":
                    suffix = "{}_{}".format(t, 1 if p == "forward" else 0)
                    from analysis.ml_predictors.mlp import get_pretrained_net, get_data
                    if os.path.exists("{}/best_config_{}.json".format(dir_prefix, suffix)):
                        n_feature, x, y = get_data(t, p == "backward")
                        with open("{}/best_config_{}.json".format(dir_prefix, suffix), "r") as f:
                            best_config = json.load(f)
                        net = get_pretrained_net(t, p == "backward")
                        estimated_time = torch.exp(net(x.cpu()).detach().view(-1))
                        error = abs_err(estimated_time, torch.exp(y.cpu().detach()).view(-1))
                        min_error = gmae(error)
                        print("{}-{}: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(t, p, gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
                if t == "concat":
                    infer_concat()
                if t == "memcpy":
                    infer_memcpy()
                if t == "embedding_lookup":
                    infer_el(p)
                