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
import pandas as pd
import numpy as np
import json, math, zlib
from scipy.stats import binom
from .utils import *
from .memory_bw_utils import *
import analysis.extend_distributed as ext_dist

peak_throughput = GPU_PARAMS["peak_throughput"]
peak_PCIe_BW = GPU_PARAMS["peak_PCIe_BW"]
peak_DRAM_BW = GPU_PARAMS["peak_DRAM_BW"]
DRAM_BW_func = GPU_PARAMS["DRAM_BW_func"]
peak_L2_BW = GPU_PARAMS["peak_L2_BW"]
peak_SMEM_BW = GPU_PARAMS["peak_SMEM_BW"]
num_SM = GPU_PARAMS["num_SM"]
L2_size = GPU_PARAMS["L2_size"]
SMEM_size = GPU_PARAMS["SMEM_size"]


# Stack that buffers embedding ops in FW for time prediction in BW
embedding_ops_stack = []


def embedding_forward_predictor_simple(**kwargs):
    B = kwargs["batch_size"]
    T = kwargs["num_tables"]
    L = kwargs["bag_size"]
    D = kwargs["embedding_dim"]

    table_offsets_t = 32
    offsets_t = 64
    indices_t = div_round_up(4 * L, 32) * 32
    weights_t = L * div_round_up(4 * D, 32) * 32
    output_t = div_round_up(4 * D, 32) * 32
    total_t = B * T * (table_offsets_t + offsets_t + indices_t + weights_t + output_t)
    return GPU_PARAMS["DRAM_BW_time"](total_t)


def get_cached_row_count(**kwargs):
    # Average number of rows per table in L2
    y = kwargs
    if "rows_per_block" in y.keys():
        num_warps_per_sm = y["rows_per_block"] # Number of warps per sm
    else: # FBGEMM
        num_warps_per_sm = 45 # 64 * 0.71 (avg achieved occupancy)

    num_warps_simul = num_SM * num_warps_per_sm # Total number of warps simultaneously running on the device
    num_tables_simul = math.ceil(num_warps_simul / y["batch_size"]) # Number of tables simultaneously being accessed on the device
    avg_num_rows_per_table = min(
        int(L2_size * 0.9 / num_tables_simul / 4 / y["embedding_dim"]), 
        y["num_embeddings"],
        y["batch_size"] * y["bag_size"]
    ) # Average table size that reside on the device
    return avg_num_rows_per_table


# Ready for same E & D & L (both FBGEMM and non-), not ready for flexible FBGEMM. TODO: Fix this.
def embedding_forward_predictor(**kwargs):
    avg_num_rows_per_table_in_l2 = get_cached_row_count(**kwargs)
    B = kwargs["batch_size"]
    E = kwargs["num_embeddings"]
    T = kwargs["num_tables"]
    L = kwargs["bag_size"]
    D = kwargs["embedding_dim"]

    # Hit rate
    u = binom.pmf(0, B * L, 1 / E)
    # Cold start considered
    if (E - u * E) > avg_num_rows_per_table_in_l2: # Cache not able to hold all unique rows for each table
        X = avg_num_rows_per_table_in_l2
        hr = 1 - (X + (B * L - X) * (E - X) / E) / (B * L)
    else:
        hr = 1 - (E - u * E) / (B * L)

    # Traffics
    table_offsets_t = 32
    offsets_t = 64
    indices_t = div_round_up(4 * L, 32) * 32
    weights_t = L * div_round_up(4 * D, 32) * 32
    output_t = div_round_up(4 * D, 32) * 32

    total_l2_traffic = T * B * hr * (weights_t)
    total_dram_traffic = T * (
        B * (indices_t + table_offsets_t + offsets_t + output_t) +
        ((B * (1 - hr) * weights_t))
    )
    factor = 1 if D > 64 else (1.1 if D > 32 else 1.55) # Empirical, probably divergence. TODO: Explain this

    return max(
        GPU_PARAMS["DRAM_BW_time"](total_dram_traffic),
        total_l2_traffic / GPU_PARAMS["L2_BW_func"](total_l2_traffic) / 1000.0
    ) * factor


def embedding_backward_sgd_predictor_simple(**kwargs):
    B = kwargs["batch_size"]
    T = kwargs["num_tables"]
    L = kwargs["bag_size"]
    D = kwargs["embedding_dim"]

    # Traffic per warp = t_offsets + t_table_offsets + t_indices + t_weights + t_grad_outputs
    total_t_per_warp = 32 + \
                        64 + \
                        div_round_up(L * 4, 32) * 32 + \
                        2 * L * (div_round_up(D * 4, 32) * 32) + \
                        div_round_up(D * 4, 32) * 32

    # Traffic = warp * traffic per warp
    total_t = B * T * total_t_per_warp

    # Total compute throughput
    mac_per_warp = L * 4 * (D // 4)
    total_mac = B * T * mac_per_warp

    return max(GPU_PARAMS["DRAM_BW_time"](total_t), total_mac / peak_throughput / 1000)


# Not ready for any FBGEMM. TODO: Fix this.
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
    if "rows_per_block" in y.keys():
        num_warps_per_sm = y["rows_per_block"] # Number of warps per sm
    else: # FBGEMM
        used_smem_bytes = int(SMEM_size / 3 * 2 / 16) * 16
        num_warps_per_sm = 32
        while num_warps_per_sm * 4 * 4 * 32 * div_round_up(y["embedding_dim"], 128) >= used_smem_bytes:
            num_warps_per_sm = num_warps_per_sm // 2
        assert num_warps_per_sm >= 1

    num_warps_simul = num_SM * num_warps_per_sm # Total number of warps simultaneously running on the device
    num_tables_simul = div_round_up(num_warps_simul, y["batch_size"]) # Number of tables simultaneously being accessed on the device
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

    total_l2_traffic = y["num_tables"] * (
                            y["batch_size"] * (
                                table_offsets_traffic + \
                                offsets_traffic + \
                                indices_l2_traffic) + \
                            hr * (
                                table_traffic * y["batch_size"] - \
                                avg_table_size
                            )
                        )
    total_dram_traffic = y["num_tables"] * (
                            y["batch_size"] * (
                                indices_dram_traffic + \
                                output_traffic) + \
                            (1 - hr) * (
                                table_traffic * y["batch_size"] - \
                                avg_table_size) + \
                            avg_table_size
                        )

    # Total compute throughput
    mac_per_warp = y["bag_size"] * 4 * (y["embedding_dim"] // 4)
    total_mac = y["batch_size"] * y["num_tables"] * mac_per_warp

    return max(total_dram_traffic / peak_DRAM_BW / 1000.0, total_l2_traffic / peak_L2_BW / 1000.0, total_mac / peak_throughput / 1000)


def concat_predictor(**kwargs):
    sum_size = kwargs["sum_size"]
    actual_peak_DRAM_BW = 0.79 * peak_DRAM_BW
    return 2 * (sum_size) * 4 / actual_peak_DRAM_BW / 1000


def memcpy_predictor(**kwargs):
    tensor_size = kwargs["tensor_size"]
    return tensor_size * 4 / peak_PCIe_BW / 1000


def mlp_predictor_tensor(x, op_type, backward=False):
    net = get_pretrained_net(op_type, backward)
    result = torch.exp(net(x, fbgemm=(op_type=="embedding_lookup"))).cpu().detach().view(-1)
    if result.shape == torch.Size((1,)):
        result = result.item()
    return result


def all_to_all_predictor(tensor_size):
    return predict_data_movement_time(
            tensor_size,
            ALL_TO_ALL_PARAMS["mul_factor"],
            ALL_TO_ALL_PARAMS["mem_ch"],
            ALL_TO_ALL_PARAMS["sigmoid_param"])


def all_reduce_predictor(tensor_size):
    return predict_data_movement_time(
            tensor_size,
            ALL_REDUCE_PARAMS["mul_factor"],
            ALL_REDUCE_PARAMS["mem_ch"],
            ALL_REDUCE_PARAMS["sigmoid_param"])


def collective_predictor(c, tensor_size):
    if c == "nccl:all_to_all":
        return all_to_all_predictor(tensor_size)
    else: # "nccl:all_reduce"
        return all_reduce_predictor(tensor_size)


def mlp_predictor_kwargs(op_type, backward=False, **kwargs):
    if op_type == "fully_connected":
        input_args = torch.tensor([np.log(kwargs[x]) for x in ["batch_size", "M", "N", "K"]], dtype=torch.float32)
    elif op_type == "embedding_lookup":
        input_args = [
            torch.tensor([
            ([
                np.log(kwargs["batch_size"]),
                np.log(E),
                np.log(L if L != 0.0 else 1e-3), # Avoid L = 0
                np.log(D),
            ] + rfs) for E, L, D, rfs in zip(
                kwargs["num_embeddings"],
                kwargs["pooling_factors"],
                kwargs["embedding_dims"],
                kwargs["reuse_factors"],
            )], dtype=torch.float32)
        ]
    elif op_type == "conv2d":
        input_args = torch.tensor([np.log(kwargs[x]) for x in ['batch_size', 'H', 'IC', 'OC']] + [kwargs[x] for x in ['stride', 'dilation', 'FH', 'FW', 'is_dw']], dtype=torch.float32)
    elif op_type == "conv1d":
        input_args = torch.tensor([np.log(kwargs[x]) for x in ['batch_size', 'L', 'IC', 'OC']] + [kwargs['groups']], dtype=torch.float32)
    elif op_type == "transpose" or op_type == "ln":
        input_args = torch.tensor([np.log(kwargs[x]) for x in ["batch_size", "M", "N"]], dtype=torch.float32)
    elif op_type == "bn":
        input_args = torch.tensor([np.log(kwargs[x]) for x in ["batch_size", "H", "OC"]], dtype=torch.float32)
    elif op_type == "dropout":
        input_args = torch.tensor([np.log(kwargs[x]) for x in ["batch_size", "M", "N"]] + [kwargs["p"]], dtype=torch.float32)
    else: # tril
        input_args = torch.tensor([np.log(kwargs[x]) for x in ["batch_size", "M", "N"]] + [kwargs["diag"]], dtype=torch.float32)
    return mlp_predictor_tensor(input_args, op_type, backward=backward)


def infer_concat():
    concat_data = pd.read_csv('{}/data/{}/kernel/concat_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    concat_data = preprocess(concat_data)
    concat_data = concat_data[concat_data["batch_size"] > 1]
    concat_data = concat_data[concat_data["kernel_name"].str.contains("Cat")]
    A_size = concat_data["batch_size"] * concat_data["M"] * concat_data["K"]
    B_size = concat_data["batch_size"] * concat_data["N"] * concat_data["K"]
    estimated_time = concat_predictor(sum_size=A_size+B_size)
    error = abs_err(estimated_time, concat_data['kernel_runtime'])
    print("Concat: GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0))
    return None, error


def infer_memcpy():
    memcpy_data = pd.read_csv('{}/data/{}/kernel/memcpy_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    memcpy_data = preprocess(memcpy_data)
    tensor_size = memcpy_data['batch_size'] * memcpy_data['M'] * memcpy_data['N']
    filter = (tensor_size * 4 / memcpy_data['kernel_runtime'] / 1e3 < peak_PCIe_BW) # Filter out samples with unreasonable timing
    memcpy_data = memcpy_data[filter]
    estimated_time = memcpy_predictor(tensor_size=tensor_size[filter])
    error = abs_err(estimated_time, memcpy_data['kernel_runtime'])
    print("Memcpy: GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0))
    return None, error


def infer_a2a():
    a2a_data = process_general_a2a_param_data(
        prefix="{}/3rdparty/param/train/comms/pt/bench_results".format(PM_HOME),
        num_gpus=GPU_COUNT,
    )
    size = a2a_data["size"]
    estimated_time = size.apply(all_to_all_predictor)
    error1 = abs_err(estimated_time, a2a_data['latency'])
    print("All-to-all (sum): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(gmae(error1) * 100.0, mape(error1) * 100.0, error1.std() * 100.0))

    a2a_data[['batch_size', "tables", "dim"]] = a2a_data['btd'].str.split(',', expand=True)
    tables = a2a_data["tables"].str.split('-', expand=True).astype(int)
    num_gpus = len(tables.iloc[0])
    a2a_data["batch_size"] = a2a_data["batch_size"].astype(int) // num_gpus # Divide evenly and send to each device
    a2a_data["dim"] = a2a_data["dim"].astype(int)
    sizes = tables.mul(a2a_data["batch_size"], axis="index").mul(a2a_data["dim"], axis="index") * 4 # float32

    size = sizes.apply(lambda x: get_max_message_size(x.tolist()), axis=1)
    estimated_time = (size).apply(all_to_all_predictor)
    error2 = abs_err(estimated_time, a2a_data['latency'])
    print("All-to-all (max of max): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(gmae(error2) * 100.0, mape(error2) * 100.0, error2.std() * 100.0))

    size = sizes.apply(lambda x: get_max_sum_message_size(x.tolist()), axis=1)
    estimated_time = (size).apply(all_to_all_predictor)
    error3 = abs_err(estimated_time, a2a_data['latency'])
    print("All-to-all (max of sum): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(gmae(error3) * 100.0, mape(error3) * 100.0, error3.std() * 100.0))

    error = error1 if gmae(error1) < gmae(error2) and gmae(error1) < gmae(error3) else (
        error2 if gmae(error2) < gmae(error1) and gmae(error2) < gmae(error3) else error3
    )

    return None, error


def infer_all_reduce():
    all_reduce_data = process_param_data(
        prefix="{}/3rdparty/param/train/comms/pt/bench_results".format(PM_HOME),
        collectives=["general_all_reduce"],
        num_gpus=GPU_COUNT,
    )["general_all_reduce"]
    estimated_time = all_reduce_data["size"].apply(all_reduce_predictor)
    error = abs_err(estimated_time, all_reduce_data['latency'])
    print("All-reduce: GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0))
    return None, error


def infer_elf(big=False, hit_rate_estimation=False, fbgemm=False):
    data = pd.read_csv('{}/data/{}/kernel/embedding_lookup_1_{}.csv'.format(PM_HOME, GPU_NAME, 'fbgemm' if fbgemm else 'shmem'), delimiter=',')
    data = preprocess(data)
    if fbgemm:
        data = preprocess_fbgemm(data)
        data.insert(0, 'kernel_name', ['dummy'] * len(data))
    else:
        data = data[data["kernel_name"].str.contains("batched_embedding")]

    if not hit_rate_estimation:
        if not big:
            time = data.apply(lambda x: embedding_forward_predictor_simple(**x[1:6]), axis=1)
            error = err(time, data['kernel_runtime'])
            print("ELF all sizes ({}simple): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_predictor_simple(**x[1:6]), axis=1)
            error = err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELF big sizes ({}simple): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )
    else:
        if not big:
            time = data.apply(lambda x: embedding_forward_predictor(**x[1:7]), axis=1)
            error = err(time, data['kernel_runtime'])
            print("ELF all sizes ({}hit rate estimation): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_predictor(**x[1:7]), axis=1)
            error = err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELF big sizes ({}hit rate estimation): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )

    return None, error


def infer_elb(big=False, hit_rate_estimation=False, fbgemm=False):
    data = pd.read_csv('{}/data/{}/kernel/embedding_lookup_0_sgd_{}.csv'.format(PM_HOME, GPU_NAME, 'fbgemm' if fbgemm else 'shmem'), delimiter=',')
    data = preprocess(data)
    if fbgemm:
        data = preprocess_fbgemm(data)
        data.insert(0, 'kernel_name', ['dummy'] * len(data))
    else:
        data = data[data["kernel_name"].str.contains("batched_embedding")]

    if not hit_rate_estimation:
        if not big:
            time = data.apply(lambda x: embedding_backward_sgd_predictor_simple(**x[1:6]), axis=1)
            error = err(time, data['kernel_runtime'])
            print("ELB all sizes ({}simple): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_predictor_simple(**x[1:6]), axis=1)
            error = err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELB big sizes ({}simple): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )
    else:
        if not big:
            time = data.apply(lambda x: embedding_backward_sgd_predictor(**x[1:7]), axis=1)
            error = err(time, data['kernel_runtime'])
            print("ELB all sizes ({}hit rate estimation): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_predictor(**x[1:7]), axis=1)
            error = err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELB big sizes ({}hit rate estimation): GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(
                "FBGEMM " if fbgemm else "",
                gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0)
            )

    return None, error


def infer_el(backward=False, big=False, hit_rate_estimation=False, fbgemm=False):
    if backward:
        _, error = infer_elb(big, hit_rate_estimation, fbgemm)
    else:
        _, error = infer_elf(big, hit_rate_estimation, fbgemm)
    return None, error


def infer_from_model(op_type, backward=False, **kwargs):
    _, _, _, x, y = get_train_test_data(op_type, backward, test_frac=1.0, **kwargs)
    suffix = "{}_{}".format(op_type, 1 if not backward else 0)
    with open("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix), "r") as f:
        best_config = json.load(f)
    estimated_time = mlp_predictor_tensor(x, op_type, backward=backward)
    real_time = torch.exp(y.cpu().detach()).view(-1)
    error = abs_err(estimated_time, real_time)
    print("{}: GMAE: {:.2f}%, MAPE: {:.2f}%, std: {:.2f}%".format(suffix, gmae(error) * 100.0, mape(error) * 100.0, error.std() * 100.0))
    return best_config, error


def infer(op_type, backward=False, **kwargs):
    if op_type == "concat":
        best_config, error = infer_concat()
    elif op_type == "memcpy":
        best_config, error = infer_memcpy()
    elif op_type == "all_to_all":
        best_config, error = infer_a2a()
    elif op_type == "all_reduce":
        best_config, error = infer_all_reduce()
    elif op_type == "embedding_lookup" and ("emb_use_mlp" not in kwargs.keys() or not kwargs["emb_use_mlp"]):
        big = kwargs["big"]
        hit_rate_estimation = kwargs["hit_rate_estimation"]
        is_fbgemm = kwargs["fbgemm"]
        best_config, error = infer_el(backward=backward, big=big, hit_rate_estimation=hit_rate_estimation, fbgemm=is_fbgemm)
    else: # embedding_lookup (MLP) / fully_connected / conv2d / conv1d / transpose / bn / tril
        best_config, error = infer_from_model(op_type, backward, **kwargs)
    return best_config, gmae(error)


def get_embedding_op_info(s):
    info = zlib.decompress(eval(s)).decode()
    Es = [int(x) for x in info.split('/')[0].split('-')]
    Ds = [int(x) for x in info.split('/')[1].split('-')]
    return Es, Ds


def get_kernel_time(op, ls=None, embedding_rfs=None):
    kernel_times = []
    if op.name == "aten::linear":
        # transpose will sometimes trigger a kernel call and sometimes not
        transpose = None
        for child in op.children:
            if child.name == "aten::t":
                transpose = child
            elif "addmm" in child.name:
                t = 0
                if transpose is not None:
                    M, N = transpose.input_shapes[0][0], transpose.input_shapes[0][1]
                    t += mlp_predictor_kwargs("transpose", backward=False, batch_size=1, M=M, N=N)
                M, K, N = child.input_shapes[1][0], child.input_shapes[1][1], child.input_shapes[2][1]
                t += mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=M, N=N, K=K)
                kernel_times.append(t)
                # print(child.name, M, K, N, child.input_shapes, t)
    if op.name == "aten::addmm":
        M, K, N = op.input_shapes[1][0], op.input_shapes[1][1], op.input_shapes[2][1]
        t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=M, N=N, K=K)
        kernel_times.append(t)
    elif "AddmmBackward" in op.name:
        addmm_op = op.get_child_by_name("AddmmBackward0")
        M, K, N = addmm_op.output_shapes[0][0], addmm_op.output_shapes[2][0], addmm_op.output_shapes[2][1]
        m1, k1, n1 = M, N, K
        m2, k2, n2 = N, M, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m1, N=n1, K=k1)
        kernel_times.append(t1)
        t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m2, N=n2, K=k2)
        kernel_times.append(t2)
        # t = t1 + t2
        # print(" -- ", addmm_op.name, M, K, N, addmm_op.input_shapes, t)
    elif op.name == "aten::matmul":
        for child in op.children:
            if child.name == "aten::reshape": # Equivalent to concat
                sa = np.prod(child.input_shapes[0][0])
                sb = np.prod(child.input_shapes[0][1] if child.input_shapes[0][1] else child.input_shapes[0][0])
                t = concat_predictor(sum_size=sa+sb)
                kernel_times.append(t)
            elif "aten::bmm" in child.name: # aten::bmm
                batch_size, M, K, N = child.input_shapes[0][0], child.input_shapes[0][1], child.input_shapes[0][2], child.input_shapes[1][2]
                t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=M, N=N, K=K)
                kernel_times.append(t)
                # print(child.name, batch_size, M, K, N, child.input_shapes, t)
    elif op.name == "aten::bmm":
        batch_size, M, K, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2], op.input_shapes[1][2]
        t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=M, N=N, K=K)
        kernel_times.append(t)
        # print(op.name, batch_size, M, K, N, op.input_shapes, t)
    elif "BmmBackward" in op.name:
        bmm_op = op.get_child_by_name("BmmBackward0")
        batch_size, M, N, K = bmm_op.input_shapes[0][0], bmm_op.input_shapes[0][1], bmm_op.input_shapes[0][2], bmm_op.output_shapes[0][2]
        m1, k1, n1 = K, M, N
        m2, k2, n2 = M, N, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=m1, N=n1, K=k1)
        t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=m2, N=n2, K=k2)
        kernel_times.append(t1)
        kernel_times.append(t2)
        t = t1 + t2
    elif op.name == "aten::conv2d":
        batch_size, IC, IH, _ = op.input_shapes[0]
        OC, FH, FW = op.input_shapes[1][0], op.input_shapes[1][2], op.input_shapes[1][3]
        stride, padding_h, dilation, is_dw = op.inputs[3][0], op.inputs[4][0], op.inputs[5][0], int(op.inputs[6] != 1)
        t = mlp_predictor_kwargs("conv2d", backward=False, batch_size=batch_size, H=IH+2*padding_h, IC=IC, OC=OC, stride=stride, dilation=dilation, FH=FH, FW=FW, is_dw=is_dw)
        kernel_times.append(t)
    elif op.name == "aten::conv1d":
        batch_size, ic_x_groups, L = op.input_shapes[0] # IC = 1 for now so ic_x_groups = groups
        oc_x_groups, _, _ = op.input_shapes[1]
        groups = ic_x_groups
        OC = oc_x_groups // groups
        t = mlp_predictor_kwargs("conv1d", backward=False, batch_size=batch_size, L=L, IC=1, OC=OC, groups=groups)
        kernel_times.append(t)
    elif op_name_in_list(op, ["CudnnConvolutionBackward", "ConvolutionBackward"]):
        # (Old) cudnn_convolution_backward(input, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask)
        # (New)       convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask)
        conv_bw_op = op.get_child_by_name(["aten::cudnn_convolution_backward", "aten::convolution_backward"])
        assert conv_bw_op is not None, "Cannot find the ATen convolution BW call"
        if conv_bw_op.name == "aten::convolution_backward":
            if len(conv_bw_op.inputs[4]) == 2: # 2D stride -> conv2d
                batch_size, IC, IH, _ = conv_bw_op.input_shapes[1] # [output, input, filter]
                OC, FH, FW = conv_bw_op.input_shapes[0][1], conv_bw_op.input_shapes[2][2], conv_bw_op.input_shapes[2][3]
                stride, padding_h, dilation, is_dw = conv_bw_op.inputs[4][0], conv_bw_op.inputs[5][0], conv_bw_op.inputs[6][0], int(conv_bw_op.inputs[9] != 1)
                t = mlp_predictor_kwargs("conv2d", backward=True, batch_size=batch_size, H=IH+2*padding_h, IC=IC, OC=OC, stride=stride, dilation=dilation, FH=FH, FW=FW, is_dw=is_dw)
            else: # 1D stride -> conv1d
                batch_size, ic_x_groups, L = conv_bw_op.input_shapes[1] # [output, input, filter]
                oc_x_groups = conv_bw_op.input_shapes[0][1]
                groups = ic_x_groups
                OC = oc_x_groups // groups
                t = mlp_predictor_kwargs("conv1d", backward=True, batch_size=batch_size, L=L, IC=1, OC=OC, groups=groups)
        else: # cudnn_convolution_backward
            if len(conv_bw_op.inputs[4]) == 2: # 2D stride -> conv2d
                batch_size, IC, IH, _ = conv_bw_op.input_shapes[0] # [input, output, filter]
                OC, FH, FW = conv_bw_op.input_shapes[1][1], conv_bw_op.input_shapes[2][2], conv_bw_op.input_shapes[2][3]
                stride, padding_h, dilation, is_dw = conv_bw_op.inputs[4][0], conv_bw_op.inputs[3][0], conv_bw_op.inputs[5][0], int(conv_bw_op.inputs[6] != 1)
                t = mlp_predictor_kwargs("conv2d", backward=True, batch_size=batch_size, H=IH+2*padding_h, IC=IC, OC=OC, stride=stride, dilation=dilation, FH=FH, FW=FW, is_dw=is_dw)
            else: # 1D stride -> conv1d
                batch_size, ic_x_groups, L = conv_bw_op.input_shapes[0] # [input, output, filter]
                oc_x_groups = conv_bw_op.input_shapes[1][1]
                groups = ic_x_groups
                OC = oc_x_groups // groups
                t = mlp_predictor_kwargs("conv1d", backward=True, batch_size=batch_size, L=L, IC=1, OC=OC, groups=groups)
        kernel_times.append(t)
    elif op.name == "LookupFunction":
        embedding_ops_stack.append(op)
        s = op.inputs[0]
        Es, Ds = get_embedding_op_info(s)
        T = op.input_shapes[1][0]
        B = int((op.input_shapes[3][0] - 1) / T)
        L = ls[0]
        D = Ds[0]
        rows_per_block = max(int(256 / D), 1)
        t = sum([embedding_forward_predictor(
                    batch_size=B,
                    num_embeddings=E,
                    num_tables=1,
                    bag_size=L,
                    embedding_dim=D,
                    rows_per_block=rows_per_block
            ) for E in Es]
        )
        kernel_times.append(t)
    elif "LookupFunctionBackward" in op.name:
        el_op = embedding_ops_stack.pop()
        s = el_op.inputs[0]
        Es, Ds = get_embedding_op_info(s)
        T = el_op.input_shapes[1][0]
        B = int((el_op.input_shapes[3][0] - 1) / T)
        L = ls[0]
        D = Ds[0]
        rows_per_block = max(int(256 / D), 1)
        t = sum([embedding_backward_sgd_predictor(
                    batch_size=B,
                    num_embeddings=E,
                    num_tables=1,
                    bag_size=L,
                    embedding_dim=D,
                    rows_per_block=rows_per_block
            ) for E in Es]
        )
        kernel_times.append(t)
    elif is_fbgemm_forward(op):
        embedding_ops_stack.append(op)
        batch_size = op.output_shapes[0][0]
        fbgemm_op = op.get_parent_by_name("embedding_lookup")
        s = fbgemm_op.inputs[0]
        Es, Ds = get_embedding_op_info(s)
        t = mlp_predictor_kwargs(
            op_type="embedding_lookup",
            backward=False,
            batch_size=batch_size,
            num_embeddings=Es,
            pooling_factors=ls,
            embedding_dims=Ds,
            reuse_factors=embedding_rfs,
        )
        kernel_times.append(t)
    elif "CppNode<SplitLookupFunction_" in op.name: # FBGEMM backward
        el_op = embedding_ops_stack.pop()
        batch_size = el_op.output_shapes[0][0]
        fbgemm_op = el_op.get_parent_by_name("embedding_lookup")
        s = fbgemm_op.inputs[0]
        Es, Ds = get_embedding_op_info(s)
        t = mlp_predictor_kwargs(
            op_type="embedding_lookup",
            backward=True,
            batch_size=batch_size,
            num_embeddings=Es,
            pooling_factors=ls,
            embedding_dims=Ds,
            reuse_factors=embedding_rfs,
        )
        kernel_times.append(t)
    elif op.name == "aten::batch_norm":
        if len(op.input_shapes[0]) == 4:
            batch_size, OC, H, _ = op.input_shapes[0] # BN 2D
        elif len(op.input_shapes[0]) == 3:
            batch_size, OC, H = op.input_shapes[0] # BN 1D with 3D input
        else:
            batch_size, OC = op.input_shapes[0] # BN 1D with 2D input
            H = 1
        t = mlp_predictor_kwargs("bn", backward=False, batch_size=batch_size, H=H, OC=OC)
        kernel_times.append(t)
    elif "CudnnBatchNormBackward" in op.name:
        bn_op = op.get_child_by_name("CudnnBatchNormBackward0")
        if len(bn_op.output_shapes[0]) == 4:
            batch_size, OC, H, _ = bn_op.output_shapes[0] # BN 2D
        elif len(bn_op.output_shapes[0]) == 3:
            batch_size, OC, H = bn_op.output_shapes[0] # BN 1D with 3D input
        else:
            batch_size, OC = bn_op.output_shapes[0] # BN 1D with 2D input
            H = 1
        t = mlp_predictor_kwargs("bn", backward=True, batch_size=batch_size, H=H, OC=OC)
        kernel_times.append(t)
    elif op.name == "aten::layer_norm":
        batch_size, M, N = op.input_shapes[0]
        t = mlp_predictor_kwargs("ln", backward=False, batch_size=batch_size, M=M, N=N)
        kernel_times.append(t)
    elif "LayerNormBackward" in op.name:
        batch_size, M, N = op.input_shapes[0]
        t = mlp_predictor_kwargs("ln", backward=True, batch_size=batch_size, M=M, N=N)
        kernel_times.append(t)
    elif op.name == "aten::index":
        batch_size, M, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2]
        total_output_element = op.input_shapes[1][1][0]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = mlp_predictor_kwargs("tril", backward=False, batch_size=batch_size, M=M, N=N, diag=diag)
        kernel_times.append(t)
    elif "IndexBackward" in op.name: # See all kernels as a whole
        tril_op = op.get_child_by_name("IndexBackward0")
        batch_size, M, N = tril_op.output_shapes[0][0], tril_op.output_shapes[0][1], tril_op.output_shapes[0][2]
        total_output_element = tril_op.input_shapes[0][1]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = mlp_predictor_kwargs("tril", backward=True, batch_size=batch_size, M=M, N=N, diag=diag)
        kernel_times.append(t)
    elif is_all2all_parent(op):
        # Get the collective op
        collective_op = None
        def dfs(n):
            nonlocal collective_op
            if n.name in COMMS:
                collective_op = n
            for c in n.children:
                dfs(c)
        dfs(op)
        x = collective_op.get_parent_by_name("Backward")
        if x: # a2a backward
            # Backward compatible to old Pytorch (np.prod(collective_op.parent.output_shapes[0]) * 4 should be enough for v2.0)
            tensor_size = np.prod(x.get_child_by_name("record").output_shapes[0]) * 4

            # Copies
            for y in x.get_child_by_name("contiguous").parent.children:
                if "contiguous" in y.name:
                    s = np.prod(y.input_shapes[0])
                    t = 2 * s * 4 / peak_DRAM_BW / 1000
                    kernel_times.append(t)
        else: # Forward
            tensor_size = np.prod(collective_op.input_shapes[0]) * 4

        # Concat
        cat_op = op.get_child_by_name("cat")
        sum_size = sum([np.prod(s) for s in cat_op.input_shapes[0]])
        t = concat_predictor(sum_size=sum_size)
        kernel_times.append(t)

        # Size treatment (max of max) for a2a
        tensor_size = tensor_size // ext_dist.my_size
        ipTensor = torch.tensor([tensor_size], dtype=torch.float) # On the CPU
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)] # On the CPU
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        tensor_size = get_max_message_size([x.item() for x in opTensorList])
        t = collective_predictor(collective_op.name, tensor_size=tensor_size)
        kernel_times.append(t)
    elif is_allreduce_parent(op):
        # Mul
        mul_op = op.get_child_by_name("mul")
        s = np.prod(mul_op.input_shapes[0] if mul_op.input_shapes else mul_op.children[0].input_shapes[0])
        t = max(s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000)
        kernel_times.append(t)

        # Get the collective op
        collective_op = None
        def dfs(n):
            nonlocal collective_op
            if n.name in COMMS:
                collective_op = n
            for c in n.children:
                dfs(c)
        dfs(op)
        tensor_size = np.prod(collective_op.input_shapes[0]) * 4
        t = collective_predictor(collective_op.name, tensor_size=tensor_size)
        kernel_times.append(t)
    elif op.name == "aten::cat":
        sum_size = sum([np.prod(s) for s in op.input_shapes[0]])
        t = concat_predictor(sum_size=sum_size)
        kernel_times.append(t)
    elif op.name == "aten::to":
        t = 0
        if "dtype_layout" in op.op_schema: # Some aten::to actually doesn't move data H2D or D2H.
            s = np.prod(op.input_shapes[0])
            t = memcpy_predictor(tensor_size=s)
        kernel_times.append(t)
    # Minor ops staring from here: ----
    elif op.name == "aten::t":
        kernel_times.append(0) # T is handled under addmm
    elif op_name_in_list(op, ["aten::relu", "ReluBackward"]):
        s = np.prod(op.input_shapes[0] if op.input_shapes else op.children[0].input_shapes[0])
        t = max(s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # One read one write
        kernel_times.append(t)
    elif op.name == "aten::sigmoid":
        s = np.prod(op.input_shapes[0])
        t = max(4 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # 4 flops per sigmoid (exp as one); one read one write
        kernel_times.append(t)
    elif "SigmoidBackward" in op.name:
        s = np.prod(op.children[0].input_shapes[0])
        t = max(2 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # 2 flops per sigmoid backward (f' = f*(1-f)); one read one write
        kernel_times.append(t)
    elif op.name == "aten::binary_cross_entropy":
        s = np.prod(op.input_shapes[0])
        t = max(7 * s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # 7 flops per bce (log as one); two read one write
        kernel_times.append(t)
    elif "BinaryCrossEntropyBackward" in op.name:
        s = np.prod(op.children[0].input_shapes[0])
        t = max(4 * s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # 4 flops per bce backward (E' = (y-t)/y/(1-y)); two read one write
        kernel_times.append(t)
    elif op.name == "aten::mse_loss":
        s = np.prod(op.input_shapes[0])
        t = max(3 * s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # 3 flops per mse; two read one write
        kernel_times.append(t)
    elif op.name == "aten::gelu":
        s = np.prod(op.input_shapes[0])
        t = max(175 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # Roughly 175 flops per element; one read one write
        kernel_times.append(t)
    elif "GeluBackward" in op.name:
        s = np.prod(op.children[0].input_shapes[0])
        t = max(255 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # Roughly 255 flops per element; one read one write
        kernel_times.append(t)
    elif op.name == "aten::softmax":
        s = np.prod(op.input_shapes[0])
        t = max(174 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # Roughly 174 flops per element; one read one write
        kernel_times.append(t)
    elif op.name == "aten::dropout":
        if len(op.input_shapes[0]) == 4:
            batch_size, M1, M2, N = op.input_shapes[0]
            M = M1 * M2
        elif len(op.input_shapes[0]) == 3:
            batch_size, M, N = op.input_shapes[0]
        else:
            batch_size, N = op.input_shapes[0]
            M = 1
        p = op.inputs[-2]
        t = mlp_predictor_kwargs("dropout", backward=False, batch_size=batch_size, M=M, N=N, p=p)
        kernel_times.append(t)
    elif op_name_in_list(op, [
            "aten::add", "aten::add_", "aten::__and__", "aten::sub", "AddBackward", \
            "aten::mul", "MulBackward", "aten::div", "DivBackward", "MseLossBackward" \
        ]):
        s = np.prod(op.input_shapes[0] if op.input_shapes else op.children[0].input_shapes[0])
        t = max(s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # 1 flops per mse backward (M' = y-t); two read one write
        kernel_times.append(t)
    elif op.name == "aten::sum":
        s = np.prod(op.input_shapes[0])
        t = max(s / peak_throughput / 1000, s * 4 / peak_DRAM_BW / 1000) # One reads
        kernel_times.append(t)
    elif op_name_in_list(op, ["aten::ones_like", "aten::zero_", "ViewBackward"]):
        s = np.prod(op.children[0].input_shapes[0])
        t = 2 * s * 4 / peak_DRAM_BW / 1000 # One read one write
        kernel_times.append(t)
    elif op.name == "aten::copy_":
        s = np.prod(op.input_shapes[0])
        t = 2 * s * 4 / peak_DRAM_BW / 1000 # One read one write
        kernel_times.append(t)
    elif op_name_in_list(op, ["aten::tanh", "aten::pow"]):
        s = np.prod(op.input_shapes[0])
        t = max(180 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # Roughly 180 flops per element; one read one write
        kernel_times.append(t)
    elif "TanhBackward" in op.name:
        s = np.prod(op.children[0].input_shapes[0])
        t = max(260 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # Roughly 180 flops per element; one read one write
        kernel_times.append(t)
    elif "PowBackward" in op.name:
        s = np.prod(op.children[0].input_shapes[0])

        # "aten::pow"
        t = max(180 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # Roughly 180 flops per element; one read one write
        kernel_times.append(t)

        # "aten::mul" * 2 & "aten::add_"
        t = max(s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # 1 flops per element; two read one write
        kernel_times.extend([t, t, t])
    elif op.name == "aten::max_pool2d" or op.name == "aten::avg_pool2d":
        s = np.prod(op.output_shapes[0])
        t = max((np.prod(op.inputs[1]) - 1) * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # One reads one write
        kernel_times.append(t)
    elif op_name_in_list(op, ["MaxPool2DWithIndicesBackward", "AvgPool2DBackward"]):
        pool2d_bw_op = op.get_child_by_name("_pool2d_")
        assert pool2d_bw_op is not None, "Cannot find the ATen pool2d BW call"
        s = np.prod(pool2d_bw_op.output_shapes[0])
        t = max((2 * np.prod(pool2d_bw_op.inputs[2]) - 1) * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # One reads one write
        kernel_times.append(t)
    # Grad ops starting from here: ----
    elif op_name_in_list(op, ["torch::autograd::AccumulateGrad"]): # Mismatch: empty + clone, while in trace it's add
        s = np.prod(op.children[0].input_shapes[0])
        t = 2 * s * 4 / peak_DRAM_BW / 1000 # One read one write
        t = 0 if s > 5e6 else t # Tmp solution to avoid dense add for embedding table lookup
        kernel_times.append(t)
    elif "All2All_ReqBackward" in op.name:
        sub_op = op.get_child_by_name("All2All_ReqBackward")
        # Quite a few aten::contiguous
        for output_shape in sub_op.output_shapes:
            s = np.prod(output_shape)
            t = 2 * s * 4 / peak_DRAM_BW / 1000 # One read one write
            kernel_times.append(t)
    else:
        kernel_times.append(0)
    # print(op.name, op.input_shapes, kernel_times)
    return kernel_times


def get_e2e_time_for_each_iter(graph, overheads, ls=None, embedding_rfs=None, module_marker="## ", debug=False):
    nodes = graph.get_nodes(clean=True, to_be_pruned=SKIP)
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])

    cpu_time = 0
    gpu_time = {
        "active": 0, # GPU active time, no matter whichever stream is active
        "baseline": { # Active time of each stream, no overheads -> max of the two is the baseline
            COMPUTE_STREAM: 0,
            COMMUNICATION_STREAM: 0,
        },
        "prediction": { # Delicate sync and everything
            COMPUTE_STREAM: 0,
            COMMUNICATION_STREAM: 0,
        }
    }
    stream = COMPUTE_STREAM
    last_comm_op = None
    prev_node = None
    gpu_all_streams_front = 0
    unoverlapped_comm = 0

    forward_found = False
    for _, node in sorted_nodes:
        if module_marker in node.name:
            forward_found = True
        if not forward_found: # Start inference at the beginning of Forward
            continue
        if node.is_op() and not to_skip(node):
            shapes = str(DUMMY_SHAPES)

            # Sync all GPU streams in 3 cases (similar to dependency_test):
            # 1. the current node is dependent on the current comm op
            # 2. meet wait ops
            # 3. meet another collective
            if (last_comm_op and depends_on_collective_output(node, last_comm_op[1])) or \
                    is_wait_collective(node) or \
                    has_comm_collective(node):
                last_comm_op = None
                gpu_time["prediction"][COMPUTE_STREAM] = gpu_all_streams_front
                gpu_time["prediction"][COMMUNICATION_STREAM] = gpu_all_streams_front
                if debug:
                    print("  Communication sync: ", cpu_time, gpu_time)

            cpu_time += overheads["t1"][0] # T1: between two nodes
            if debug:
                print(" ", node.name, "--------")
                print("    t1: {:.2f}".format(overheads["t1"][0]))
            if node.name in overheads["launches"].keys(): # Has kernel calls
                cpu_time += overheads["t2"][node.name][shapes][0] # T2: before the first kernel call
                if debug:
                    print("    t2: {:.2f}".format(overheads["t2"][node.name][shapes][0]))
                launches = overheads["launches"][node.name]
                if to_consider(node) or has_comm_collective(node):
                    t = [tt + GPU_KERNEL_OVERHEAD for tt in get_kernel_time(node, ls=ls, embedding_rfs=embedding_rfs)] # Get kernel time and (arguably) compensate with the overheads

                    stream = infer_multi_stream(node)
                    for idx, l in enumerate(launches):
                        t4 = overheads["t4"][l][0] # Kernel launches
                        t5 = overheads["t5"][node.name][shapes][0] # Avg overhead between

                        # Non-collective kernels under collective ops are on COMPUTE_STREAM
                        if (is_all2all_parent(node) or is_allreduce_parent(node)) and idx != len(launches) - 1:
                            stream = COMPUTE_STREAM
                        else:
                            stream = infer_multi_stream(node)
                        
                        # Contribution of CPU overheads on GPU idle time
                        gpu_time["prediction"][stream] = max(gpu_time["prediction"][stream] + 1, cpu_time + t4) # Where the kernel starts: either launch right after last kernel, or at the end of the kernel launch

                        # Kernel launches like cudaStreamSynchronize do not have actual kernel calls
                        if idx < len(t):
                            # Unoverlapped communication for eg_comm
                            if stream == COMMUNICATION_STREAM:
                                unoverlapped_comm += t[idx] - max(gpu_all_streams_front - gpu_time["prediction"][stream], 0) # Kernel time minus time overlapped by another stream
                            elif last_comm_op is not None: # A comm kernel is running
                                # Either fully overlapped or the compute stream becomes the front
                                unoverlapped_comm -= t[idx] - max(gpu_time["prediction"][COMPUTE_STREAM] + t[idx] - gpu_time["prediction"][COMMUNICATION_STREAM], 0)

                            # GPU active time:
                            # cases of max:   (stream front)   [     ]|  or  [     ]|
                            #                 (current stream)   |[-]          |[-------]
                            # cases of min:   current stream is the stream front
                            gpu_time["active"] += min(max(gpu_time["prediction"][stream] + t[idx] - gpu_all_streams_front, 0), t[idx])

                            # Current stream
                            gpu_time["prediction"][stream] += t[idx]
                            gpu_time["baseline"][stream] += t[idx]
                            if debug:
                                print("    kernel: {:.2f}".format(t[idx]))

                        if is_memcpy(node, strict=True) and len(node.children) == 0:
                            cpu_time += CPU_EVENT_OVERHEAD # Some aten::to doesn't have children
                        else:
                            cpu_time += t4

                        # Num of T5 is num of T4 - 1
                        if idx < len(launches) - 1:
                            cpu_time += t5
                        if debug:
                            print("    t4: {:.2f}".format(t4))
                            print("    t5: {:.2f}".format(t5))

                    if ext_dist.my_size > 1 and has_comm_collective(node): # Only sync after a multi-GPU collective
                        ipTensor = torch.tensor([gpu_time["prediction"][stream]], dtype=torch.float) # On the CPU
                        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)] # On the CPU
                        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
                        front = max([x.item() for x in opTensorList])
                        gpu_time["active"] += front - gpu_time["prediction"][stream]
                        gpu_time["prediction"][stream] = front
                    # print(node.name, t)
                else:
                    # Only consider CPU time then: op_cpu_time = T2 + (T4 sum) + (T5 sum) + T3
                    cpu_time += np.sum([overheads["t4"][x][0] for x in launches]) # T4
                    if node.name in overheads["t5"]:
                        cpu_time += overheads["t5"][node.name][shapes][0] * (len(launches) - 1) # T5
                    if debug:
                        print("    t4: {:.2f}".format(np.sum([overheads["t4"][x][0] for x in launches])))
                        if node.name in overheads["t5"]:
                            print("    t5: {:.2f}".format(overheads["t5"][node.name][shapes][0] * (len(launches) - 1)))
                        else:
                            print("Warning: {} is skipped for not found in the overheads.".format(node.name))
                cpu_time += overheads["t3"][node.name][shapes][0] # T3: after the last kernel call
                gpu_all_streams_front = max(gpu_time["prediction"].values()) # Track the GPU front among all streams
                if debug:
                    print("    t3: {:.2f}".format(overheads["t3"][node.name][shapes][0]))
                # print(node.name, ["{:.2f}".format(tt) for tt in t], gpu_time["active"])
            else: # aten::view, aten::ones, aten::zeros, aten::empty, etc
                if node.name in overheads["t5"]:
                    cpu_time += overheads["t5"][node.name][shapes][0] # Ops that have no kernel calls only have T5 overheads (total CPU overheads)
                if debug:
                    if node.name in overheads["t5"]:
                        print("    t5: {:.2f}".format(overheads["t5"][node.name][shapes][0]))
                    else:
                        print("Warning: {} is skipped for not found in the overheads.".format(node.name))

            # Sync after memcpy
            if is_memcpy(node, strict=True):
                cpu_time = max(cpu_time, gpu_time["prediction"][COMPUTE_STREAM])

            # Collective and dependency
            if has_comm_collective(node) and not is_wait_collective(node):
                if is_all2all_parent(node): # Has a2a call
                    tmp = node.get_child_by_name(["aten::new_empty", "aten::empty"])
                    last_comm_op = ("all_to_all", tmp.outputs[0], tmp.output_shapes[0], node)
                elif is_allreduce_parent(node): # Has all_reduce call
                    tmp = node.get_child_by_name("nccl:all_reduce")
                    last_comm_op = ("all_reduce", tmp.inputs[0], tmp.input_shapes[0], node)
                # Some cases that nccl:all_to_all/nccl:all_reduce comes with trailing record_param_comms
                elif prev_node and (is_all2all(prev_node) or is_allreduce(prev_node)):
                    last_comm_op = ("all_to_all", node.outputs[0], node.output_shapes[0], node) \
                                if is_all2all(prev_node) \
                                else ("all_reduce", node.outputs[0], node.output_shapes[0], node)

            prev_node = node
            if debug:
                tmp_total_time = max(max(gpu_time["prediction"].values()), cpu_time)
                print("      ", node.name, tmp_total_time, cpu_time, gpu_time, node.input_shapes)

    # Prediction and baseline
    e2e_time = max(max(gpu_time["prediction"].values()), cpu_time)
    gpu_active_time = gpu_time["active"]
    baseline_e2e_time = max(gpu_time["baseline"].values()) # No sync so no CPU time for baseline
    
    # eg_comm
    eg_comm = unoverlapped_comm / e2e_time

    # Sync all the processes
    if ext_dist.my_size > 1:
        # Sync e2e_time and take the max. Gather to rank 0 is enough though but all_gather is convenient
        ipTensor = torch.tensor([e2e_time], dtype=torch.float)
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)]
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        e2e_time = max([x.item() for x in opTensorList])

        # Sync GPU active time
        ipTensor = torch.tensor([gpu_active_time], dtype=torch.float)
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)]
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        gpu_active_time = max([x.item() for x in opTensorList])

        # Sync baseline_total_time and take the max
        ipTensor = torch.tensor([baseline_e2e_time], dtype=torch.float)
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)]
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        baseline_e2e_time = max([x.item() for x in opTensorList])

        # Sync eg_comm
        ipTensor = torch.tensor([eg_comm], dtype=torch.float) # On the CPU
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)] # On the CPU
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        eg_comm = min([x.item() for x in opTensorList])

    return {
        "e2e_time": e2e_time,
        "gpu_active_time": gpu_active_time,
        "baseline_e2e_time": baseline_e2e_time,
        "eg_comm": eg_comm,
    }


# Infer E2E time from an execution graph and an overhead file
def get_e2e_time(graph, overheads, iters, ls_file=None, embedding_rfs_file=None, module_marker="## ", debug=False):
    all_Ls = [None] * iters
    if ls_file is not None:
        with open(ls_file, 'r') as f:
            all_Ls = f.readlines()
    all_rfs = [None] * iters
    if embedding_rfs_file is not None:
        with open(embedding_rfs_file, 'r') as f:
            all_rfs = f.readlines()

    e2e_time_list = []
    gpu_active_time_list = []
    baseline_e2e_time_list = []
    eg_comm_list = []
    for idx in range(iters):
        Ls = [float(x) for x in all_Ls[idx].split('-')] if all_Ls[idx] else None
        embedding_rfs = [[float(x) for x in x_.split('-')] for x_ in all_rfs[idx].split('_')] if all_rfs[idx] else None
        results = get_e2e_time_for_each_iter(
            graph, overheads,
            ls=Ls,
            embedding_rfs=embedding_rfs,
            module_marker=module_marker,
            debug=debug
        )
        e2e_time_list.append(results["e2e_time"])
        gpu_active_time_list.append(results["gpu_active_time"])
        baseline_e2e_time_list.append(results["baseline_e2e_time"])
        eg_comm_list.append(results["eg_comm"])
    
    return {
        "e2e_time": np.mean(e2e_time_list),
        "gpu_active_time": np.mean(gpu_active_time_list),
        "baseline_e2e_time": np.mean(baseline_e2e_time_list),
        "eg_comm": np.mean(eg_comm_list),
    }
