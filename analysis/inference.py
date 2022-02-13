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
import json
from .utils import *
import analysis.extend_distributed as ext_dist

peak_throughput = GPU_PARAMS["peak_throughput"]
peak_PCIe_BW = GPU_PARAMS["peak_PCIe_BW"]
peak_DRAM_BW = GPU_PARAMS["peak_DRAM_BW"]
peak_L2_BW = GPU_PARAMS["peak_L2_BW"]
peak_SMEM_BW = GPU_PARAMS["peak_SMEM_BW"]
num_SM = GPU_PARAMS["num_SM"]
L2_size = GPU_PARAMS["L2_size"]


COMPUTE_STREAM = 0
MEMORY_STREAM = 1
ADDITIONAL = 2


def embedding_forward_predictor_simple(**kwargs):
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
    # num_total_warps = y["batch_size"] * y["num_tables"] # Total warp number of the kernel
    num_warps_per_sm = y["rows_per_block"] # Number of warps per sm
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
    table_traffic = y["bag_size"] * (div_round_up(y["embedding_dim"] * 4, 32) * 32)
    output_traffic = (div_round_up(y["embedding_dim"] * 4, 32) * 32)

    total_l2_traffic = y["num_tables"] * (
                            y["batch_size"] * (
                                table_offsets_traffic + \
                                offsets_traffic + \
                                indices_l2_traffic) + \
                            hr * (
                                table_traffic * y["batch_size"] - \
                                avg_table_size)
                        )
    total_dram_traffic = y["num_tables"] * (
                            y["batch_size"] * (
                                indices_dram_traffic + 
                                output_traffic) + \
                            (1 - hr) * (
                                table_traffic * y["batch_size"] - \
                                avg_table_size) + \
                            avg_table_size
                        )

    return max(total_dram_traffic / peak_DRAM_BW / 1000.0, total_l2_traffic / peak_L2_BW / 1000.0)


def embedding_backward_sgd_predictor_simple(**kwargs):
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
    # num_total_warps = y["batch_size"] * y["num_tables"] # Total warp number of the kernel
    num_warps_per_sm = y["rows_per_block"] # Number of warps per sm
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
    result = torch.exp(net(x.cpu()).detach().view(-1))
    if result.shape == torch.Size((1,)):
        result = result.item()
    return result


def predict_collective_time(size, mul_factor, incr_p, sats_p, max_bw, overhead, sigmoid_param):
    log_size = np.log2(size)
    if log_size <= incr_p:
        return overhead
    elif log_size >= sats_p:
        return size / max_bw * mul_factor / 1e3 + overhead
    else:
        bw = get_sigmoid_bw(log_size, sigmoid_param)
        return size / bw * mul_factor / 1e3


# 4-V100 DGX-1
def all_to_all_predictor(**kwargs):
    # Hardcoded for now
    incr_p = 13 # 2^13 bytes
    sats_p = 27 # 2^27 bytes
    max_bw = 56.7134 # GB/s
    overhead = 85.6 # us
    sigmoid_param = (5.97878216, 13.33976161, 0.22551425, -3.99731681)

    f = MUL_FACTOR_FUNCS["all_to_allv"]
    return predict_collective_time(kwargs["tensor_size"] * 4, f(kwargs["num_gpus"]), incr_p, sats_p, max_bw, overhead, sigmoid_param)


# 4-V100 DGX-1
def all_reduce_predictor(**kwargs):
    # Hardcoded for now
    incr_p = 12 # 2^12 bytes
    sats_p = 26 # 2^26 bytes
    max_bw = 75.6704 # GB/s
    overhead = 84.2 # us
    sigmoid_param = (6.04564109, 12.70417428, 0.2242347, -3.94164857)

    f = MUL_FACTOR_FUNCS["all_reduce"]
    return predict_collective_time(kwargs["tensor_size"] * 4, f(kwargs["num_gpus"]), incr_p, sats_p, max_bw, overhead, sigmoid_param)


def collective_predictor(c, **kwargs):
    if c == "nccl:all_to_all":
        return all_to_all_predictor(**kwargs)
    else: # "nccl:all_reduce"
        return all_reduce_predictor(**kwargs)


def mlp_predictor_kwargs(op_type, backward=False, **kwargs):
    if op_type == "fully_connected":
        n_feature = 4
        input_size = [np.log(kwargs[x]) for x in ["batch_size", "M", "N", "K"]]
    elif op_type == "conv2d":
        n_feature = 9
        input_size = [np.log(kwargs[x]) for x in ['batch_size', 'H', 'IC', 'OC']] + [kwargs[x] for x in ['stride', 'dilation', 'FH', 'FW', 'is_dw']]
    elif op_type == "conv1d":
        n_feature = 5
        input_size = [np.log(kwargs[x]) for x in ['batch_size', 'L', 'IC', 'OC']] + [kwargs['groups']]
    elif op_type == "transpose":
        n_feature = 3
        input_size = [np.log(kwargs[x]) for x in ["batch_size", "M", "N"]]
    elif op_type == "bn":
        n_feature = 3
        input_size = [np.log(kwargs[x]) for x in ["batch_size", "H", "OC"]]
    else: # tril
        n_feature = 4
        input_size = [np.log(kwargs[x]) for x in ["batch_size", "M", "N"]] + [kwargs["diag"]]
    assert len(input_size) == n_feature
    return mlp_predictor_tensor(torch.tensor(input_size, dtype=torch.float32), op_type, backward=backward)


def infer_concat():
    concat_data = pd.read_csv('{}/data/{}/kernel/concat_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    concat_data = preprocessing(concat_data)
    concat_data = concat_data[concat_data["batch_size"] > 1]
    A_size = concat_data["batch_size"] * concat_data["M"] * concat_data["K"]
    B_size = concat_data["batch_size"] * concat_data["N"] * concat_data["K"]
    estimated_time = concat_predictor(sum_size=A_size+B_size)
    error = abs_err(estimated_time, concat_data['kernel_runtime'])
    print("Concat: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    return None, error


def infer_memcpy():
    memcpy_data = pd.read_csv('{}/data/{}/kernel/memcpy_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    memcpy_data = preprocessing(memcpy_data)
    tensor_size = memcpy_data['batch_size'] * memcpy_data['M'] * memcpy_data['N']
    filter = (tensor_size * 4 / memcpy_data['kernel_runtime'] / 1e3 < peak_PCIe_BW) # Filter out samples with unreasonable timing
    memcpy_data = memcpy_data[filter]
    estimated_time = memcpy_predictor(tensor_size=tensor_size[filter])
    error = abs_err(estimated_time, memcpy_data['kernel_runtime'])
    print("Memcpy: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    return None, error


def infer_elf(big=False, hit_rate_estimation=False):
    data = pd.read_csv('{}/data/{}/kernel/embedding_lookup_1_shmem.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    data = preprocessing(data)
    data = data[data["kernel_name"].str.contains("batched_embedding")]
    data = data[data['batch_size'] > 1]

    if not hit_rate_estimation:
        if not big:
            time = data.apply(lambda x: embedding_forward_predictor_simple(**x[1:6]), axis=1)
            error = abs_err(time, data['kernel_runtime'])
            print("ELF all sizes (simple): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_predictor_simple(**x[1:6]), axis=1)
            error = abs_err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELF big sizes (simple): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    else:
        if not big:
            time = data.apply(lambda x: embedding_forward_predictor(**x[1:7]), axis=1)
            error = abs_err(time, data['kernel_runtime'])
            print("ELF all sizes (hit rate estimation): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_predictor(**x[1:7]), axis=1)
            error = abs_err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELF big sizes (hit rate estimation): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))

    return None, error


def infer_elb(big=False, hit_rate_estimation=False):
    data = pd.read_csv('{}/data/{}/kernel/embedding_lookup_0_sgd_shmem.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    data = preprocessing(data)
    data = data[data["kernel_name"].str.contains("batched_embedding")]
    data = data[data['batch_size'] > 1]

    if not hit_rate_estimation:
        if not big:
            time = data.apply(lambda x: embedding_backward_sgd_predictor_simple(**x[1:6]), axis=1)
            error = abs_err(time, data['kernel_runtime'])
            print("ELB all sizes (simple): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_predictor_simple(**x[1:6]), axis=1)
            error = abs_err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELB big sizes (simple): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    else:
        if not big:
            time = data.apply(lambda x: embedding_backward_sgd_predictor(**x[1:7]), axis=1)
            error = abs_err(time, data['kernel_runtime'])
            print("ELB all sizes (hit rate estimation): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_predictor(**x[1:7]), axis=1)
            error = abs_err(time, data[data['num_embeddings'] >= 100000]['kernel_runtime'])
            print("ELB big sizes (hit rate estimation): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))

    return None, error


def infer_el(backward=False, big=False, hit_rate_estimation=False):
    if backward:
        _, error = infer_elb(big, hit_rate_estimation)
    else:
        _, error = infer_elf(big, hit_rate_estimation)
    return None, error


def infer_from_model(op_type, backward=False):
    _, x, y = get_data(op_type, backward)
    suffix = "{}_{}".format(op_type, 1 if not backward else 0)
    with open("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix), "r") as f:
        best_config = json.load(f)
    estimated_time = mlp_predictor_tensor(x, op_type, backward=backward)
    real_time = torch.exp(y.cpu().detach()).view(-1)
    error = abs_err(estimated_time, real_time)
    print("{}: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(suffix, gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    return best_config, error


def infer(op_type, backward=False, **kwargs):
    if op_type == "concat":
        best_config, error = infer_concat()
    elif op_type == "memcpy":
        best_config, error = infer_memcpy()
    elif op_type == "embedding_lookup":
        big = kwargs["big"] 
        hit_rate_estimation = kwargs["hit_rate_estimation"]
        best_config, error = infer_el(backward=backward, big=big, hit_rate_estimation=hit_rate_estimation)
    else: # fully_connected / conv2d / conv1d / transpose / bn / tril
        best_config, error = infer_from_model(op_type, backward)
    return best_config, gmae(error)


def get_kernel_time(op, op_lists):
    kernel_times = []
    if op.name == "aten::linear":
        # transpose will sometimes trigger a kernel call and sometimes not
        transpose = None
        for child in op.children:
            if child.name == "aten::t":
                transpose = child
            elif "addmm" in child.name:
                if transpose is not None:
                    M, N = transpose.input_shapes[0][0], transpose.input_shapes[0][1]
                    t = mlp_predictor_kwargs("transpose", backward=False, batch_size=1, M=M, N=N)
                    kernel_times.append(t)
                op_lists["addmm"].append(child)
                M, K, N = child.input_shapes[1][0], child.input_shapes[1][1], child.input_shapes[2][1]
                t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=M, N=N, K=K)
                kernel_times.append(t)
                # print(child.name, M, K, N, child.input_shapes, t)
            elif child.name == "aten::matmul":
                op_lists["mm"].append(child)
                M, K, N = child.input_shapes[0][0] * child.input_shapes[0][1], child.input_shapes[0][2], child.input_shapes[1][1] if len(child.input_shapes[1]) > 1 else 1
                t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=M, N=N, K=K)
                kernel_times.append(t)
                # print(child.name, M, K, N, child.input_shapes, t)
    elif "AddmmBackward" in op.name:
        addmm_op = op_lists["addmm"].pop()
        M, K, N = addmm_op.input_shapes[1][0], addmm_op.input_shapes[1][1], addmm_op.input_shapes[2][1]
        m1, k1, n1 = M, N, K
        m2, k2, n2 = N, M, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m1, N=n1, K=k1)
        kernel_times.append(t1)
        t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m2, N=n2, K=k2)
        kernel_times.append(t2)
        t = t1 + t2
        # print(" -- ", addmm_op.name, M, K, N, addmm_op.input_shapes, t)
    elif "MmBackward" in op.name:
        mm_op = op_lists["mm"].pop()
        M, K, N = mm_op.input_shapes[0][0] * mm_op.input_shapes[0][1], mm_op.input_shapes[0][2], mm_op.input_shapes[1][1] if len(mm_op.input_shapes[1]) > 1 else 1
        m1, k1, n1 = M, N, K
        m2, k2, n2 = N, M, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m1, N=n1, K=k1)
        kernel_times.append(t1)
        t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m2, N=n2, K=k2)
        kernel_times.append(t2)
        t = t1 + t2
        # print(" -- ", mm_op.name, M, K, N, mm_op.input_shapes, t)
    elif op.name == "aten::matmul":
        for child in op.children:
            if child.name == "aten::reshape": # Equivalent to concat
                sa = np.prod(child.input_shapes[0][0])
                sb = np.prod(child.input_shapes[0][1])
                t = concat_predictor(sum_size=sa+sb)
                kernel_times.append(t)
            elif "aten::bmm" in child.name: # aten::bmm
                op_lists["bmm"].append(child)
                batch_size, M, K, N = child.input_shapes[0][0], child.input_shapes[0][1], child.input_shapes[0][2], child.input_shapes[1][2]
                t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=M, N=N, K=K)
                kernel_times.append(t)
                # print(child.name, batch_size, M, K, N, child.input_shapes, t)
    elif op.name == "aten::bmm":
        op_lists["bmm"].append(op)
        batch_size, M, K, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2], op.input_shapes[1][2]
        t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=M, N=N, K=K)
        kernel_times.append(t)
        # print(op.name, batch_size, M, K, N, op.input_shapes, t)
    elif "BmmBackward" in op.name:
        bmm_op = op_lists["bmm"].pop()
        batch_size, M, K, N = bmm_op.input_shapes[0][0], bmm_op.input_shapes[0][1], bmm_op.input_shapes[0][2], bmm_op.input_shapes[1][2]
        m1, k1, n1 = K, M, N
        m2, k2, n2 = M, N, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=m1, N=n1, K=k1)
        t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=m2, N=n2, K=k2)
        kernel_times.append(t1)
        kernel_times.append(t2)
        t = t1 + t2
        # print(" -- ", bmm_op.name, batch_size, M, K, N, bmm_op.input_shapes, t)
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
        conv_bw_op = op.get_child_by_name("aten::convolution_backward")
        assert conv_bw_op is not None, "Cannot find the ATen convolution BW call"
        if len(conv_bw_op.inputs[4]) == 2: # 2D stride -> conv2d
            batch_size, IC, IH, _ = conv_bw_op.input_shapes[1] # [output, input, filter]
            OC, FH, FW = conv_bw_op.input_shapes[0][1], conv_bw_op.input_shapes[2][2], conv_bw_op.input_shapes[2][3]
            stride, padding_h, dilation, is_dw = conv_bw_op.inputs[4][0], conv_bw_op.inputs[5][0], conv_bw_op.inputs[6][0], int(conv_bw_op.inputs[9] != 1)
            t = mlp_predictor_kwargs("conv2d", backward=True, batch_size=batch_size, H=IH+2*padding_h, IC=IC, OC=OC, stride=stride, dilation=dilation, FH=FH, FW=FW, is_dw=is_dw)
            kernel_times.append(t)
        else: # 1D stride -> conv1d
            batch_size, ic_x_groups, L = conv_bw_op.input_shapes[1] # [output, input, filter]
            oc_x_groups = conv_bw_op.input_shapes[0][1]
            groups = ic_x_groups
            OC = oc_x_groups // groups
            t = mlp_predictor_kwargs("conv1d", backward=True, batch_size=batch_size, L=L, IC=1, OC=OC, groups=groups)
            kernel_times.append(t)
    elif op.name == "LookupFunction":
        op_lists["el"].append(op)
        T = op.input_shapes[1][0]
        if op.parent.inputs: # Check if table sizes are passed as argument to the parent record function
            Es = [int(e) for e in op.parent.inputs[0].split('-')]
            assert len(Es) == T
        else:
            Es = [int(op.input_shapes[0][0] / T)] * T # Use the average
        D = op.input_shapes[0][1]
        B = int((op.input_shapes[3][0] - 1) / T)
        L = int(op.input_shapes[2][0] / B / T)
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
        el_op = op_lists["el"].pop()
        T = el_op.input_shapes[1][0]
        if el_op.parent.inputs: # Check if table sizes are passed as argument to the parent record function
            Es = [int(e) for e in el_op.parent.inputs[0].split('-')]
            assert len(Es) == T
        else:
            Es = [int(el_op.input_shapes[0][0] / T)] * T # Use the average
        D = el_op.input_shapes[0][1]
        B = int((el_op.input_shapes[3][0] - 1) / T)
        L = int(el_op.input_shapes[2][0] / B / T)
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
    elif op.name == "aten::batch_norm":
        op_lists["bn"].append(op)
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
        bn_op = op_lists["bn"].pop()
        batch_size, OC, H, _ = bn_op.input_shapes[0]
        t = mlp_predictor_kwargs("bn", backward=True, batch_size=batch_size, H=H, OC=OC)
        kernel_times.append(t)
    elif op.name == "aten::index":
        op_lists["tril"].append(op)
        batch_size, M, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2]
        total_output_element = op.input_shapes[1][1][0]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = mlp_predictor_kwargs("tril", backward=False, batch_size=batch_size, M=M, N=N, diag=diag)
        kernel_times.append(t)
    elif "IndexBackward" in op.name: # See all kernels as a whole
        tril_op = op_lists["tril"].pop()
        batch_size, M, N = tril_op.input_shapes[0][0], tril_op.input_shapes[0][1], tril_op.input_shapes[0][2]
        total_output_element = tril_op.input_shapes[1][1][0]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = mlp_predictor_kwargs("tril", backward=True, batch_size=batch_size, M=M, N=N, diag=diag)
        kernel_times.append(t)
    elif op.name == "record_param_comms" or "All2All" in op.name:
        collective_op = None
        # Get the collective op
        def dfs(n):
            nonlocal collective_op
            if n.name in COMMS:
                collective_op = n
            for c in n.children:
                dfs(c)
        dfs(op)
        tensor_size = np.prod(collective_op.input_shapes[0])
        t = collective_predictor(collective_op.name, tensor_size=tensor_size, num_gpus=4) # TODO: Replace 4 with world size
        kernel_times.append(t)
    elif op.name == "aten::cat":
        sum_size = sum([np.prod(s) for s in op.input_shapes[0]])
        t = concat_predictor(sum_size=sum_size)
        kernel_times.append(t)
    elif op.name == "aten::to":
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
    elif op_name_in_list(op, ["aten::add", "aten::add_", "aten::__and__", "aten::sub", "aten::mul", "MulBackward", "MseLossBackward"]):
        s = np.prod(op.input_shapes[0] if op.input_shapes else op.children[0].input_shapes[0])
        t = max(s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # 1 flops per mse backward (M' = y-t); two read one write
        kernel_times.append(t)
    elif op.name == "aten::sum":
        s = np.prod(op.input_shapes[0])
        t = max(s / peak_throughput / 1000, s * 4 / peak_DRAM_BW / 1000) # One reads
        kernel_times.append(t)
    elif op.name == "aten::ones_like":
        s = np.prod(op.children[0].input_shapes[0])
        t = 2 * s * 4 / peak_DRAM_BW / 1000 # One read one write
        kernel_times.append(t)
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
    elif op.name == "Optimizer.step#SGD.step":
        for child in op.children:
            s = np.prod(child.input_shapes[0])
            if s > 5e6:
                continue # Tmp solution to avoid dense add for embedding table lookup
            t = max(s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # Two reads one write
            kernel_times.append(t)
    elif op.name == "Optimizer.zero_grad#SGD.zero_grad": # Mismatch: empty, while in trace it's zero
        # for child in op.children:
        #     s = np.prod(child.input_shapes[0])
        #     if s > 5e6:
        #         continue # Tmp solution to avoid dense add for embedding table lookup
        #     t = memcpy_predictor(tensor_size=s)
        #     kernel_times.append(t)
        kernel_times.append(0)
    # print(op.name, op.input_shapes, kernel_times)
    return kernel_times


# Simple heuristic to infer if an op will schedule a kernel in a new stream
def infer_multi_stream(op):
    if is_collective(op):
        return MEMORY_STREAM
    return COMPUTE_STREAM


# Infer E2E time from an execution graph and an overhead file
def get_e2e_time(graph, overheads, module_marker, debug=False):
    nodes = graph.get_nodes(clean=True)
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])

    # TODO: Remove this as all input shapes can be obtained from children nodes
    op_lists = {
        "addmm": [],
        "bmm": [],
        "bn": [],
        "el": [],
        "mm": [],
        "tril": [],
    }
    cpu_time = 0
    gpu_time = {
        "active": 0,
        COMPUTE_STREAM: 0,
        MEMORY_STREAM: 0
    }
    stream = COMPUTE_STREAM
    gpu_all_streams_front = 0

    forward_found = False
    for _, node in sorted_nodes:
        if module_marker in node.name:
            forward_found = True
        if not forward_found:
            continue
        if node.is_op():
            if (to_skip(node)):
                continue
            stream = infer_multi_stream(node)
            cpu_time += overheads["t1"][0] # T1: between two nodes
            if debug:
                print(" ", node.name, "--------")
                print("    t1: {:.2f}".format(overheads["t1"][0]))
            if node.name in overheads["launches"].keys(): # Has kernel calls
                cpu_time += overheads["t2"][node.name][0] # T2: before the first kernel call
                if debug:
                    print("    t2: {:.2f}".format(overheads["t2"][node.name][0]))
                launches = overheads["launches"][node.name]
                # print(node.name, to_consider(node))
                if to_consider(node) or is_collective(node):
                    t = [tt + GPU_EVENT_OVERHEAD for tt in get_kernel_time(node, op_lists)] # Get kernel time and (arguably) compensate with the overheads

                    for idx, l in enumerate(launches):
                        t4 = overheads["t4"][l][0] # Kernel launches
                        t5 = overheads["t5"][node.name][0] # Avg overhead between

                        # Contribution of CPU overheads on GPU idle time
                        gpu_time[stream] = max(gpu_time[stream] + 1, cpu_time + t4) # Where the kernel starts: either launch right after last kernel, or at the end of the kernel launch

                        # Kernel launches like cudaStreamSynchronize do not have actual kernel calls
                        if idx < len(t):
                            # GPU active time:
                            # cases of max:   (stream front)   [     ]|  or  [     ]|
                            #                 (current stream)   |[-]          |[-------]
                            # cases of min:   current stream is the stream front
                            gpu_time["active"] += min(max(gpu_time[stream] + t[idx] - gpu_all_streams_front, 0), t[idx])

                            # Current stream
                            gpu_time[stream] += t[idx]
                            if ext_dist.my_size > 1 and is_collective(node): # Only sync after a multi-GPU collective
                                ipTensor = torch.tensor([gpu_time[stream]], dtype=torch.float) # On the CPU
                                opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)] # On the CPU
                                ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
                                gpu_time[stream] = max([x.item() for x in opTensorList])
                            if debug:
                                print("    kernel: {:.2f}".format(t[idx]))

                        if "aten::to" == node.name and len(node.children) == 0:
                            cpu_time += CPU_EVENT_OVERHEAD # Some aten::to doesn't have children
                        else:
                            cpu_time += t4

                        # Num of T5 is num of T4 - 1
                        if idx < len(launches) - 1:
                            cpu_time += t5
                        if debug:
                            print("    t4: {:.2f}".format(t4))
                            print("    t5: {:.2f}".format(t5))

                    # print(node.name, t)
                else:
                    # Only consider CPU time then: op_cpu_time = T2 + (T4 sum) + (T5 sum) + T3
                    cpu_time += np.sum([overheads["t4"][x][0] for x in launches]) # T4
                    cpu_time += overheads["t5"][node.name][0] * (len(launches) - 1) # T5
                    if debug:
                        print("    t4: {:.2f}".format(np.sum([overheads["t4"][x][0] for x in launches])))
                        print("    t5: {:.2f}".format(overheads["t5"][node.name][0] * (len(launches) - 1)))
                cpu_time += overheads["t3"][node.name][0] # T3: after the last kernel call
                gpu_all_streams_front = max(gpu_time.values()) # Track the GPU front among all streams
                if debug:
                    print("    t3: {:.2f}".format(overheads["t3"][node.name][0]))
                # print(node.name, ["{:.2f}".format(tt) for tt in t], gpu_time["active"])
            else: # aten::view, aten::ones, aten::zeros, aten::empty, etc
                cpu_time += overheads["t5"][node.name][0] # Ops that have no kernel calls only have T5 overheads (total CPU overheads)
                if debug:
                    print("    t5: {:.2f}".format(overheads["t5"][node.name][0]))
            if debug:
                print(node.name, cpu_time, gpu_time, node.input_shapes)

    # It's safe because gpu_time["active"] won't be the maximum
    total_time = max(max(gpu_time.values()), cpu_time)

    # Sync all the processes
    if ext_dist.my_size > 1:
        # Sync total_time and take the max. Gather to rank 0 is enough though but all_gather is convenient
        ipTensor = torch.tensor([gpu_time[stream]], dtype=torch.float)
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)]
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        total_time = max([x.item() for x in opTensorList])

        # Sync GPU active time
        ipTensor = torch.tensor([gpu_time["active"]], dtype=torch.float)
        opTensorList = [torch.empty([1]) for _ in range(ext_dist.my_size)]
        ext_dist.all_gather(opTensorList=opTensorList, ipTensor=ipTensor)
        gpu_time["active"] = max([x.item() for x in opTensorList])

    return total_time, gpu_time["active"]
