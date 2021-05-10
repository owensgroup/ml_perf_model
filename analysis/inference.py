import torch
import pandas as pd
import json
from utils import preprocessing, abs_err, div_round_up, gmae, get_pretrained_net, get_data, PM_HOME, GPU_NAME, GPU_PARAMS

peak_throughput = GPU_PARAMS["peak_throughput"]
peak_PCIe_BW = GPU_PARAMS["peak_PCIe_BW"]
peak_DRAM_BW = GPU_PARAMS["peak_DRAM_BW"]
peak_L2_BW = GPU_PARAMS["peak_L2_BW"]
peak_SMEM_BW = GPU_PARAMS["peak_SMEM_BW"]
num_SM = GPU_PARAMS["num_SM"]
L2_size = GPU_PARAMS["L2_size"]

def infer_concat():
    concat_data = pd.read_csv('{}/data/{}/concat_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    concat_data = preprocessing(concat_data)
    concat_data = concat_data[concat_data["batch_size"] > 1]
    A_size = concat_data["batch_size"] * concat_data["M"] * concat_data["K"]
    B_size = concat_data["batch_size"] * concat_data["N"] * concat_data["K"]
    actual_peak_DRAM_BW = 0.79 * peak_DRAM_BW
    concat_traffic = 2 * (A_size + B_size) * 4
    estimated_time = concat_traffic / actual_peak_DRAM_BW / 1000
    error = abs_err(estimated_time, concat_data['kernel_runtime'])
    print("Concat: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    return None, error


def infer_memcpy():
    memcpy_data = pd.read_csv('{}/data/{}/memcpy_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    memcpy_data = preprocessing(memcpy_data)
    memcpy_traffic = memcpy_data['batch_size'] * memcpy_data['M'] * memcpy_data['N'] * 4
    estimated_time = memcpy_traffic / peak_PCIe_BW / 1000
    error = abs_err(estimated_time, memcpy_data['kernel_runtime'])
    print("Memcpy: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    return None, error


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
    # num_total_warps = y["batch_size"] * y["num_tables"] # Total warp number of the kernel
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
    # num_total_warps = y["batch_size"] * y["num_tables"] # Total warp number of the kernel
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


def infer_elf(big=False, hit_rate_estimation=False):
    data = pd.read_csv('{}/data/{}/embedding_lookup_1_shmem.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    data = preprocessing(data)
    data = data[data["kernel_name"].str.contains("batched_embedding")]
    data = data[data['batch_size'] > 1]

    if not hit_rate_estimation:
        if not big:
            time = data.apply(lambda x: embedding_forward_simple(**x[1:6]), axis=1)
            error = abs_err(time, data['kernel_runtime'])
            print("ELF all sizes (simple): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_forward_simple(**x[1:6]), axis=1)
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
    data = pd.read_csv('{}/data/{}/embedding_lookup_0_sgd_shmem.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    data = preprocessing(data)
    data = data[data["kernel_name"].str.contains("batched_embedding")]
    data = data[data['batch_size'] > 1]

    if not hit_rate_estimation:
        if not big:
            time = data.apply(lambda x: embedding_backward_sgd_simple(**x[1:6]), axis=1)
            error = abs_err(time, data['kernel_runtime'])
            print("ELB all sizes (simple): GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
        else:
            time = data[data['num_embeddings'] >= 100000].apply(lambda x: embedding_backward_sgd_simple(**x[1:6]), axis=1)
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
    net = get_pretrained_net(op_type, backward)
    _, x, y = get_data(op_type, backward)

    suffix = "{}_{}".format(op_type, 1 if not backward else 0)
    with open("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix), "r") as f:
        best_config = json.load(f)
    estimated_time = torch.exp(net(x.cpu()).detach().view(-1))
    error = abs_err(estimated_time, torch.exp(y.cpu().detach()).view(-1))
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
    else: # fully_connected / transpose / tril
        best_config, error = infer_from_model(op_type, backward)
    return best_config, gmae(error)
