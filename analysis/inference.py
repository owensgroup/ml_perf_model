import torch
import pandas as pd
import numpy as np
import json
from .utils import preprocessing, abs_err, div_round_up, gmae, get_pretrained_net, get_data, PM_HOME, GPU_NAME, GPU_PARAMS, GPU_EVENT_OVERHEAD
from .exec_graph_utils import NodeType

peak_throughput = GPU_PARAMS["peak_throughput"]
peak_PCIe_BW = GPU_PARAMS["peak_PCIe_BW"]
peak_DRAM_BW = GPU_PARAMS["peak_DRAM_BW"]
peak_L2_BW = GPU_PARAMS["peak_L2_BW"]
peak_SMEM_BW = GPU_PARAMS["peak_SMEM_BW"]
num_SM = GPU_PARAMS["num_SM"]
L2_size = GPU_PARAMS["L2_size"]


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


def concat_predictor(**kwargs):
    A_size = kwargs["A_size"]
    B_size = kwargs["B_size"]
    actual_peak_DRAM_BW = 0.79 * peak_DRAM_BW
    return 2 * (A_size + B_size) * 4 / actual_peak_DRAM_BW / 1000


def memcpy_predictor(**kwargs):
    tensor_size = kwargs["tensor_size"]
    return tensor_size * 4 / peak_PCIe_BW / 1000


def mlp_predictor_tensor(x, op_type, backward=False):
    net = get_pretrained_net(op_type, backward)
    return torch.exp(net(x.cpu()).detach().view(-1)).item()


def mlp_predictor_kwargs(op_type, backward=False, **kwargs):
    if op_type == "fully_connected":
        n_feature = 4
        input_size = [np.log(kwargs[x]) for x in ["batch_size", "M", "N", "K"]]
    elif op_type == "transpose":
        n_feature = 3
        input_size = [np.log(kwargs[x]) for x in ["batch_size", "M", "N"]]
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
    estimated_time = concat_predictor(A_size=A_size, B_size=B_size)
    error = abs_err(estimated_time, concat_data['kernel_runtime'])
    print("Concat: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    return None, error


def infer_memcpy():
    memcpy_data = pd.read_csv('{}/data/{}/kernel/memcpy_1.csv'.format(PM_HOME, GPU_NAME), delimiter=',')
    memcpy_data = preprocessing(memcpy_data)
    tensor_size = memcpy_data['batch_size'] * memcpy_data['M'] * memcpy_data['N']
    estimated_time = memcpy_predictor(tensor_size=tensor_size)
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
    estimated_time = mlp_predictor_tensor(x, op_type, backward=backward, )
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


def get_kernel_time(op, op_lists):
    kernel_times = []
    if op.name == "aten::linear":
        for child in op.children:
            if "aten::t" in child.name:
                M, N = child.input_shapes[0][0], child.input_shapes[0][1]
                t = mlp_predictor_kwargs("transpose", backward=False, batch_size=1, M=M, N=N)
                kernel_times.append(t)
            else: # addmm
                op_lists["addmm"].append(child)
                M, K, N = child.input_shapes[1][0], child.input_shapes[1][1], child.input_shapes[2][1]
                t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=M, N=N, K=K)
                kernel_times.append(t)
    elif op.name == "AddmmBackward":
        addmm_op = op_lists["addmm"].pop()
        M, K, N = addmm_op.input_shapes[1][0], addmm_op.input_shapes[1][1], addmm_op.input_shapes[2][1]
        m1, k1, n1 = M, N, K
        m2, k2, n2 = N, M, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m1, N=n1, K=k1)
        kernel_times.append(t1)
        t2 = 0
        if M != N:
            t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=1, M=m2, N=n2, K=k2)
            kernel_times.append(t2)
        t = t1 + t2
    elif op.name == "aten::bmm":
        op_lists["bmm"].append(op)
        batch_size, M, K, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2], op.input_shapes[1][2]
        t = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=M, N=N, K=K)
        kernel_times.append(t)
    elif op.name == "BmmBackward0":
        bmm_op = op_lists["bmm"].pop()
        batch_size, M, K, N = bmm_op.input_shapes[0][0], bmm_op.input_shapes[0][1], bmm_op.input_shapes[0][2], bmm_op.input_shapes[1][2]
        m1, k1, n1 = K, M, N
        m2, k2, n2 = M, N, K
        t1 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=m1, N=n1, K=k1)
        t2 = mlp_predictor_kwargs("fully_connected", backward=False, batch_size=batch_size, M=m2, N=n2, K=k2)
        kernel_times.append(t1)
        kernel_times.append(t2)
        t = t1 + t2
    elif op.name == "LookupFunction":
        op_lists["el"].append(op)
        T = op.input_shapes[1][0]
        D = op.input_shapes[0][1]
        B = int((op.input_shapes[3][0] - 1) / T)
        E = int(op.input_shapes[0][0] / T)
        L = int(op.input_shapes[2][0] / B / T)
        rows_per_block = max(int(256 / D), 1)
        t = embedding_forward_predictor(batch_size=B, num_embeddings=E, num_tables=T, bag_size=L, embedding_dim=D, rows_per_block=rows_per_block)
        kernel_times.append(t)
    elif op.name == "LookupFunctionBackward":
        el_op = op_lists["el"].pop()
        T = el_op.input_shapes[1][0]
        D = el_op.input_shapes[0][1]
        B = int((el_op.input_shapes[3][0] - 1) / T)
        E = int(el_op.input_shapes[0][0] / T)
        L = int(el_op.input_shapes[2][0] / B / T)
        rows_per_block = max(int(256 / D), 1)
        t = embedding_backward_sgd_predictor(batch_size=B, num_embeddings=E, num_tables=T, bag_size=L, embedding_dim=D, rows_per_block=rows_per_block)
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
    elif op.name == "IndexBackward": # See all kernels as a whole
        tril_op = op_lists["tril"].pop()
        batch_size, M, N = tril_op.input_shapes[0][0], tril_op.input_shapes[0][1], tril_op.input_shapes[0][2]
        total_output_element = tril_op.input_shapes[1][1][0]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = mlp_predictor_kwargs("tril", backward=True, batch_size=batch_size, M=M, N=N, diag=diag)
        kernel_times.append(t)
    # Minor ops staring from here: ----
    elif op.name == "aten::relu":
        op_lists["relu"].append(op)
        s = np.prod(op.input_shapes[0])
        t = max(s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # One read one write
        kernel_times.append(t)
    elif op.name == "ReluBackward0":
        relu_op = op_lists["relu"].pop()
        s = np.prod(relu_op.input_shapes[0])
        t = max(s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # One read one write
        kernel_times.append(t)
    elif op.name == "aten::sigmoid":
        op_lists["sigmoid"].append(op)
        s = np.prod(op.input_shapes[0])
        t = max(4 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # 4 flops per sigmoid (exp as one); one read one write
        kernel_times.append(t)
    elif op.name == "SigmoidBackward":
        sigmoid_op = op_lists["sigmoid"].pop()
        s = np.prod(sigmoid_op.input_shapes[0])
        t = max(2 * s / peak_throughput / 1000, 2 * s * 4 / peak_DRAM_BW / 1000) # 2 flops per sigmoid backward (f' = f*(1-f)); one read one write
        kernel_times.append(t)
    elif op.name == "aten::t":
        kernel_times.append(0) # T is handled under addmm
    elif op.name == "aten::add":
        s = np.prod(op.input_shapes[0])
        t = max(s / peak_throughput / 1000, 3 * s * 4 / peak_DRAM_BW / 1000) # Two reads one write
        kernel_times.append(t)
    elif op.name == "aten::cat":
        sa = np.prod(op.input_shapes[0][0])
        sb = np.prod(op.input_shapes[0][1])
        t = concat_predictor(A_size=sa, B_size=sb)
        kernel_times.append(t)
    elif op.name == "aten::sum":
        s = np.prod(op.input_shapes[0])
        t = max(s / peak_throughput / 1000, s * 4 / peak_DRAM_BW / 1000) # One reads
        kernel_times.append(t)
    elif op.name == "aten::to":
        s = np.prod(op.input_shapes[0])
        t = memcpy_predictor(tensor_size=s)
        kernel_times.append(t)
    # print(op.name, op.input_shapes, t)
    return kernel_times


# Infer E2E time from an execution graph and an overhead file
def get_e2e_time(graph, overheads):
    nodes = graph.get_nodes(clean=True)
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
    op_lists = {
        "addmm": [],
        "bmm": [],
        "el": [],
        "relu": [],
        "sigmoid": [],
        "tril": [],
    }
    cpu_time = 0
    gpu_time = 0
    gpu_active_time = 0

    consider = ["aten::linear", "AddmmBackward", "aten::bmm", "BmmBackward0", \
                "LookupFunction", "LookupFunctionBackward", \
                "aten::index", "IndexBackward", \
                "aten::relu", "ReluBackward0", \
                "aten::sigmoid", "SigmoidBackward", \
                "aten::add", "aten::cat", "aten::sum", "aten::to"]

    for _, op in sorted_nodes:
        is_op = (op.type == NodeType.OPERATOR and op.parent.type != NodeType.OPERATOR)
        if is_op:
            cpu_time += overheads["t1"][0] # T1: between two ops
            if op.name in overheads["launches"].keys(): # Has kernel calls
                cpu_time += overheads["t2"][op.name][0] # T2: before the first kernel call
                launches = overheads["launches"][op.name]
                if op.name in consider:
                    t = get_kernel_time(op, op_lists) # Get kernel time
                    gpu_active_time += np.sum(t)

                    for idx, l in enumerate(launches):
                        t4 = overheads["t4"][l][0] # Kernel launches
                        t5 = overheads["t5"][op.name][0] # Avg overhead between

                        # Contribution of CPU overheads on GPU idle time
                        gpu_time = max(gpu_time + 1, cpu_time + t4 - GPU_EVENT_OVERHEAD) # Where the kernel starts: either launch right after last kernel, or at the end of the kernel launch

                        # Kernel launches like cudaStreamSynchronize do not have actual kernel calls
                        if idx < len(t):
                            gpu_time += t[idx]

                        if "aten::to" == op.name and idx != 0:
                            cpu_time += t[0] # The time cudaStreamSynchronize in aten::to is tied to the memcpy kernel
                        else:
                            cpu_time += t4

                        # Num of T5 is num of T4 - 1
                        if idx < len(launches) - 1:
                            cpu_time += t5
                else:
                    # Only consider CPU time then: op_cpu_time = T2 + (T4 sum) + (T5 sum) + T3
                    cpu_time += overheads["t5"][op.name][0] * (len(launches) - 1) # T5
                    cpu_time += np.sum([overheads["t4"][x][0] for x in launches]) # T4
                cpu_time += overheads["t3"][op.name][0] # T3: after the last kernel call
            else: # aten::view, aten::ones, aten::zeros, aten::empty, etc
                cpu_time += overheads["t5"][op.name][0] # Ops that have no kernel calls only have T5 overheads (total CPU overheads)
            # print(op.name, op.input_shapes, gpu_active_time)
    total_time = max(gpu_time, cpu_time)

    return total_time, gpu_active_time