import argparse
import os
import json

import numpy as np
import torch

from analysis.utils import PM_HOME, GPU_COUNT

MEMORY_SCALE_FACTOR = 0.9

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate DLRM tasks")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-gpu-memory', type=int, default=16*1024*1024*1024) # V100, in bytes
    parser.add_argument('--data-dir', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag/2021")
    parser.add_argument('--out-dir', type=str, default="{}/benchmark".format(PM_HOME))
    parser.add_argument('--per-gpu-table-limit', type=int, default=8)
    parser.add_argument('--num-tasks', type=int, default=30)

    args = parser.parse_args()
    np.random.seed(args.seed)
    min_table_count = int(args.per_gpu_table_limit * GPU_COUNT * 0.9)
    max_table_count = int(args.per_gpu_table_limit * GPU_COUNT * 1.1)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.data_dir, "common_configs.json"), "r") as f:
        table_configs = json.load(f)["tables"]
    T = len(table_configs)
    Es = [t["num_embeddings"] for t in table_configs]
    Ds = [t["embedding_dim"] for t in table_configs]
    table_indices = [i for i in range(T)]
    np.random.shuffle(table_indices)
    dataset_suffix = args.data_dir.split('/')[-1]

    out_path = os.path.join(args.out_dir, "tasks_{}_{}.txt".format(dataset_suffix, GPU_COUNT))
    # Generate tasks
    i = 0
    with open(out_path, "w") as f:
        while i < args.num_tasks:
            all_device_table_count = np.random.randint(min_table_count, max_table_count)
            table_ids = np.random.choice(table_indices, size=all_device_table_count, replace=False)
            Es = [table_configs[t]["num_embeddings"] for t in table_ids]
            Ds = [table_configs[t]["embedding_dim"] for t in table_ids]
            # Skip if OOM
            table_mem_sum = sum([E * D for E, D in zip(Es, Ds)]) * 4
            if table_mem_sum > args.per_gpu_memory * MEMORY_SCALE_FACTOR:
                continue
            
            table_ids = "-".join(list(map(str, table_ids)))
            f.write(table_ids + "\n")
            i += 1
