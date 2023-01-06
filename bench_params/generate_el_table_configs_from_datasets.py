import argparse
import glob
import gzip
import json
import os

import numpy as np
import torch


def gen_table_configs(args):
    filename = gzip.open(args.dataset_path) if args.dataset_path.endswith('gz') else args.dataset_path
    indices, offsets, lengths = torch.load(filename)
    num_tables, batch_size = lengths.shape

    indices = indices.cuda()
    offsets = offsets.cuda()

    # Split the tables
    lS_pooling_factors = list(map(int, lengths.float().mean(dim=1).tolist()))
    lS_rows, lS_bin_counts = [], []
    for t in range(num_tables):
        start = t * batch_size
        end = (t + 1) * batch_size + 1
        table_offsets = offsets[start:end]
        table_indices = indices[table_offsets[0]:table_offsets[-1]]
        table_offsets = table_offsets - offsets[start]

        row = table_indices.max().int().item() + 1 if len(table_indices) > 0 else 1
        row = max(100, row)

        _, indices_counts = torch.unique(table_indices, return_counts=True)
        unique_counts, counts_counts = torch.unique(indices_counts, return_counts=True)
        total_counts = counts_counts.sum().item()

        if total_counts == 0:
            bin_counts = [0.0 for _ in range(17)]
        else:
            start, end = 0, 1
            bin_counts = []
            for i in range(16):
                bin_counts.append(counts_counts[(unique_counts > start) & (unique_counts <= end)].sum().item())
                start = end
                end *= 2
            bin_counts.append(counts_counts[unique_counts > start].sum().item())
            bin_counts = [x/total_counts for x in bin_counts]

        lS_rows.append(row)
        lS_bin_counts.append(bin_counts)
    
    T = len(lS_rows) # number of tables
    lS_dims = np.random.choice(args.dim_range, T)

    table_configs = {}
    table_configs["tables"] = []
    for i in range(T):
        table_config = {}
        table_config["index"] = i
        table_config["num_embeddings"] = int(lS_rows[i])
        table_config["embedding_dim"] = int(lS_dims[i])
        table_config["pooling_factor"] = int(lS_pooling_factors[i])
        for j, _ in enumerate(lS_bin_counts[i]):
            table_config["bin_"+str(j)] = lS_bin_counts[i][j]
        table_configs["tables"].append(table_config)

    with open(args.table_config_path, "w") as f:
        json.dump(table_configs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate all-to-all params.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset-path', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag/2021")
    parser.add_argument('--dim-range', type=str, default="32,64,128,256")
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.dim_range = list(map(int, args.dim_range.split(",")))

    if os.path.isdir:
        datasets = [f for f_ in [glob.glob(args.dataset_path + e) for e in ('/*.pt', '/*.pt.gz')] for f in f_]
    else:
        datasets = [args.dataset_path]

    for d in datasets:
        print("Path:", d)
        args.dataset_path = d

        np.random.seed(args.seed) # Guarantee datasets with the same number of tables use the same dims
        splitext_filename = os.path.splitext(args.dataset_path)[0]
        if args.dataset_path.endswith('gz'):
            splitext_filename = os.path.splitext(splitext_filename)[0]

        args.table_config_path = splitext_filename + '_configs.json'
        if not os.path.exists(args.table_config_path):
            gen_table_configs(args)
