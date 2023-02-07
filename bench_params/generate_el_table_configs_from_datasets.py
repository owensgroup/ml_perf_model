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

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch


def gen_config_per_table(args, dataset_path):
    indices, offsets, lengths = torch.load(dataset_path)
    num_tables, batch_size = lengths.shape

    indices = indices.cuda()
    offsets = offsets.cuda()

    # Split the tables
    lS_pooling_factor_mean = lengths.float().mean(dim=1).tolist()
    lS_pooling_factor_std = lengths.float().std(dim=1).tolist()
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
        table_config["embedding_dim"] = int(lS_dims[i])
        table_config["pooling_factor"] = lS_pooling_factor_mean[i]
        table_config["pooling_factor_std"] = lS_pooling_factor_std[i]
        for j, _ in enumerate(lS_bin_counts[i]):
            table_config["bin_"+str(j)] = lS_bin_counts[i][j]
        table_configs["tables"].append(table_config)

    config_path = os.path.splitext(dataset_path)[0] + '_configs.json'
    with open(config_path, "w") as f:
        json.dump(table_configs, f)

    return lS_rows


def gen_table_configs(args, datasets):
    final_rows_per_table = None
    for d in datasets:
        print("Processing dataset:", d)
        np.random.seed(args.seed) # Guarantee datasets with the same number of tables use the same dims
        config_path = os.path.splitext(d)[0] + '_configs.json'
        if not os.path.exists(config_path):
            lS_rows = gen_config_per_table(args, d)
        else:
            lS_rows = None
            with open(config_path, "r") as f:
                tables = json.load(f)["tables"]
                if "num_embeddings" in tables[0].keys():
                    lS_rows = [x["num_embeddings"] for x in tables]
            if not lS_rows:
                lS_rows = gen_config_per_table(args, d)
        if not final_rows_per_table:
            final_rows_per_table = lS_rows
        else:
            final_rows_per_table = [max(x, y) for x, y in zip(lS_rows, final_rows_per_table)]

    for d in datasets:
        config_path = os.path.splitext(d)[0] + '_configs.json'
        with open(config_path, "r") as f:
            table_configs = json.load(f)
            for t in range(len(table_configs["tables"])):
                table_configs["tables"][t]["num_embeddings"] = final_rows_per_table[t]
        with open(config_path, "w") as f:
            json.dump(table_configs, f)


def compare_and_generate_common_configs(args, datasets):
    configs = [os.path.splitext(d)[0] + '_configs.json' for d in datasets]
    common_config = {
        "tables": []
    }
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            with open(configs[i], 'r') as f:
                json1 = json.load(f)["tables"]
            with open(configs[j], 'r') as f:
                json2 = json.load(f)["tables"]
            for idy, x in enumerate([json1, json2]):
                for idx in range(len(x)):
                    tbd = []
                    for k, v in x[idx].items():
                        if k.startswith('bin') or k.startswith('pooling'):
                            tbd.append(k)
                    for k in tbd:
                        del x[idx][k]
                    if i == 0 and j == 1 and idy == 0:
                        common_config["tables"].append(x[idx])
                    x[idx] = tuple(sorted(x[idx].items()))
            for x, y in zip(json1, json2):
                if x != y:
                    sys.exit("{} and {} are not from the same dataset! (Different field: {}, {})", configs[i], configs[j], x, y)
    with open(os.path.join(args.dataset_path, "common_configs.json"), 'w') as f:
        json.dump(common_config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate embedding lookup configs.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset-path', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag/2021")
    parser.add_argument('--dim-range', type=str, default="32,64,128,256")
    args = parser.parse_args()
    args.dim_range = list(map(int, args.dim_range.split(",")))

    if os.path.isdir:
        datasets = [f for f_ in [glob.glob(args.dataset_path + e) for e in ('/*.pt',)] for f in f_ if not os.path.basename(f).startswith('merged')]
    else:
        datasets = [args.dataset_path]
    gen_table_configs(args, datasets)
    compare_and_generate_common_configs(args, datasets)
