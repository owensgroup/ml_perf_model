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
import json
import glob
import os
import sys

import numpy as np
import torch
from numpy import random as ra

BATCH_SIZES = [512, 1024, 2048, 4096, 8192]


def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)
    return torch.tensor(P)


def merge_datasets(args):
    merged_file_path = os.path.join(args.dataset_path, "merged{}.pt".format('_simple' if args.simple else ''))
    if os.path.exists(merged_file_path):
        return
    if os.path.isdir:
        datasets = [f for f_ in [glob.glob(args.dataset_path + e) for e in ('/*.pt',)] for f in f_ if not os.path.basename(f).startswith('merged')]
    else:
        datasets = [args.dataset_path]
    if len(datasets) == 0:
        sys.exit("Datasets don't exist! Download from https://github.com/facebookresearch/dlrm_datasets.")

    num_tables = int(datasets[0].split('/')[-1].split('_')[1].strip('t'))
    complete_y = []
    complete_dense = []
    complete_indices = [[] for _ in range(num_tables)]
    complete_offsets = [[] for _ in range(num_tables)]
    complete_lengths = []
    complete_batch_size = 0
    m_den = 512

    table_with_zeros = set(list(range(num_tables)))
    table_all_zeros = set(list(range(num_tables)))
    for dataset_path in datasets:
        indices, offsets, lengths = torch.load(dataset_path)
        _, total_batch_size = lengths.shape # L per table per batch
        batch_sampled_per_dataset = 4096 if args.simple else total_batch_size # Whether merge full datasets or parts of each of them
        print("Processing {}, taking {}/{} batches...".format(dataset_path, batch_sampled_per_dataset, total_batch_size))

        # Check if data of a table in a dataset has zeros or is all zeros
        has_zeros = set([idx for idx, x in enumerate(torch.count_nonzero(lengths, dim=1)) if x != total_batch_size])
        all_zeros = set([idx for idx, x in enumerate(torch.sum(lengths, dim=1)) if x == 0])
        table_with_zeros = table_with_zeros.intersection(has_zeros)
        table_all_zeros = table_all_zeros.intersection(all_zeros)

        complete_y.append(generate_random_output_batch(batch_sampled_per_dataset, 1))
        complete_dense.append(torch.tensor(np.random.rand(batch_sampled_per_dataset, m_den).astype(np.float32)))

        # Each dataset has 856 / 788 tables
        # Each table has 65536 / 131072 batches
        # Old order: table 1-856/788 (batch 0-65535/131071), table 1-856/788 (batch 65536/131071-131072/262143), ...
        # New order: table 1 (batch 0-1048575/2097151), table 2 (batch 0-1048575/2097151), ...
        # New lengths: 856 * (1048576/2097152) (pooling factors per lookup per table, as it is)

        for t in range(num_tables):
            start = t * total_batch_size
            end = start + batch_sampled_per_dataset + 1
            table_offsets = offsets[start:end] # length = batch_sampled_per_dataset + 1 (as prefix sum)
            table_indices = indices[table_offsets[0]:table_offsets[-1]] # length = total number of lookups
            table_offsets = table_offsets - offsets[start] # length = batch_sampled_per_dataset, values start from 0

            complete_indices[t].append(table_indices)
            complete_offsets[t].append(table_offsets)

        complete_lengths.append(lengths[:, :batch_sampled_per_dataset])
        complete_batch_size += batch_sampled_per_dataset

    print("Table with zeros:", len(table_with_zeros), table_with_zeros)
    print("Table with all zeros:", len(table_all_zeros), table_all_zeros)

    # Up to here: complete_offsets = [
    #    [(0, ... x11 (65537/131073 values for dataset 1 table 1)), (0, ... x21 (65537/131073 values for dataset 2 table 1)) ... ], (list of 16 tensors)
    #    [(0, ... x12 (65537/131073 values for dataset 1 table 2)), (0, ... x22 (65537/131073 values for dataset 2 table 2)) ... ], (list of 16 tensors)
    #    ...
    # ]

    for t in range(num_tables):
        dataset_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in complete_indices[t]]).tolist() # Cumsum of number-of-lookups PER DATASET (size 17)
        complete_indices[t] = torch.cat([x.view(-1) for x in complete_indices[t]], dim=0).int() # Concatenated indices for each table, sizes vary
        complete_offsets[t] = torch.cat([torch.tensor([0])] + [x[1:] + y for x, y in zip(complete_offsets[t], dataset_offsets[:-1])], dim=0).int() # Offset of each lookup when lookups are concatenated
    table_indices_offsets = torch.tensor([0] + np.cumsum([x.view(-1).shape[0] for x in complete_indices]).tolist())

    # Up to here: complete_offsets = [
    #    [0, ...] (size of 1048576/2097152 + 1)
    #    [0, ...] (size of 1048576/2097152 + 1)
    #    ...
    # ]

    torch.save({
        "nbatches": complete_batch_size,
        "num_tables": complete_lengths[0].shape[0],
        "ly": torch.cat(complete_y, dim=0),
        "lX": torch.cat(complete_dense, dim=0),
        "lS_offsets": torch.cat([x for x in complete_offsets]),
        "lS_indices": torch.cat([x for x in complete_indices]), # Concat, as different tables have different numbers of lookups and it can't be a 2D tensor
        "table_indices_offsets": table_indices_offsets,
        "lengths": torch.cat(complete_lengths, dim=1)
    }, merged_file_path)

    merged = torch.load(merged_file_path)
    nbatches, num_tables, ly, lX, lS_offsets, lS_indices, table_indices_offsets, lengths = \
        merged["nbatches"], \
        merged["num_tables"], \
        merged["ly"], \
        merged["lX"], \
        merged["lS_offsets"], \
        merged["lS_indices"], \
        merged["table_indices_offsets"], \
        merged["lengths"]
    print(nbatches, num_tables, ly.shape, lX.shape, lS_offsets.shape, lS_indices.shape, table_indices_offsets.shape, lengths.shape)


def get_reuse_factor(indices):
    _, indices_counts = torch.unique(indices, return_counts=True)
    unique_counts, counts_counts = torch.unique(indices_counts, return_counts=True)
    total_counts = counts_counts.sum().item()

    if total_counts == 0:
        bin_counts = [0.0 for _ in range(17)]
    else:
        start, end = 0, 1
        bin_counts = []
        for _ in range(16):
            bin_counts.append(counts_counts[(unique_counts > start) & (unique_counts <= end)].sum().item())
            start = end
            end *= 2
        bin_counts.append(counts_counts[unique_counts > start].sum().item())
        bin_counts = [x/total_counts for x in bin_counts]
    return bin_counts


def get_average_rfs(dataset_path, B):
    dataset = torch.load(dataset_path)
    total_batch_size, num_tables, indices, offsets, table_indices_offsets = \
        dataset["nbatches"], \
        dataset["num_tables"], \
        dataset["lS_indices"], \
        dataset["lS_offsets"], \
        dataset["table_indices_offsets"]
    assert B <= total_batch_size
    TIDs = list(range(num_tables))

    # Sample 50 batches, calculate rfs, and take the mean as the average rfs of each table
    rfs = np.zeros((num_tables, 17))
    for _ in range(50):
        lS_bin_counts = []
        index = np.random.choice(total_batch_size - B, 1, replace=False).item()
        for idx in TIDs:
            offsets_start = (total_batch_size + 1) * idx + index
            offsets_end = (total_batch_size + 1) * idx + index + B + 1
            indices_start = table_indices_offsets[idx] + offsets[offsets_start]
            indices_end = table_indices_offsets[idx] + offsets[offsets_end - 1]
            batch_indices = indices[indices_start:indices_end]
            lS_bin_counts.append(get_reuse_factor(batch_indices))
        rfs += np.array(lS_bin_counts)
    return rfs / 50


def get_average_L(dataset_path):
    dataset = torch.load(dataset_path)
    total_batch_size, num_tables, offsets = \
        dataset["nbatches"], \
        dataset["num_tables"], \
        dataset["lS_offsets"]
    TIDs = list(range(num_tables))
    Ls = []
    for idx in TIDs:
        offsets_start = (total_batch_size + 1) * idx
        offsets_end = (total_batch_size + 1) * idx + total_batch_size + 1
        table_offsets = offsets[offsets_start:offsets_end] - offsets[offsets_start]
        start, end = table_offsets[0], table_offsets[-1]
        Ls.append((end - start).item() / total_batch_size)
    return Ls


def generate_merged_dataset_configs(args):
    # Dataset
    merged_file_path = os.path.join(args.dataset_path, "merged{}.pt".format('_simple' if args.simple else ''))

    # Config
    merged_configs_path = os.path.join(args.dataset_path, "merged{}_configs.json".format('_simple' if args.simple else ''))
    common_configs_paths = os.path.join(args.dataset_path, "common_configs.json")
    with open(common_configs_paths, "r") as f:
        common_configs = json.load(f)
    Ls = get_average_L(merged_file_path)
    for x in range(len(common_configs["tables"])):
        tmp = Ls[x]
        common_configs["tables"][x]["pooling_factor"] = float(f'{tmp:.4f}')
    for B in BATCH_SIZES:
        result = get_average_rfs(merged_file_path, B)
        for x in range(len(common_configs["tables"])):
            common_configs["tables"][x]["rfs_{}".format(B)] = '-'.join([f'{rf:.4f}' for rf in result[x, :].tolist()])
    with open(merged_configs_path, "w") as f:
        json.dump(common_configs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocess DLRM open source dataset: compare configs, generate common configs, and merge all parts.')
    parser.add_argument('--dataset-path', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag/2021")
    parser.add_argument('--simple', action="store_true", default=False)
    args = parser.parse_args()
    merge_datasets(args)
    generate_merged_dataset_configs(args)
