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

import torch, argparse, os, json, glob
import numpy as np

MEMORY_SCALE_FACTOR = 0.9

def process_data(args):
    torch.set_printoptions(profile="full")
    with open(args.table_config_path, "r") as f:
        table_configs = json.load(f)["tables"]
    _, _, lengths = torch.load(args.dataset_path)
    num_tables, _ = lengths.shape # L per table per batch?

    params = []
    while True:
        # Batch size
        B = np.random.choice(args.batch_sizes, 1).item()

        # Number of tables
        T = np.random.randint(2, args.per_gpu_table_limit)

        # Table IDs
        TIDs = []

        break_while = False
        multiples = 2
        while not break_while:
            # Randomly pick a bunch of tables. Extra to handle zero-lookup
            t_idx = np.sort(np.random.choice(num_tables, min(T * multiples, num_tables), replace=False))

            TIDs.clear()
            for t in t_idx:
                Ls_nonzero = torch.nonzero(lengths[t])
                # Not enough batches with non-zero lookups
                if len(Ls_nonzero) < B:
                    continue

                TIDs.append(t.item())

                if len(TIDs) >= T:
                    break_while = True
                    break
            multiples += 1

        # Es and Ds
        Es = [table_configs[tid]['num_embeddings'] for tid in TIDs]
        Ds = [table_configs[tid]['embedding_dim'] for tid in TIDs]

        # Skip if OOM
        table_mem_sum = sum([E * D for E, D in zip(Es, Ds)]) * 4
        if table_mem_sum > args.per_gpu_memory * MEMORY_SCALE_FACTOR:
            continue

        params.append("{} {} {} 0 {} {}\n".format(
            B,
            "-".join([str(t) for t in TIDs]),
            T,
            "-".join([str(d) for d in Ds]),
            args.dataset_path,
        ))
        if len(params) >= args.num_samples:
            break

    with open("./embedding_lookup_params_dlrm_datasets{}.txt".format("_test" if args.is_test else ""), 'a+') as f:
        for p in params:
            f.write(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Sample batches of embedding lookup params from datasets.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--is-test", action="store_true", default=False)
    parser.add_argument('--per-gpu-memory', type=int, default=16*1024*1024*1024) # V100, in bytes
    parser.add_argument('--dataset-path', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag")
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--per-gpu-table-limit', type=int, default=8)
    parser.add_argument('--batch-sizes', type=str, default="512,1024,2048,4096")
    args = parser.parse_args()
    np.random.seed(args.seed + (42 if args.is_test else 0))
    args.batch_sizes = list(map(int, args.batch_sizes.split(",")))

    if os.path.isdir:
        datasets = glob.glob(args.dataset_path + '/*/*.pt')
        if not args.is_test: # Only sample from half of the datasets for training
            idxs = np.random.choice(len(datasets), len(datasets) // 2, replace=False)
            datasets = [datasets[i] for i in idxs]
    else:
        datasets = [args.dataset_path]
    args.num_samples = int(args.num_samples / len(datasets)) # Num of samples per dataset

    for d in datasets:
        print(args.num_samples, d)
        args.dataset_path = d

        splitext_filename = os.path.splitext(args.dataset_path)[0]
        args.table_config_path = splitext_filename + '_configs.json'

        process_data(args)
