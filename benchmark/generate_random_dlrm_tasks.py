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

###############
# This code is adapted from https://github.com/daochenzha/dreamshard/blob/main/tools/gen_tasks.py
###############


import argparse, os, json
import numpy as np

from analysis.utils import PM_HOME, GPU_COUNT, GPU_PARAMS

MEMORY_SCALE_FACTOR = 0.8
TABLE_LOWER_LIMIT_FACTOR = 0.7
TABLE_UPPER_LIMIT_FACTOR = 1.3


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate DLRM tasks")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-gpu-memory', type=int, default=-1)
    parser.add_argument('--data-dir', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag/2021")
    parser.add_argument('--config-name', type=str, default="common")
    parser.add_argument('--out-dir', type=str, default="{}/benchmark".format(PM_HOME))
    parser.add_argument('--per-gpu-table-limit', type=int, default=13)
    parser.add_argument('--num-tasks', type=int, default=30)
    parser.add_argument('--heavy-only', action="store_true", default=False)

    args = parser.parse_args()
    np.random.seed(args.seed)
    min_table_count = int(args.per_gpu_table_limit * GPU_COUNT * TABLE_LOWER_LIMIT_FACTOR)
    max_table_count = int(args.per_gpu_table_limit * GPU_COUNT * TABLE_UPPER_LIMIT_FACTOR)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.data_dir, args.config_name + "_configs.json"), "r") as f:
        table_configs = json.load(f)["tables"]
    T = len(table_configs)
    # Only consider L >= 20 if heavy-only
    table_indices = [i for i in range(T) if (
        (table_configs[i]["all_zero"] == 0 and table_configs[i]["pooling_factor"] >= 20) \
            if args.heavy_only else \
        (table_configs[i]["all_zero"] == 0)
    )]
    np.random.shuffle(table_indices)
    dataset_suffix = args.data_dir.split('/')[-1]

    out_path = os.path.join(args.out_dir, "tasks_{}_{}.txt".format(dataset_suffix, GPU_COUNT))
    if os.path.exists(out_path):
        exit()

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
            DRAM_size = GPU_PARAMS["DRAM_size"] if args.per_gpu_memory == -1 else args.per_gpu_memory
            if table_mem_sum > GPU_PARAMS["DRAM_size"] * MEMORY_SCALE_FACTOR:
                continue

            table_ids = "-".join(list(map(str, table_ids)))
            f.write(table_ids + "\n")
            i += 1
