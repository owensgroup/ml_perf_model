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
import numpy as np
from itertools import combinations_with_replacement

TABLE_LIMIT = {
    2: 15,
    4: 10,
    8: 6,
}
MEMORY_SCALE_FACTOR = 0.9

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate all-to-all params.')
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--per-gpu-memory', type=int, default=16*1024*1024*1024) # V100, in bytes
    parser.add_argument('--num-samples', type=int, default=20000)
    args = parser.parse_args()

    num_table_limit = TABLE_LIMIT[args.num_gpus]
    batch_sizes = [256, 512, 1024, 2048, 4096]
    embedding_dims = [32, 64, 128, 256]
    all_perms = set()
    if args.num_samples == -1: # All permutations
        for B in batch_sizes:
            cs = combinations_with_replacement(np.arange(1, num_table_limit+1), args.num_gpus)
            for c in cs:
                for D in embedding_dims:
                    if B * sum(c) * D * 4 < args.per_gpu_memory * MEMORY_SCALE_FACTOR:
                        all_perms.append(tuple([B] + list(c) + [D]))
    else: # Random
        while len(all_perms) < args.num_samples:
            B = np.random.choice(batch_sizes, 1).item()
            c = np.random.choice(np.arange(1, num_table_limit+1), args.num_gpus, replace=True).tolist()
            D = np.random.choice(embedding_dims, 1).item()
            if B * sum(c) * D * 4 < args.per_gpu_memory * MEMORY_SCALE_FACTOR:
                all_perms.add(tuple([B] + c + [D]))
    all_perms = sorted(list(set(all_perms)))

    with open('./a2a_{}_params.txt'.format(args.num_gpus), 'a+') as f:
        for p in all_perms:
            f.write(' '.join([str(x) for x in p]) + '\n')
