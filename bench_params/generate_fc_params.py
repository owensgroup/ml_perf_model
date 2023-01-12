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

RANGE = [
    32, 64, 96, 128, 160, 192, 224, 
    256, 264, 320, 328, 384, 392, 448, 456, 
    512, 520, 640, 648, 768, 776, 896, 904, 
    1024, 1032, 1536, 1544, 2048, 2056, 3080, 3088, 4096, 4104
]
MEMORY_SCALE_FACTOR = 0.9

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate all-to-all params.')
    parser.add_argument('--is-big', action='store_true', default=False)
    parser.add_argument('--per-gpu-memory', type=int, default=16*1024*1024*1024) # V100, in bytes
    parser.add_argument('--num-samples', type=int, default=5000)
    args = parser.parse_args()

    batch_sizes = [512, 1024, 2048, 4096] if args.is_big else [1]
    all_perms = []

    for iter in range(args.num_samples):
        B = np.random.choice(batch_sizes, 1).item()
        M = np.random.choice(RANGE, 1).item()
        K = np.random.choice(RANGE, 1).item()
        N = np.random.choice(RANGE, 1).item()
        if B * (M * K + N * K + M * N) * 4 < args.per_gpu_memory * MEMORY_SCALE_FACTOR:
            all_perms.append((B, M, K, N)) # Forward
            all_perms.append((B, M, N, K)) # Backward: MN * NK = MK
            all_perms.append((B, M, N, K)) # Backward: MN' (NM) * MK = NK
    all_perms = sorted(list(set(all_perms)))

    with open('./fc_params{}.txt'.format('_big' if args.is_big else ''), 'a+') as f:
        for p in all_perms:
            f.write(' '.join([str(x) for x in p]) + '\n')
