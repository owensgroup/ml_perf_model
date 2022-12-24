import argparse
import numpy as np
from itertools import combinations_with_replacement

TABLE_LIMIT = {
    2: 15,
    4: 10,
    8: 6,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate all-to-all params.')
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--per-gpu-memory', type=int, default=16*1024*1024*1024)
    parser.add_argument('--num-samples', type=int, default=-1)
    args = parser.parse_args()

    num_table_limit = TABLE_LIMIT[args.num_gpus]
    batch_sizes = [256, 512, 1024, 2048, 4096]
    embedding_dims = [32, 64, 128, 256]
    all_perms = []
    if args.num_samples == -1: # All permutations
        for B in batch_sizes:
            cs = combinations_with_replacement(np.arange(1, num_table_limit+1), args.num_gpus)
            for c in cs:
                for D in embedding_dims:
                    if B * sum(c) * D * 4 < args.per_gpu_memory:
                        all_perms.append([B] + list(c) + [D])
    else: # Random
        for iter in range(args.num_samples):
            B = np.random.choice(batch_sizes, 1).item()
            c = np.random.choice(np.arange(1, num_table_limit+1), args.num_gpus, replace=True).tolist()
            D = np.random.choice(embedding_dims, 1).item()
            if B * sum(c) * D * 4 < args.per_gpu_memory:
                all_perms.append([B] + c + [D])

    with open('./a2a_{}_params.txt'.format(args.num_gpus), 'a+') as f:
        for p in all_perms:
            f.write(' '.join([str(x) for x in p]) + '\n')
