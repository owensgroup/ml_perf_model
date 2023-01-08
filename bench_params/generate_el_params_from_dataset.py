import torch, argparse, os, json
import numpy as np
from .generate_el_table_configs_from_datasets import gen_table_configs

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

        params.append("{} {} {} 0 {}\n".format(
            B,
            "-".join([str(t) for t in TIDs]),
            T,
            "-".join([str(d) for d in Ds])
        ))
        if len(params) >= args.num_samples:
            break

    with open("./embedding_lookup_params_dlrm_datasets.txt", 'a+') as f:
        for p in params:
            f.write(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate all-to-all params.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-gpu-memory', type=int, default=16*1024*1024*1024) # V100, in bytes
    parser.add_argument('--dataset-path', type=str, default="/nvme/deep-learning/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536.pt")
    parser.add_argument('--num-samples', type=int, default=20000)
    parser.add_argument('--per-gpu-table-limit', type=int, default=8)
    parser.add_argument('--batch-sizes', type=str, default="512,1024,2048,4096")
    parser.add_argument('--dim-range', type=str, default="32,64,128,256")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.batch_sizes = list(map(int, args.batch_sizes.split(",")))
    args.dim_range = list(map(int, args.dim_range.split(",")))

    args.table_config_path = os.path.splitext(args.dataset_path)[0] + '_configs.json'
    if not os.path.exists(args.table_config_path):
        gen_table_configs(args)

    process_data(args)
