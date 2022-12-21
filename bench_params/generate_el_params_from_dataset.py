import torch
import numpy as np

def process_data(data_path):
    torch.set_printoptions(profile="full")
    np.random.seed(0)

    _, _, lengths = torch.load(data_path)
    num_tables, _ = lengths.shape # L per table per batch?

    table_limit = 8
    num_tests = 20000
    batch_sizes = [512, 1024, 2048, 4096]
    dims = np.random.choice([32, 64, 128, 256], num_tables)

    for _ in range(num_tests):
        # Batch size
        B = np.random.choice(batch_sizes, 1).item()

        # Number of tables
        T = np.random.randint(2, table_limit)

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

                TIDs.append(str(t.item()))

                if len(TIDs) >= T:
                    break_while = True
                    break
            multiples += 1

        # Dims of tables
        Ds = [str(dims[int(t)].item()) for t in TIDs]

        with open("./embedding_lookup_params_dlrm_datasets.txt", 'a+') as f:
            f.write("{} {} {} 0 {} 0\n".format(
                B,
                "-".join(TIDs),
                T,
                "-".join(Ds)
            ))

if __name__ == "__main__":
    process_data("/nvme/deep-learning/dlrm_datasets/embedding_bag/fbgemm_t856_bs65536.pt")