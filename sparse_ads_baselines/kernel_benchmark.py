import numpy as np
import torch, logging, click, sys, functools
import table_batched_embeddings_ops
logging.basicConfig(level=logging.DEBUG)


def div_round_up(a, b):
    return int((a + b - 1) // b) * b


def get_table_batched_offsets_from_dense(merged_indices):
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.int().contiguous().view(-1).cuda(),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).int().cuda(),
    )


def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def benchmark_concat(batch_size, M, N, K, iters):
    A = torch.randn(batch_size, M, K).cuda()
    B = torch.randn(batch_size, N, K).cuda()

    time_per_iter = benchmark_torch_function(
        iters,
        torch.cat,
        (A, B),
        dim=1
    )

    logging.info(
        f"Concat, tensor A size: ({batch_size}, {M}, {K}), tensor B size: ({batch_size}, {N}, {K}),\
            BW: {2 * (batch_size * M * K + batch_size * N * K) / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
    )


def benchmark_memcpy(batch_size, M, N, iters):
    A = torch.randn(batch_size, M, N)

    time_per_iter = benchmark_torch_function(
        iters,
        A.to,
        device="cuda"
    )

    logging.info(
        f"Memcpy, size: ({batch_size}, {M}, {N}), \
            BW: {(batch_size * M * N) / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
    )


def benchmark_forward(B, E, T, L, D, iters, fp16, managed, mixed):
    logging.basicConfig(level=logging.DEBUG)
    import torch
    import table_batched_embeddings

    np.random.seed(42)
    if mixed:
        mixed_D = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(mixed_D)
    cc = (
        table_batched_embeddings_ops.TableBatchedEmbeddingBags(
            T,
            E,
            D,
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.1,
            managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE
            if not managed
            else table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
            eps=0.1,
            stochastic_rounding=False,
            fp16=fp16,
        ).cuda()
        if not mixed
        else table_batched_embeddings_ops.MixedDimTableBatchedEmbeddingBags(
            [(E, d) for d in mixed_D],
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.1,
            managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE
            if not managed
            else table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
            eps=0.1,
            stochastic_rounding=False,
            fp16=fp16,
        ).cuda()
    )

    logging.info(
        f"Embedding parameters: {cc.embedding_weights.numel() / 1.0e9:.2f}GParam"
    )

    R = False

    def w2(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(w, o, x, *args):
            c(w, o, x.random_(0, E - 1), *args)

        return z

    def w3(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, x, *args):
            c(g, w, o, x.random_(0, E - 1), *args)

        return z

    def w4(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, a, x, *args):
            c(g, w, o, a, x.random_(0, E - 1), *args)

        return z

    def w6(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, a, b, d, x, *args):
            c(g, w, o, a, b, d, x.random_(0, E - 1), *args)

        return z

    zs = [
        torch.tensor(np.random.zipf(a=1.2, size=(B, L))).int().cuda()
        % E
        # torch.randint(low=0, high=E - 1, size=(T, B, L)).int().cuda()
    ]

    print(
        f"Duplicate proportion: {1.0 - np.unique(zs[0].detach().cpu().numpy()).size / zs[0].detach().cpu().numpy().size}"
    )
    merged_indices = torch.stack(zs, dim=0)

    merged_indices = torch.randint(low=0, high=E - 1, size=(T, B, L)).int().cuda()

    print(merged_indices.shape)

    (indices, offsets) = get_table_batched_offsets_from_dense(merged_indices)

    assert indices.shape[0] == B * T * L
    assert all(
        l == L for l in (offsets[1:] - offsets[:-1]).detach().cpu().numpy().tolist()
    )
    per_sample_weights = None
    print(indices.shape, indices.min(), indices.max(), indices)
    y0 = (
        table_batched_embeddings.forward(
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            False,
        )
        if not mixed
        else table_batched_embeddings.forward_mixed_D(
            cc.embedding_weights,
            cc.table_offsets,
            cc.dim_offsets,
            cc.total_D,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            False,
        )
    )

    for BT_block_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for shmem in [True, False]:
            y = (
                table_batched_embeddings.forward(
                    cc.embedding_weights,
                    cc.table_offsets,
                    indices,
                    offsets,
                    per_sample_weights,
                    L,
                    BT_block_size,
                    shmem,
                )
                if not mixed
                else table_batched_embeddings.forward_mixed_D(
                    cc.embedding_weights,
                    cc.table_offsets,
                    cc.dim_offsets,
                    cc.total_D,
                    indices,
                    offsets,
                    per_sample_weights,
                    L,
                    BT_block_size,
                    False,
                )
            )
            torch.testing.assert_allclose(y, y0)

    for BT_block_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for shmem in [True, False]:
            time_per_iter = (
                benchmark_torch_function(
                    iters,
                    w2(table_batched_embeddings.forward),
                    cc.embedding_weights,
                    cc.table_offsets,
                    indices,
                    offsets,
                    per_sample_weights,
                    L,
                    BT_block_size,
                    shmem,
                )
                if not mixed
                else benchmark_torch_function(
                    iters,
                    w4(table_batched_embeddings.forward_mixed_D),
                    cc.embedding_weights,
                    cc.table_offsets,
                    cc.dim_offsets,
                    cc.total_D,
                    indices,
                    offsets,
                    per_sample_weights,
                    L,
                    BT_block_size,
                    shmem,
                )
            )
            logging.info(
                f"Forward, B: {B} {(BT_block_size, shmem)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
            )

    go = torch.randn_like(y0)

    learning_rate = 0.05
    eps = 0.01
    for BT_block_size in [1, 2, 4, 8, 16, 32]:
        for shmem in [True, False]:
            time_per_iter = benchmark_torch_function(
                iters,
                w3(table_batched_embeddings.backward_sgd),
                go,
                cc.embedding_weights,
                cc.table_offsets,
                indices,
                offsets,
                learning_rate,
                L,
                BT_block_size,
                shmem,
            )

            logging.info(
                f"Backward-SGD, B: {B} {(BT_block_size, shmem)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
            )
    for BT_block_size in [
        1,
        2,
        4,
        8,
        16,
        32,
    ]:
        for exact in [0, 1]:
            for stochastic in [0, 1] if fp16 else [0]:
                if not exact:
                    time_per_iter = (
                        benchmark_torch_function(
                            iters,
                            w3(table_batched_embeddings.backward_approx_adagrad),
                            go,
                            cc.embedding_weights,
                            cc.table_offsets,
                            indices,
                            offsets,
                            per_sample_weights,
                            cc.optimizer_state,
                            learning_rate,
                            eps,
                            L,
                            stochastic,
                            BT_block_size,
                        )
                        if not mixed
                        else benchmark_torch_function(
                            iters,
                            w6(
                                table_batched_embeddings.backward_approx_adagrad_mixed_D
                            ),
                            go,
                            cc.embedding_weights,
                            cc.table_offsets,
                            cc.table_dim_offsets,
                            cc.dim_offsets,
                            cc.total_D,
                            indices,
                            offsets,
                            per_sample_weights,
                            cc.optimizer_state,
                            learning_rate,
                            eps,
                            L,
                            stochastic,
                            BT_block_size,
                        )
                    )
                else:
                    time_per_iter = (
                        benchmark_torch_function(
                            iters,
                            w3(table_batched_embeddings.backward_exact_adagrad),
                            go,
                            cc.embedding_weights,
                            cc.table_offsets,
                            indices,
                            offsets,
                            per_sample_weights,
                            cc.optimizer_state,
                            learning_rate,
                            eps,
                            stochastic,
                            BT_block_size,
                        )
                        if not mixed
                        else benchmark_torch_function(
                            iters,
                            w6(table_batched_embeddings.backward_exact_adagrad_mixed_D),
                            go,
                            cc.embedding_weights,
                            cc.table_offsets,
                            cc.table_dim_offsets,
                            cc.dim_offsets,
                            cc.total_D,
                            indices,
                            offsets,
                            per_sample_weights,
                            cc.optimizer_state,
                            learning_rate,
                            eps,
                            stochastic,
                            BT_block_size,
                        )
                    )

                logging.info(
                    f"Backward-ADAGRAD-{'nonstochastic' if not stochastic else 'stochastic'}-{'EXACT' if exact else 'APPROX'}-{'R' if R else 'NR'}, B: {B} ({BT_block_size}), E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
                )


@click.command()
@click.option("--op-type", default="embedding_lookup")
@click.option("--batch-size", default=128)
@click.option("--num-embeddings", default=1000)
@click.option("--num-tables", default=64)
@click.option("--bag-size", default=38)
@click.option("--embedding-dim", default=32)
@click.option("--iters", default=100)
@click.option("--M", default=512)
@click.option("--N", default=512)
@click.option("--K", default=512)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--managed", is_flag=True, default=False)
@click.option("--mixed", is_flag=True, default=False)
def cli(
    op_type,
    batch_size,
    num_embeddings,
    num_tables,
    bag_size,
    embedding_dim,
    iters,
    m,
    n,
    k,
    fp16,
    managed,
    mixed,
):
    if op_type == "embedding_lookup":
        benchmark_forward(
            batch_size,
            num_embeddings,
            num_tables,
            bag_size,
            embedding_dim,
            iters,
            fp16,
            managed,
            mixed,
        )
    elif op_type == "concat":
        benchmark_concat(
            batch_size,
            m,
            n,
            k,
            iters,
        )
    elif op_type == "memcpy":
        benchmark_memcpy(
            batch_size,
            m,
            n,
            iters,
        )
    else:
        raise Exception("Op type not supported!")


if __name__ == "__main__":
    cli()
