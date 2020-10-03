import click
import torch
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
import horovod.torch as hvd
import numpy as np
from sparse_embedding_cuda_ops import sparse_embedding_cuda, UniformShardedEmbeddingBags
import logging
logging.basicConfig(level=logging.DEBUG)
import sys


def get_offsets_from_dense(indices):
    (B, L) = indices.size()
    return indices.contiguous().view(-1), torch.tensor(np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int32)).cuda()

def get_merged_offsets_from_dense(merged_indices):
    (B, T, L) = merged_indices.size()
    merged_offsets = torch.tensor(np.fromfunction(lambda b, t: (b*T + t) * L, (B, T + 1), dtype=np.int32)).cuda()
    return merged_indices.contiguous().view(-1), merged_offsets
    
def benchmark_torch_function(iters, f, *args):
    f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        f(*args)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def benchmark_forward(B, E, T, L, D, iters, fp16):
    cc = UniformShardedEmbeddingBags(T, E, D).cuda()

    ccs = [
        torch.nn.EmbeddingBag(E, D, sparse=True, mode="sum").cuda()
        for _ in range(T)
    ]

    x = torch.randint(low=0, high=E - 1, size=(B, T, L)).cuda().int()
    xi = x.long()

    if fp16:
        ccs = [x.half() for x in ccs]
        cc = cc.half()

    x.requires_grad = False
    assert tuple(cc.embedding_weights.size()) == (E, T, D)
    assert tuple(x.size()) == (B, T, L)
    assert tuple(cc(x).size()) == (B, T, D)

    time_per_iter_sequential = benchmark_torch_function(
        iters, lambda: [c(xi[:, i, :]) for i, c in enumerate(ccs)])
    time_per_iter = benchmark_torch_function(iters, cc, x)
    yy = cc(x)
    print(yy.dtype, yy.shape)
    time_per_iter_fast = benchmark_torch_function(
        iters, sparse_embedding_cuda.forward_fast_single, cc.embedding_weights,
        x)

    (indices, offsets) = get_merged_offsets_from_dense(x)

    time_per_iter_fast_offsets = benchmark_torch_function(
        iters, sparse_embedding_cuda.forward_offsets, cc.embedding_weights,
        indices, offsets)

    import json
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter_fast, implementation="Fused", method="forward")))
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter_fast_offsets, implementation="Fused-Offsets", method="forward")))
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter, implementation="Fused-Slow", method="forward")))
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter_sequential, implementation="Baseline", method="forward")))

    logging.info(
        f"Forward, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {4 * B * T * L * D / time_per_iter_fast / 1.0e9}GB/s, speedup: {time_per_iter / time_per_iter_fast}, offset-cost: {time_per_iter_fast / time_per_iter_fast_offsets}, speedup-seq: {time_per_iter_sequential / time_per_iter_fast}"
    )
    time_per_iter = benchmark_torch_function(
        iters, sparse_embedding_cuda.backward_update_single, yy,
        cc.embedding_weights, x, 0.05)
    time_per_iter_fast = benchmark_torch_function(
        iters, sparse_embedding_cuda.backward_update_fast_single, yy,
        cc.embedding_weights, x, 0.05)

    time_per_iter_fast_offsets = benchmark_torch_function(
        iters, sparse_embedding_cuda.backward_update_offsets, yy,
        cc.embedding_weights, indices, offsets, 0.05)

    ys = [c(xi[:, i, :]) for i, c in enumerate(ccs)]
    gos = [torch.rand_like(y) for y in ys]
    try:
        time_per_iter_sequential = benchmark_torch_function(
            iters, lambda:
            [y.backward(go, retain_graph=True) for y, go in zip(ys, gos)])
    except:
        logging.exception("Failed computing backward")
        # TODO: OOMs?
        time_per_iter_sequential = time_per_iter


    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter_fast, implementation="Fused", method="backward")))
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter_fast_offsets, implementation="Fused-Offsets", method="backward")))
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter, implementation="Fused-Slow", method="backward")))
    print(json.dumps(dict(B=B, E=E, T=T, D=D, L=L, time_per_iter=time_per_iter_sequential, implementation="Baseline", method="backward")))

    logging.info(
        f"Backward, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {4 * B * T * L * D / time_per_iter_fast / 1.0e9}GB/s, speedup: {time_per_iter / time_per_iter_fast}, offset-cost: {time_per_iter_fast / time_per_iter_fast_offsets}, speedup-seq: {time_per_iter_sequential / time_per_iter_fast}"
    )

@click.command()
@click.option("--num-tables", default=64)
@click.option("--num-embeddings", default=int(1e4))
@click.option("--embedding-dim", default=32)
@click.option("--batch-size", default=128)
@click.option("--bag-size", default=32)
@click.option("--iters", default=100)
@click.option("--remote", is_flag=True, default=False)
@click.option("--fp16", is_flag=True, default=False)

def cli(num_tables, num_embeddings, embedding_dim, batch_size, bag_size, iters,
        remote, fp16):
    def f():
        benchmark_forward(batch_size, num_embeddings, num_tables, bag_size,
                          embedding_dim, iters, fp16)

    if remote:
        import submitit
        executor = submitit.AutoExecutor(folder="sparse_embedding_perf")
        executor.update_parameters(timeout_min=10,
                                   partition="dev",
                                   constraint="volta",
                                   gpus_per_node=1)
        job = executor.submit(f)
        job.wait()
        job.result()
        logging.info("Finished")
        import time
        time.sleep(5)
        print(job.stdout())
        print(job.stderr(), file=sys.stderr)
        logging.info("Finished")
    else:
        f()


if __name__ == "__main__":
    cli()
