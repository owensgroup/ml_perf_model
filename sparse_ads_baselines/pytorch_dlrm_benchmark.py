import click

import torch
from torch import nn, Tensor
from typing import List

import horovod.torch as hvd
import apex
from sparse_embedding_cuda_ops import UniformShardedEmbeddingBags, FastZeroFusedSGD
from models import SNN, UniformShardedSNN, Criterion

import logging
logging.basicConfig(level=logging.DEBUG)
import sys
import json
import os

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


def benchmark_torch_snn_forward(name, num_tables, num_embeddings,
                                embedding_dim, dense_features_dim, batch_size,
                                bag_size, iters):
    net = SNN(num_tables, num_embeddings, embedding_dim, dense_features_dim)

    dense_features = torch.randn(batch_size, dense_features_dim).cuda()
    dense_features.requires_grad = False

    sparse_features = [
        torch.randint(low=0,
                      high=embedding_dim - 1,
                      size=(batch_size, bag_size)).cuda()
        for _ in range(num_tables)
    ]
    for sf in sparse_features:
        sf.requires_grad = False

    labels = torch.rand(size=(batch_size, 1),
                        device=torch.cuda.current_device())
    weights = torch.rand(size=(batch_size, 1),
                        device=torch.cuda.current_device())
    weights.requires_grad = False
    logits = net(dense_features, sparse_features)
    net = torch.jit.trace(net,
                          example_inputs=(dense_features, sparse_features))

    def forward(dense_features, sparse_features):
        return net(dense_features, sparse_features)

    time_per_batch = benchmark_torch_function(iters, forward, dense_features,
                                              sparse_features)

    logging.info(
        f"{name}, DLRM, FORWARD: Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, Dense Features Dim: {dense_features_dim}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
    )
    print(json.dumps(dict(name=name, implementation="baseline", method="forward", B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))
    criterion = Criterion().cuda()
    criterion = torch.jit.trace(criterion, example_inputs=(logits, labels, weights))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    def forward_backward_update(dense_features, sparse_features, labels):
        optimizer.zero_grad()
        logits = net(dense_features, sparse_features)
        loss = criterion(logits.float(), labels, weights)
        loss.backward()
        optimizer.step()

    time_per_batch = benchmark_torch_function(iters, forward_backward_update,
                                              dense_features, sparse_features,
                                              labels)
    logging.info(
        f"{name}, DLRM, FORWARDBACKWARDUPDATE: Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, Dense Features Dim: {dense_features_dim}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
    )
    print(json.dumps(dict(name=name, implementation="baseline", method="forwardbackwardupdate", B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))



def benchmark_torch_uniform_snn_forward(name,
                                        num_tables,
                                        num_embeddings,
                                        embedding_dim,
                                        dense_features_dim,
                                        batch_size,
                                        bag_size,
                                        iters,
                                        fp16=0):
    net = UniformShardedSNN(num_tables, num_embeddings, embedding_dim,
                            dense_features_dim).cuda()
    dense_features = torch.randn(batch_size,
                                 dense_features_dim,
                                 device=torch.cuda.current_device())
    dense_features.requires_grad = False
    sharded_sparse_features = torch.randint(low=0,
                                            high=num_embeddings,
                                            size=(batch_size, num_tables,
                                                  bag_size),
                                            device=torch.cuda.current_device()).int()
    # hack - compute grad for sparse-features to avoid having to allocate a grad for weights.
    sharded_sparse_features.requires_grad = False
    labels = torch.rand(size=(batch_size, 1),
                        device=torch.cuda.current_device())
    weights = torch.rand(size=(batch_size, 1),
                        device=torch.cuda.current_device())
    weights.requires_grad = False

    if fp16:
        net_fp16 = apex.amp.initialize(net, opt_level="O2", verbosity=0)
        # net_fp16 = torch.jit.trace(net_fp16,
        #                            example_inputs=(dense_features,
        #                                            sharded_sparse_features))
    net = torch.jit.trace(net,
                          example_inputs=(dense_features,
                                          sharded_sparse_features))

    def forward(dense_features, partitioned_sparse_features):
        return (net if not fp16 else net_fp16)(dense_features,
                                               sharded_sparse_features.random_(0, num_embeddings))

    if os.environ.get('BIGADS_PROFILE_FORWARD'):
        with torch.autograd.profiler.profile(use_cuda=True,
                                             record_shapes=True) as prof:
            time_per_batch = benchmark_torch_function(iters, forward, dense_features,
                                                    sharded_sparse_features)
        prof.export_chrome_trace(("fp16-" if fp16 else "fp32-") + os.environ.get('BIGADS_PROFILE_FORWARD'))
    else:
        time_per_batch = benchmark_torch_function(iters, forward, dense_features,
                                                  sharded_sparse_features)
    if fp16:
        del net_fp16

    logging.info(
        f"{name}, UNIFORM, FORWARD: Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, Dense Features Dim: {dense_features_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
    )
    print(json.dumps(dict(name=name, implementation="fused", method="forward", B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))

    criterion = Criterion().cuda()
    logits = net(dense_features, sharded_sparse_features)
    criterion = torch.jit.trace(criterion, example_inputs=(logits, labels, weights))

    if fp16:
        net = UniformShardedSNN(num_tables, num_embeddings, embedding_dim,
                                dense_features_dim).cuda()
    dense_named_parameters = [(k, v) for (k, v) in net.named_parameters()
                              if "embedding" not in k]
    # optimizer = apex.optimizers.FusedSGD([v for (_, v) in dense_named_parameters], lr=0.05)
    optimizer = FastZeroFusedSGD([v for (_, v) in dense_named_parameters],
                                 lr=0.05)

    if fp16:
        net, optimizer = apex.amp.initialize(net,
                                             optimizer,
                                             opt_level="O2",
                                             verbosity=0)
        # net = torch.jit.trace(net,
        #                       example_inputs=(dense_features,
        #                                       sharded_sparse_features))
        optimizer.zero_grad = optimizer.amp_zero_grad

    def forward_backward_update(dense_features, sharded_sparse_features,
                                labels):
        optimizer.zero_grad()
        logits = net(dense_features, sharded_sparse_features.random_(0, num_embeddings))
        loss = criterion(logits.float(), labels, weights)
        loss.backward()
        optimizer.step()

    if os.environ.get('BIGADS_PROFILE'):
        with torch.autograd.profiler.profile(use_cuda=True,
                                             record_shapes=True) as prof:
            time_per_batch = benchmark_torch_function(iters,
                                                      forward_backward_update,
                                                      dense_features,
                                                      sharded_sparse_features,
                                                      labels)
        prof.export_chrome_trace(("fp16-" if fp16 else "fp32-") + os.environ.get('BIGADS_PROFILE'))
    else:
        time_per_batch = benchmark_torch_function(iters,
                                                  forward_backward_update,
                                                  dense_features,
                                                  sharded_sparse_features,
                                                  labels)

    logging.info(
        f"{name}, UNIFORM, FORWARDBACKWARDUPDATE: Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, Dense Features Dim: {dense_features_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
    )
    print(json.dumps(dict(name=name, implementation="fused", method="forwardbackwardupdate", B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))



def div_round_up(a, b):
    return int((a + b - 1) // b) * b


@click.command()
@click.option("--num-tables", default=64)
@click.option("--num-embeddings", default=div_round_up(1e4, 96))
@click.option("--embedding-dim", default=32)
@click.option("--dense-features-dim", default=512)
@click.option("--batch-size", default=128)
@click.option("--bag-size", default=32)
@click.option("--iters", default=100)
@click.option("--remote", is_flag=True, default=False)
def cli(num_tables, num_embeddings, embedding_dim, dense_features_dim,
        batch_size, bag_size, iters, remote):
    def f():
        benchmark_torch_snn_forward("dlrm", num_tables, num_embeddings,
                                    embedding_dim, dense_features_dim,
                                    batch_size, bag_size, iters)
        benchmark_torch_uniform_snn_forward("fused", num_tables,
                                            num_embeddings, embedding_dim,
                                            dense_features_dim, batch_size,
                                            bag_size, iters)
        benchmark_torch_uniform_snn_forward("fused-fp16", num_tables,
                                            num_embeddings, embedding_dim,
                                            dense_features_dim, batch_size,
                                            bag_size, iters, fp16=1)

        # benchmark_torch_uniform_snn_forward("fused",
        #                                     num_tables,
        #                                     num_embeddings,
        #                                     embedding_dim,
        #                                     dense_features_dim,
        #                                     batch_size,
        #                                     bag_size,
        #                                     iters,
        #                                     fp16=1)

    if remote:
        import submitit
        import sys
        executor = submitit.AutoExecutor(folder="dlrm_perf")
        executor.update_parameters(timeout_min=10,
                                   partition="dev",
                                   constraint="volta",
                                   gpus_per_node=1)
        job = executor.submit(f)
        job.wait()
        job.result()
        logging.info("Finished")
        import time
        time.sleep(1)
        print(job.stdout())
        print(job.stderr(), file=sys.stderr)
        logging.info("Finished")
    else:
        f()


if __name__ == "__main__":
    cli()