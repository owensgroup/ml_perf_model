import torch
from torch import nn, Tensor
from typing import List

from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
import horovod.torch as hvd
import apex
import sparse_embedding_cuda
from mpi4py import MPI

import click

import torch
from torch import nn, Tensor
from typing import List

import horovod.torch as hvd
import apex
from sparse_embedding_cuda_ops import UniformShardedEmbeddingBags, FastZeroFusedSGD
from models import DistributedUniformShardedSNN, DistributedPartitionShardedSNN, Criterion
from mpi4py import MPI
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import logging
logging.basicConfig(level=logging.DEBUG)

FP16_LEVEL = "O2"

from contextlib import contextmanager

class SingleGPUDDP(DDP):
    def __init__(self, *args, **kwargs):
        assert torch.distributed.is_initialized()
        kwargs.update(dict(broadcast_buffers=False))
        super(SingleGPUDDP, self).__init__(*args, **kwargs)

    # Minor optimization to avoid calls to Scatter/Gather for 1-GPU case
    def scatter(self, inputs, kwargs, device_ids):
        if len(device_ids) == 1:
            return [inputs], [kwargs]
        return super(SingleGPUDDP, self).__init__(inputs, kwargs, device_ids)

    def forward(self, *args, **kwargs):
        return super(SingleGPUDDP, self).forward(*args, **kwargs)
        
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


def benchmark_torch_uniform_snn_forward(name,
                                        num_tables,
                                        num_embeddings,
                                        embedding_dim,
                                        dense_features_dim,
                                        batch_size,
                                        bag_size,
                                        iters,
                                        fp16=0):
    assert batch_size % hvd.size() == 0
    assert num_embeddings % hvd.size() == 0
    assert bag_size % hvd.size() == 0
    net = DistributedUniformShardedSNN(num_tables, num_embeddings,
                                       embedding_dim,
                                       dense_features_dim).cuda()

    dense_features = torch.randn(batch_size // hvd.size(),
                                 dense_features_dim,
                                 device=torch.cuda.current_device())
    dense_features.requires_grad = False
    sharded_sparse_features = torch.randint(low=0,
                                            high=num_embeddings // hvd.size(),
                                            size=(batch_size, num_tables,
                                                  bag_size // hvd.size()),
                                            device=torch.cuda.current_device()).int()
    sharded_sparse_features.requires_grad = False
    labels = torch.rand(size=(batch_size // hvd.size(), 1),
                        device=torch.cuda.current_device())
    weights = torch.rand(size=(batch_size // hvd.size(), 1),
                        device=torch.cuda.current_device())
    weights.requires_grad = False

    logits = net(dense_features, sharded_sparse_features)

    net = torch.jit.trace(net,
                          example_inputs=(dense_features,
                                          sharded_sparse_features))

    if fp16:
        net = apex.amp.initialize(net, opt_level=FP16_LEVEL, verbosity=0)
        # net_fp16 = torch.jit.trace(net_fp16,
        #                            example_inputs=(dense_features,
        #                                            sharded_sparse_features))

    def forward(dense_features, partitioned_sparse_features):
        return net(
            dense_features,
            sharded_sparse_features.random_(0, num_embeddings // hvd.size()))

    MPI.COMM_WORLD.Barrier()
    time_per_batch = benchmark_torch_function(iters, forward, dense_features,
                                              sharded_sparse_features)
    print(json.dumps(dict(name=name, method="forward", implementation="uniform", rank=hvd.rank(), workers=hvd.size(), B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, fp16=fp16, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))
    if hvd.rank() == 0:
        logging.info(
            f"{name}, UNIFORM, FORWARD: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )

    criterion = Criterion().cuda()
    criterion = torch.jit.trace(criterion, example_inputs=(logits.float(), labels, weights))

    def forward_backward(dense_features, sharded_sparse_features, labels):
        logits = net(
            dense_features,
            sharded_sparse_features.random_(0, num_embeddings // hvd.size()))
        loss = criterion(logits.float(), labels, weights)
        loss.backward()
    MPI.COMM_WORLD.Barrier()
    time_per_batch = benchmark_torch_function(iters, forward_backward,
                                              dense_features,
                                              sharded_sparse_features, labels)
    print(json.dumps(dict(name=name, method="forwardbackward", implementation="uniform", rank=hvd.rank(), workers=hvd.size(), B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, fp16=fp16, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))

    if hvd.rank() == 0:
        logging.info(
            f"{name}, UNIFORM, FORWARDBACKWARD: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )

    # optimizer = apex.optimizers.FusedSGD([v for (_, v) in dense_named_parameters], lr=0.05)
    net = DistributedUniformShardedSNN(num_tables, num_embeddings,
                                       embedding_dim,
                                       dense_features_dim).cuda()
    # TODO: Broken, PyTorch JIT bug.
    # net = torch.jit.trace(net,
    #                       example_inputs=(dense_features,
    #                                       sharded_sparse_features))
    dense_named_parameters = [(k, v) for (k, v) in net.named_parameters()
                              if "embedding" not in k]
    optimizer = FastZeroFusedSGD([v for (_, v) in dense_named_parameters],
                                 lr=0.05)

    if fp16:
        net, optimizer = apex.amp.initialize(net,
                                             optimizer,
                                             opt_level=FP16_LEVEL,
                                             verbosity=0)
        # net = torch.jit.trace(net,
        #                       example_inputs=(dense_features,
        #                                       sharded_sparse_features))
        optimizer.zero_grad = optimizer.amp_zero_grad

    net.dense_arch = SingleGPUDDP(net.dense_arch,
                                 device_ids=[torch.cuda.current_device()])
    net.over_arch = SingleGPUDDP(net.over_arch,
                                  device_ids=[torch.cuda.current_device()])


    # hvd.broadcast_parameters(dense_named_parameters, root_rank=0)
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    def forward_backward_update(dense_features, sharded_sparse_features,
                                labels):
        optimizer.zero_grad()
        logits = net(
            dense_features,
            sharded_sparse_features.random_(0, num_embeddings // hvd.size()))
        loss = criterion(logits, labels, weights)
        loss.backward()
        optimizer.step()

    MPI.COMM_WORLD.Barrier()
    import os
    if os.environ.get('BIGADS_PROFILE') and hvd.rank() == 0:
        with torch.autograd.profiler.profile(use_cuda=True,
                                             record_shapes=True) as prof:
            time_per_batch = benchmark_torch_function(iters,
                                                      forward_backward_update,
                                                      dense_features,
                                                      sharded_sparse_features,
                                                      labels)
        prof.export_chrome_trace(f"{'fp16' if fp16 else 'fp32'}-{hvd.size()}-uniform_{os.environ.get('BIGADS_PROFILE')}")
    else:
        time_per_batch = benchmark_torch_function(iters,
                                                  forward_backward_update,
                                                  dense_features,
                                                  sharded_sparse_features,
                                                  labels)
    print(json.dumps(dict(name=name, method="forwardbackwardupdate", implementation="uniform", rank=hvd.rank(), workers=hvd.size(), B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, fp16=fp16, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))
    if hvd.rank() == 0:
        logging.info(
            f"{name}, UNIFORM, FORWARDBACKWARDUPDATE: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )



def benchmark_torch_partitioned_snn_forward(name,
                                        num_tables,
                                        num_embeddings,
                                        embedding_dim,
                                        dense_features_dim,
                                        batch_size,
                                        bag_size,
                                        iters,
                                        fp16=0):
    assert batch_size % hvd.size() == 0
    assert num_tables % hvd.size() == 0
    net = DistributedPartitionShardedSNN(num_tables, num_embeddings,
                                       embedding_dim,
                                       dense_features_dim).cuda()

    dense_features = torch.randn(batch_size // hvd.size(),
                                 dense_features_dim,
                                 device=torch.cuda.current_device())
    dense_features.requires_grad = False
    sharded_sparse_features = torch.randint(low=0,
                                            high=num_embeddings,
                                            size=(batch_size, num_tables // hvd.size(),
                                                  bag_size),
                                            device=torch.cuda.current_device()).int()
    sharded_sparse_features.requires_grad = False
    labels = torch.rand(size=(batch_size // hvd.size(), 1),
                        device=torch.cuda.current_device())
    weights = torch.rand(size=(batch_size // hvd.size(), 1),
                        device=torch.cuda.current_device())
    weights.requires_grad = False
    logits = net(dense_features, sharded_sparse_features)

    net = torch.jit.trace(net,
                          example_inputs=(dense_features,
                                          sharded_sparse_features))

    if fp16:
        net = apex.amp.initialize(net, opt_level=FP16_LEVEL, verbosity=0)
        # net_fp16 = torch.jit.trace(net_fp16,
        #                            example_inputs=(dense_features,
        #                                            sharded_sparse_features))

    def forward(dense_features, partitioned_sparse_features):
        return net(
            dense_features,
            sharded_sparse_features.random_(0, num_embeddings))
    MPI.COMM_WORLD.Barrier()
    time_per_batch = benchmark_torch_function(iters, forward, dense_features,
                                              sharded_sparse_features)
    print(json.dumps(dict(name=name, method="forward", implementation="partitioned", rank=hvd.rank(), workers=hvd.size(), B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, fp16=fp16, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))
    if hvd.rank() == 0:
        logging.info(
            f"{name}, PARTITIONED, FORWARD: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )

    criterion = Criterion().cuda()
    criterion = torch.jit.trace(criterion, example_inputs=(logits, labels, weights))

    def forward_backward(dense_features, sharded_sparse_features, labels):
        logits = net(
            dense_features,
            sharded_sparse_features.random_(0, num_embeddings))
        loss = criterion(logits, labels, weights)
        loss.backward()
    MPI.COMM_WORLD.Barrier()
    time_per_batch = benchmark_torch_function(iters, forward_backward,
                                              dense_features,
                                              sharded_sparse_features, labels)
    print(json.dumps(dict(name=name, method="forwardbackward", implementation="partitioned", rank=hvd.rank(), workers=hvd.size(), B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, fp16=fp16, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))

    if hvd.rank() == 0:
        logging.info(
            f"{name}, PARTITIONED, FORWARDBACKWARD: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )

    # optimizer = apex.optimizers.FusedSGD([v for (_, v) in dense_named_parameters], lr=0.05)
    net = DistributedPartitionShardedSNN(num_tables, num_embeddings,
                                        embedding_dim,
                                        dense_features_dim).cuda()
    dense_named_parameters = [(k, v) for (k, v) in net.named_parameters()
                              if "embedding" not in k]
    optimizer = FastZeroFusedSGD([v for (_, v) in dense_named_parameters],
                                 lr=0.05)

    if fp16:
        net, optimizer = apex.amp.initialize(net,
                                             optimizer,
                                             opt_level=FP16_LEVEL,
                                             verbosity=0)
        # net = torch.jit.trace(net,
        #                       example_inputs=(dense_features,
        #                                       sharded_sparse_features))
        optimizer.zero_grad = optimizer.amp_zero_grad

    net.dense_arch = SingleGPUDDP(net.dense_arch,
                                  device_ids=[torch.cuda.current_device()])
    net.over_arch = SingleGPUDDP(net.over_arch,
                                  device_ids=[torch.cuda.current_device()])

    def forward_backward_update(dense_features, sharded_sparse_features,
                                labels):
        optimizer.zero_grad()
        logits = net(
            dense_features,
            sharded_sparse_features.random_(0, num_embeddings))
        loss = criterion(logits, labels, weights)
        loss.backward()
        optimizer.step()

    MPI.COMM_WORLD.Barrier()

    import os
    if os.environ.get('BIGADS_PROFILE') and hvd.rank() == 0:
        with torch.autograd.profiler.profile(use_cuda=True,
                                             record_shapes=True) as prof:
            time_per_batch = benchmark_torch_function(iters,
                                                      forward_backward_update,
                                                      dense_features,
                                                      sharded_sparse_features,
                                                      labels)
        prof.export_chrome_trace(f"{'fp16' if fp16 else 'fp32'}-{hvd.size()}-partitioned_{os.environ.get('BIGADS_PROFILE')}")
    else:
        time_per_batch = benchmark_torch_function(iters,
                                                  forward_backward_update,
                                                  dense_features,
                                                  sharded_sparse_features,
                                                  labels)
    print(json.dumps(dict(name=name, method="forwardbackwardupdate", implementation="partitioned", rank=hvd.rank(), workers=hvd.size(), B=batch_size, L=bag_size, T=num_tables, D=embedding_dim, dense_D=dense_features_dim, fp16=fp16, time_per_batch=time_per_batch, qps=batch_size * 1.0 / time_per_batch)))
    if hvd.rank() == 0:
        logging.info(
            f"{name}, PARTITIONED, FORWARDBACKWARDUPDATE: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, Bag Size: {bag_size}, Num Tables: {num_tables}, Embedding Dim: {embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )



def div_round_up(a, b):
    return int((a + b - 1) // b) * b


@click.command()
@click.option("--num-tables", default=64)
@click.option("--num-embeddings", default=1e4)
@click.option("--embedding-dim", default=32)
@click.option("--dense-features-dim", default=512)
@click.option("--batch-size", default=128)
@click.option("--bag-size", default=32)
@click.option("--iters", default=100)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--weak-scaling", is_flag=True, default=False)
def cli(num_tables, num_embeddings, embedding_dim, dense_features_dim,
        batch_size, bag_size, iters, fp16, weak_scaling):
    fp16 = int(fp16)
    hvd.init(comm=MPI.COMM_WORLD.Dup())
    import socket
    import random
    ip = socket.gethostbyname(socket.gethostname())
    # TODO: less hacky
    port = random.randint(20000, 60000)

    (master_ip, master_port) = MPI.COMM_WORLD.bcast((ip, port), root=0)
    MPI.COMM_WORLD.Barrier()
    dist.init_process_group("nccl",
                            init_method=f"file:///private/home/tulloch/src/bigads_{master_ip}_{master_port}.rendevouz",
                            rank=hvd.rank(),
                            world_size=hvd.size())
    logging.info(
        f"Horovod initialized: size={hvd.size()}, rank={hvd.rank()}, local_rank={hvd.local_rank()}"
    )
    torch.cuda.set_device(hvd.local_rank())

    num_tables = div_round_up(num_tables, hvd.size())
    num_embeddings = div_round_up(num_embeddings, hvd.size())
    batch_size = div_round_up(batch_size, hvd.size())
    bag_size = div_round_up(bag_size, hvd.size())
    if weak_scaling:
        batch_size = batch_size * hvd.size()

    name = "uniform-strong" if not weak_scaling else "uniform-weak"
    benchmark_torch_uniform_snn_forward(name, num_tables, num_embeddings,
                                        embedding_dim, dense_features_dim,
                                        batch_size, bag_size, iters, fp16=fp16)

    name = "partitioned-strong" if not weak_scaling else "partitioned-weak"
    benchmark_torch_partitioned_snn_forward(name, num_tables, num_embeddings,
                                embedding_dim, dense_features_dim,
                                batch_size, bag_size, iters, fp16=fp16)

if __name__ == "__main__":
    cli()