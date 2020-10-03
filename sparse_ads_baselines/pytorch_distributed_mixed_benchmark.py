import os

if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
print(os.environ["CUDA_VISIBLE_DEVICES"])
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
import os
import torch
from torch import nn, Tensor
from typing import List

import horovod.torch as hvd
import apex
from sparse_embedding_cuda_ops import (
    UniformShardedEmbeddingBags,
    FastZeroFusedSGD,
    ReduceScatterFunction,
    All2AllFunction,
    EmbeddingLocation,
)

from mpi4py import MPI
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import logging

logging.basicConfig(level=logging.DEBUG)

FP16_LEVEL = "O3"

from contextlib import contextmanager




class MixedShardedSNN(nn.Module):
    def __init__(
        self,
        gpu_num_tables,
        gpu_num_embeddings,
        gpu_embedding_dim,
        cpu_num_tables,
        cpu_num_embeddings,
        cpu_embedding_dim,
        dense_features_dim,
        over_dim,
        gpu_communicator,
        cpu_communicator,
        fp16,
    ):
        super(MixedShardedSNN, self).__init__()
        self.dense_arch = nn.Sequential(
            nn.Linear(dense_features_dim, over_dim),
            nn.ReLU(),
            nn.Linear(over_dim, gpu_embedding_dim),
        ).cuda()
        self.gpu_embedding_tables = UniformShardedEmbeddingBags(
            gpu_num_tables,
            gpu_num_embeddings,
            gpu_embedding_dim,
            managed=EmbeddingLocation.DEVICE,
            fp16=fp16,
        ).cuda()
        logging.info(
            f"GPU embeddings: {gpu_num_tables}, {gpu_num_embeddings}, {gpu_embedding_dim}"
        )
        self.cpu_embedding_tables = UniformShardedEmbeddingBags(
            cpu_num_tables,
            cpu_num_embeddings,
            cpu_embedding_dim,
            managed=EmbeddingLocation.HOST_MAPPED,
            fp16=fp16,
        ).cuda()
        logging.info(
            f"CPU embeddings: {cpu_num_tables}, {cpu_num_embeddings}, {cpu_embedding_dim}"
        )
        torch.cuda.synchronize()
        logging.info(
            f"CPU embeddings initialized: {cpu_num_tables}, {cpu_num_embeddings}, {cpu_embedding_dim}"
        )
        in_feature_dim = (
            gpu_embedding_dim * gpu_num_tables * hvd.size()
            + cpu_embedding_dim * cpu_num_tables
            + gpu_embedding_dim
        )
        self.over_arch = nn.Sequential(
            nn.Linear(in_feature_dim, over_dim),
            nn.ReLU(),
            nn.Linear(over_dim, over_dim),
            nn.ReLU(),
            nn.Linear(over_dim, over_dim),
            nn.ReLU(),
            nn.Linear(over_dim, over_dim),
            nn.ReLU(),
            nn.Linear(over_dim, 1),
        ).cuda()
        self.gpu_communicator = gpu_communicator
        self.cpu_communicator = cpu_communicator

    def forward(
        self, dense_features, gpu_sharded_sparse_features, cpu_sharded_sparse_features
    ):
        gpu_embeddings = self.gpu_communicator(self.gpu_embedding_tables(gpu_sharded_sparse_features))
        # (b / w, t, d)
        cpu_embeddings = self.cpu_embedding_tables(cpu_sharded_sparse_features)
        cpu_embeddings = cpu_embeddings.view(cpu_embeddings.shape[0] * hvd.size(), cpu_embeddings.shape[1], cpu_embeddings.shape[2] // hvd.size())
        # (b, t, d / w)
        cpu_embeddings = self.cpu_communicator(cpu_embeddings)
        dense_embeddings = self.dense_arch(dense_features)
        embeddings = torch.cat(
            [
                gpu_embeddings.flatten(start_dim=1),
                cpu_embeddings.flatten(start_dim=1),
                dense_embeddings,
            ],
            dim=1,
        )

        logits = self.over_arch(embeddings)
        return logits


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

def benchmark_torch_mixed_snn(
    name,
    gpu_num_tables,
    gpu_num_embeddings,
    gpu_embedding_dim,
    cpu_num_tables,
    cpu_num_embeddings,
    cpu_embedding_dim,
    dense_features_dim,
    over_dim,
    batch_size,
    gpu_bag_size,
    cpu_bag_size,
    iters,
    fp16,
):  
    local_gpu_num_tables = gpu_num_tables // hvd.size()
    local_gpu_bag_size = gpu_bag_size
    local_gpu_num_embeddings = gpu_num_embeddings
    net = MixedShardedSNN(
        local_gpu_num_tables,
        local_gpu_num_embeddings,
        gpu_embedding_dim,
        cpu_num_tables,
        cpu_num_embeddings // hvd.size(),
        cpu_embedding_dim,
        dense_features_dim,
        over_dim,
        gpu_communicator=All2AllFunction.apply,
        cpu_communicator=All2AllFunction.apply,
        fp16=fp16,
    ).cuda()

    logging.info(
        f"Model size (Gparameters): {sum(p.numel() for p in net.parameters()) / 1024 / 1024 / 1024}"
    )

    dense_features = torch.randn(
        batch_size // hvd.size(), dense_features_dim, device=torch.cuda.current_device()
    )

    dense_features.requires_grad = False
    gpu_sharded_sparse_features = torch.randint(
        low=0,
        high=local_gpu_num_embeddings,
        size=(batch_size, local_gpu_num_tables, local_gpu_bag_size),
        device=torch.cuda.current_device(),
    ).int()
    gpu_sharded_sparse_features.requires_grad = False

    cpu_sharded_sparse_features = torch.randint(
        low=0,
        high=cpu_num_embeddings // hvd.size(),
        size=(batch_size // hvd.size(), cpu_num_tables, cpu_bag_size),
        device=torch.cuda.current_device(),
    ).int()
    cpu_sharded_sparse_features.requires_grad = False

    labels = torch.rand(
        size=(batch_size // hvd.size(), 1), device=torch.cuda.current_device()
    )

    if fp16:
        # 'fake' fp16 training
        net = net.half()
        dense_features = dense_features.half()
        # net.dense_arch = apex.amp.initialize(
        #     net.dense_arch, opt_level=FP16_LEVEL, verbosity=1,
        # )
        # net.over_arch = apex.amp.initialize(
        #     net.over_arch, opt_level=FP16_LEVEL, verbosity=1,
        # )

        # net_fp16 = torch.jit.trace(net_fp16,
        #                            example_inputs=(dense_features,
        #                                            sharded_sparse_features))

    logits = net(
        dense_features, gpu_sharded_sparse_features, cpu_sharded_sparse_features
    )
    # net = torch.jit.trace(
    #     net,
    #     example_inputs=(
    #         dense_features,
    #         gpu_sharded_sparse_features,
    #         cpu_sharded_sparse_features,
    #     ),
    # )


    def forward(
        dense_features, gpu_sharded_sparse_features, cpu_sharded_sparse_features
    ):
        return net(
            dense_features,
            gpu_sharded_sparse_features.random_(0, local_gpu_num_embeddings),
            cpu_sharded_sparse_features.random_(0, cpu_num_embeddings // hvd.size()),
        )

    logging.info("Initialized")
    MPI.COMM_WORLD.Barrier()
    logging.info("Starting benchmark")

    if os.environ.get("BIGADS_PROFILE") and hvd.rank() == 0:
        with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
            time_per_batch = benchmark_torch_function(
                iters,
                forward,
                dense_features,
                gpu_sharded_sparse_features,
                cpu_sharded_sparse_features,
            )
        prof.export_chrome_trace(
            f"{'fp16' if fp16 else 'fp32'}-{hvd.size()}-forward_{os.environ.get('BIGADS_PROFILE')}"
        )
    else:
        time_per_batch = benchmark_torch_function(
            iters,
            forward,
            dense_features,
            gpu_sharded_sparse_features,
            cpu_sharded_sparse_features,
        )

    print(
        json.dumps(
            dict(
                name=name,
                method="forward",
                implementation="table-partition (gpu), row-partition (cpu)",
                rank=hvd.rank(),
                workers=hvd.size(),
                B=batch_size,
                L_gpu=gpu_bag_size,
                L_cpu=cpu_bag_size,
                T_gpu=gpu_num_tables,
                T_cpu=cpu_num_tables,
                D_gpu=gpu_embedding_dim,
                D_cpu=cpu_embedding_dim,
                D_dense=dense_features_dim,
                over_dim=over_dim,
                fp16=fp16,
                time_per_batch=time_per_batch,
                qps=batch_size * 1.0 / time_per_batch,
            )
        )
    )
    if hvd.rank() == 0:
        logging.info(
            f"{name}, PARTITIONED, FORWARD: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, GPU Bag Size: {gpu_bag_size}, CPU bag size: {cpu_bag_size}, GPU Num Tables: {gpu_num_tables}, CPU Num Tables: {cpu_num_tables}, GPU Embedding Dim: {gpu_embedding_dim}, CPU Embedding Dim: {cpu_embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )

    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = torch.jit.trace(criterion, example_inputs=(logits, labels))

    def forward_backward(
        dense_features, gpu_sharded_sparse_features, cpu_sharded_sparse_features, labels
    ):
        logits = net(
            dense_features,
            gpu_sharded_sparse_features.random_(0, local_gpu_num_embeddings),
            cpu_sharded_sparse_features.random_(0, cpu_num_embeddings // hvd.size()),
        )
        loss = criterion(logits, labels)
        loss.backward()

    MPI.COMM_WORLD.Barrier()

    time_per_batch = benchmark_torch_function(
        iters,
        forward_backward,
        dense_features,
        gpu_sharded_sparse_features,
        cpu_sharded_sparse_features,
        labels,
    )

    print(
        json.dumps(
            dict(
                name=name,
                method="forwardbackward",
                implementation="table-partition (gpu), row-partition (cpu)",
                rank=hvd.rank(),
                workers=hvd.size(),
                B=batch_size,
                L_gpu=gpu_bag_size,
                L_cpu=cpu_bag_size,
                T_gpu=gpu_num_tables,
                T_cpu=cpu_num_tables,
                D_gpu=gpu_embedding_dim,
                D_cpu=cpu_embedding_dim,
                D_dense=dense_features_dim,
                over_dim=over_dim,
                fp16=fp16,
                time_per_batch=time_per_batch,
                qps=batch_size * 1.0 / time_per_batch,
            )
        )
    )

    if hvd.rank() == 0:
        logging.info(
            f"{name}, PARTITIONED, FORWARDBACKWARD: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, GPU Bag Size: {gpu_bag_size}, CPU bag size: {cpu_bag_size}, GPU Num Tables: {gpu_num_tables}, CPU Num Tables: {cpu_num_tables}, GPU Embedding Dim: {gpu_embedding_dim}, CPU Embedding Dim: {cpu_embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )

    dense_named_parameters = [
        (k, v) for (k, v) in net.named_parameters() if "embedding" not in k
    ]
    optimizer = FastZeroFusedSGD([v for (_, v) in dense_named_parameters], lr=0.05)

    if fp16:
        # net, optimizer = apex.amp.initialize(
        #     net, optimizer, opt_level=FP16_LEVEL, verbosity=0
        # )
        # net = torch.jit.trace(net,
        #                       example_inputs=(dense_features,
        #                                       sharded_sparse_features))
        # optimizer.zero_grad = optimizer.amp_zero_grad
        pass

    net.dense_arch = SingleGPUDDP(
        net.dense_arch, device_ids=[torch.cuda.current_device()]
    )
    net.over_arch = SingleGPUDDP(
        net.over_arch, device_ids=[torch.cuda.current_device()]
    )

    def forward_backward_update(
        dense_features, gpu_sharded_sparse_features, cpu_sharded_sparse_features, labels
    ):
        optimizer.zero_grad()
        logits = net(
            dense_features,
            gpu_sharded_sparse_features.random_(0, local_gpu_num_embeddings),
            cpu_sharded_sparse_features.random_(0, cpu_num_embeddings // hvd.size()),
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    MPI.COMM_WORLD.Barrier()

    if os.environ.get("BIGADS_PROFILE") and hvd.rank() == 0:
        with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
            time_per_batch = benchmark_torch_function(
                iters,
                forward_backward_update,
                dense_features,
                gpu_sharded_sparse_features,
                cpu_sharded_sparse_features,
                labels,
            )
        prof.export_chrome_trace(
            f"{'fp16' if fp16 else 'fp32'}-{hvd.size()}-forwardbackwardupdate_{os.environ.get('BIGADS_PROFILE')}"
        )
    else:
        time_per_batch = benchmark_torch_function(
            iters,
            forward_backward_update,
            dense_features,
            gpu_sharded_sparse_features,
            cpu_sharded_sparse_features,
            labels,
        )
    print(
        json.dumps(
            dict(
                name=name,
                method="forwardbackwardupdate",
                implementation="table-partition (gpu), row-partition (cpu)",
                rank=hvd.rank(),
                workers=hvd.size(),
                B=batch_size,
                L_gpu=gpu_bag_size,
                L_cpu=cpu_bag_size,
                T_gpu=gpu_num_tables,
                T_cpu=cpu_num_tables,
                D_gpu=gpu_embedding_dim,
                D_cpu=cpu_embedding_dim,
                D_dense=dense_features_dim,
                over_dim=over_dim,
                fp16=fp16,
                time_per_batch=time_per_batch,
                qps=batch_size * 1.0 / time_per_batch,
            )
        )
    )
    if hvd.rank() == 0:
        logging.info(
            f"{name}, PARTITIONED, FORWARDBACKWARDUPDATE: rank={hvd.rank()}, Workers: {hvd.size()}, Batch Size: {batch_size}, GPU Bag Size: {gpu_bag_size}, CPU bag size: {cpu_bag_size}, GPU Num Tables: {gpu_num_tables}, CPU Num Tables: {cpu_num_tables}, GPU Embedding Dim: {gpu_embedding_dim}, CPU Embedding Dim: {cpu_embedding_dim}, fp16: {fp16}, Time per batch: {time_per_batch * 1.0e6}us, QPS: {batch_size * 1.0 / time_per_batch:.2e}"
        )


def div_round_up(a, b):
    return int((a + b - 1) // b) * b


@click.command()
@click.option("--gpu-embedding-gb-per-rank", default=1, type=float)
@click.option("--gpu-embedding-dim", default=64)
@click.option("--gpu-num-tables", default=96)
@click.option("--gpu-bag-size", default=14)
@click.option("--cpu-embedding-gb-per-rank", default=1, type=float)
@click.option("--cpu-embedding-dim", default=288)
@click.option("--cpu-num-tables", default=2)
@click.option("--cpu-bag-size", default=1)
@click.option("--dense-features-dim", default=200)
@click.option("--over-dim", default=2048)
@click.option("--batch-size", default=2 * 1024)
@click.option("--iters", default=100)
@click.option("--fp16", is_flag=True, default=False)
def cli(
    gpu_embedding_gb_per_rank,
    gpu_embedding_dim,
    gpu_num_tables,
    gpu_bag_size,
    cpu_embedding_gb_per_rank,
    cpu_embedding_dim,
    cpu_num_tables,
    cpu_bag_size,
    dense_features_dim,
    over_dim,
    batch_size,
    iters,
    fp16,
):

    fp16 = int(fp16)
    hvd.init(comm=MPI.COMM_WORLD.Dup())
    import socket
    import random

    ip = socket.gethostbyname(socket.gethostname())
    # TODO: less hacky
    port = random.randint(20000, 60000)

    (master_ip, master_port) = MPI.COMM_WORLD.bcast((ip, port), root=0)
    MPI.COMM_WORLD.Barrier()
    dist.init_process_group(
        "nccl",
        init_method=f"file:///private/home/tulloch/src/bigads_{master_ip}_{master_port}.rendevouz",
        rank=hvd.rank(),
        world_size=hvd.size(),
    )
    logging.info(
        f"Horovod initialized: size={hvd.size()}, rank={hvd.rank()}, local_rank={hvd.local_rank()}"
    )
    torch.cuda.set_device(0)  # hvd.local_rank())
    elem_size = 4 if not fp16 else 2
    cpu_num_embeddings = div_round_up(
        cpu_embedding_gb_per_rank
        * hvd.size()
        * 1024
        * 1024
        * 1024
        / (elem_size * cpu_num_tables * cpu_embedding_dim),
        hvd.size(),
    )
    cpu_embedding_dim = div_round_up(cpu_embedding_dim, hvd.size())
    gpu_num_tables = div_round_up(gpu_num_tables, hvd.size())

    gpu_num_embeddings = int(
        (
            gpu_embedding_gb_per_rank
            * 1024
            * 1024
            * 1024
            / (elem_size * (gpu_num_tables // hvd.size()) * gpu_embedding_dim)
        )
    )

    batch_size = div_round_up(batch_size, hvd.size())

    name = "mixed-gpu-cpu"
    benchmark_torch_mixed_snn(
        name,
        gpu_num_tables,
        gpu_num_embeddings,
        gpu_embedding_dim,
        cpu_num_tables,
        cpu_num_embeddings,
        cpu_embedding_dim,
        dense_features_dim,
        over_dim,
        batch_size,
        gpu_bag_size,
        cpu_bag_size,
        iters=iters,
        fp16=fp16,
    )


if __name__ == "__main__":
    cli()
