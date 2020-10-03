import torch
from torch import nn, Tensor
from typing import List

from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
import apex
from ctypes import cdll
lib1 = cdll.LoadLibrary('libnccl.so.2')

import sparse_embedding_cuda
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import amp_C
import apex
import horovod.torch as hvd

import sys

import logging


class LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices, offsets=None):
        ctx.save_for_backward(weights, indices, offsets)
        if offsets is None:
            return sparse_embedding_cuda.forward_fast_single(weights, indices)
        else:
            return sparse_embedding_cuda.forward_offsets(weights, indices, offsets)

    @staticmethod
    def backward(ctx, grad_output):
        weights, indices, offsets = ctx.saved_tensors
        # TODO: obvious hack
        LR = 0.05
        if offsets is None:
            sparse_embedding_cuda.backward_update_fast_single(
                grad_output, weights, indices, LR)
            return (torch.cuda.sparse.FloatTensor(*weights.size()), None, None)
        else:
            sparse_embedding_cuda.backward_update_offsets(
                grad_output, weights, indices, offsets, LR)
            return (torch.cuda.sparse.FloatTensor(*weights.size()), None, None)

import enum

class EmbeddingLocation(enum.Enum):
    DEVICE = 0
    MANAGED = 1
    HOST_MAPPED = 2

class UniformShardedEmbeddingBags(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim, managed=EmbeddingLocation.DEVICE, fp16=False):
        super(UniformShardedEmbeddingBags, self).__init__()
        # Whole tables (i.e. all rows for a table) are partitioned uniformly across devices
        import table_batched_embeddings
        # zeros is 2.5x faster than randn for initialization
        if managed == EmbeddingLocation.MANAGED:
            logging.info("Allocating managed embedding bag")
            embedding_data = torch.randn(size=(num_embeddings, num_tables, embedding_dim), 
                out=table_batched_embeddings.new_managed_tensor(torch.randn(1).cuda() if not fp16 else torch.randn(1).cuda().half(), (num_embeddings, num_tables, embedding_dim)))
        elif managed == EmbeddingLocation.HOST_MAPPED:
            logging.info("Allocating host mapped embedding bag")
            embedding_data = torch.randn(size=(num_embeddings, num_tables, embedding_dim), 
                out=table_batched_embeddings.new_managed_tensor(torch.randn(1).cuda() if not fp16 else torch.randn(1).cuda().half(), (num_embeddings, num_tables, embedding_dim)))
        elif managed == EmbeddingLocation.DEVICE:
            logging.info("Allocating device embedding bag")
            embedding_data = torch.randn(num_embeddings, num_tables, embedding_dim, device=torch.cuda.current_device(), dtype=torch.float16 if fp16 else torch.float32)
        self.embedding_weights = nn.Parameter(embedding_data)

    def forward(self, sharded_sparse_features, sharded_offsets=None):
        return LookupFunction.apply(self.embedding_weights,
                                    sharded_sparse_features, sharded_offsets)


class ReduceScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sharded_embeddings):
        return sparse_embedding_cuda.forward_reducescatter(sharded_embeddings)

    @staticmethod
    def backward(ctx, grad_output):
        return sparse_embedding_cuda.forward_allgather(grad_output)


class All2AllFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partitioned_embeddings):
        (B, T, D) = partitioned_embeddings.size()
        assert B % hvd.size() == 0
        return sparse_embedding_cuda.forward_all2all_nccl(
            partitioned_embeddings)

        # butterfly_embeddings = torch.empty(B // hvd.size(),
        #                                    T * hvd.size(),
        #                                    D,
        #                                    device=torch.cuda.current_device())
        # sparse_embedding_cuda.forward_all2all(partitioned_embeddings,
        #                                       butterfly_embeddings)
        # return butterfly_embeddings

    @staticmethod
    def backward(ctx, grad_output):
        # in: (B // hvd.size(), T * hvd.size(), D)
        # out: (B, T, D)
        # solution: transpose to (T * hvd.size(), B // hvd.size(), D), make contiguous
        # All2All to get (T, B, D)
        # Transpose to get (B, T, D)
        # (B_div_world_size, T_mul_world_size, D) = grad_output.size()
        # B = B_div_world_size * hvd.size()
        # T = T_mul_world_size // hvd.size()
        # grad_input = torch.empty(T, B, D, device=torch.cuda.current_device())
        grad_input = sparse_embedding_cuda.forward_all2all_nccl(
            grad_output.transpose(1, 0).contiguous())
        return grad_input.transpose(1, 0)
        # sparse_embedding_cuda.forward_all2all(
        #     grad_output.transpose(1, 0).contiguous(), grad_input)
        # return grad_input.transpose(1, 0)


class FastZeroFusedSGD(apex.optimizers.FusedSGD):
    def __init__(self, *args, **kwargs):
        super(FastZeroFusedSGD, self).__init__(*args, **kwargs)
        self._overflow_buf = torch.cuda.IntTensor([0])

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        grads = [
            p.grad for group in self.param_groups for p in group['params']
            if p.grad is not None
        ]
        if not grads:
            return
        for grad in grads:
            grad.detach_()
        apex.multi_tensor_apply.multi_tensor_applier(amp_C.multi_tensor_scale,
                                                     self._overflow_buf,
                                                     [grads, grads], 0.0)
    def amp_zero_grad(self):
        stash = self._amp_stash
        self._amp_lazy_init()
        # Zero the model grads.
        grads = [p.grad for group in [stash.all_fp16_params, stash.all_fp32_from_fp32_params] for p in group if p.grad is not None]
        if not grads:
            return
        for grad in grads:
            grad.detach_()

        apex.multi_tensor_apply.multi_tensor_applier(amp_C.multi_tensor_scale,
                                                     self._overflow_buf,
                                                     [grads, grads], 0.0)
        for param in self._amp_stash.all_fp32_from_fp16_params:
            param.grad = None