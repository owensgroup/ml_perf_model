import click

import torch
from torch import nn, Tensor
from typing import List

import horovod.torch as hvd
import apex
from sparse_embedding_cuda_ops import UniformShardedEmbeddingBags, FastZeroFusedSGD
from models import SNN, UniformShardedSNN, DistributedUniformShardedSNN, DistributedPartitionShardedSNN
from parameterized import parameterized


@parameterized([(5, int(1e5), 32, 64, 8, 32), (2, int(1e5), 7, 7, 2, 17)])
def test_torch_snn_forward(num_tables, num_embeddings, embedding_dim,
                           dense_features_dim, batch_size, bag_size):
    hvd.init()
    assert hvd.size() == 1      
    net = SNN(num_tables, num_embeddings, embedding_dim, dense_features_dim).cuda()
    net_s = UniformShardedSNN(num_tables, num_embeddings, embedding_dim,
                              dense_features_dim).cuda()
    net_ds = DistributedUniformShardedSNN(
        num_tables, num_embeddings, embedding_dim,
        dense_features_dim).cuda()
    net_ps = DistributedPartitionShardedSNN(
        num_tables, num_embeddings, embedding_dim,
        dense_features_dim).cuda()

    net_s.load_state_dict(net.state_dict(), strict=False)
    net_ds.load_state_dict(net.state_dict(), strict=False)
    net_ps.load_state_dict(net.state_dict(), strict=False)

    for t in range(num_tables):
        net_s.embedding_tables.embedding_weights.data[:,
                                                      t, :] = net.embedding_tables[
                                                          t].weight
        net_ds.embedding_tables.embedding_weights.data[:,
                                                      t, :] = net.embedding_tables[
                                                          t].weight
        net_ps.embedding_tables.embedding_weights.data[:,
                                                      t, :] = net.embedding_tables[
                                                          t].weight

    dense_features = torch.randn(batch_size, dense_features_dim).cuda()
    sparse_features = [
        torch.randint(low=0,
                      high=embedding_dim - 1,
                      size=(batch_size, bag_size)).cuda()
        for _ in range(num_tables)
    ]
    sharded_sparse_features = torch.cat(
        [sf.view(batch_size, 1, bag_size) for sf in sparse_features], dim=1)
    logits = net(dense_features, sparse_features)
    logits_s = net_s(dense_features, sharded_sparse_features)
    logits_ds = net_ds(dense_features, sharded_sparse_features)
    logits_ps = net_ds(dense_features, sharded_sparse_features)

    torch.testing.assert_allclose(logits, logits_s)
    torch.testing.assert_allclose(logits, logits_ds)
    torch.testing.assert_allclose(logits, logits_ps)
