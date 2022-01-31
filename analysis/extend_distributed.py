# BSD 3-Clause License
#
# Copyright (c) 2021, The Regents of the University of California, Davis
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copied from the Facebook DLRM repo


import builtins, os, sys
import torch
import torch.distributed as dist


my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def init_distributed(rank=-1, local_rank=-1, size=-1, use_gpu=False, backend=""):
    global my_rank
    global my_size
    global my_local_rank
    global my_local_size

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(
        ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"]
    )
    if backend == "" and num_mpi_ranks > 1:
        if use_gpu and dist.is_nccl_available():
            backend = "nccl"
        elif dist.is_mpi_available():
            backend = "mpi"
        else:
            print(
                "WARNING: MPI multi-process launch detected but PyTorch MPI backend not available."
            )
            backend = "gloo"

    if backend != "":
        # guess Rank and size
        if rank == -1:
            rank = env2int(
                ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "RANK"], 0
            )
        if size == -1:
            size = env2int(
                [
                    "PMI_SIZE",
                    "OMPI_COMM_WORLD_SIZE",
                    "MV2_COMM_WORLD_SIZE",
                    "WORLD_SIZE",
                ],
                1,
            )
        if not os.environ.get("RANK", None) and rank != -1:
            os.environ["RANK"] = str(rank)
        if not os.environ.get("WORLD_SIZE", None) and size != -1:
            os.environ["WORLD_SIZE"] = str(size)
        if not os.environ.get("MASTER_PORT", None):
            os.environ["MASTER_PORT"] = "29500"
        if not os.environ.get("MASTER_ADDR", None):
            local_size = env2int(
                [
                    "MPI_LOCALNRANKS",
                    "OMPI_COMM_WORLD_LOCAL_SIZE",
                    "MV2_COMM_WORLD_LOCAL_SIZE",
                ],
                1,
            )
            if local_size != size and backend != "mpi":
                print(
                    "Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default"
                )
                print(
                    "If this run hangs, try exporting rank 0's hostname as MASTER_ADDR"
                )
            os.environ["MASTER_ADDR"] = "127.0.0.1"

    if size > 1:
        if local_rank == -1:
            my_local_rank = env2int(
                [
                    "MPI_LOCALRANKID",
                    "OMPI_COMM_WORLD_LOCAL_RANK",
                    "MV2_COMM_WORLD_LOCAL_RANK",
                    "LOCAL_RANK",
                ],
                0,
            )
        else:
            my_local_rank = local_rank
        my_local_size = env2int(
            [
                "MPI_LOCALNRANKS",
                "OMPI_COMM_WORLD_LOCAL_SIZE",
                "MV2_COMM_WORLD_LOCAL_SIZE",
            ],
            1,
        )
        if use_gpu:
            if my_local_size > torch.cuda.device_count():
                print(
                    "Not sufficient GPUs available... local_size = %d, ngpus = %d"
                    % (my_local_size, torch.cuda.device_count())
                )
                sys.exit(1)
            torch.cuda.set_device(my_local_rank)
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        if my_rank == 0:
            print("Running on %d ranks using %s backend" % (my_size, backend))
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1
    print_all(
        "world size: %d, current rank: %d, local rank: %d"
        % (my_size, my_rank, my_local_rank)
    )


def barrier():
    if my_size > 1:
        dist.barrier()


# Override builtin print function to print only from rank 0
orig_print = builtins.print


def rank0_print(*args, **kwargs):
    if my_rank <= 0 or kwargs.get("print_all", False):
        orig_print(*args, **kwargs)


builtins.print = rank0_print
# Allow printing from all rank with explicit print_all
def print_all(*args, **kwargs):
    orig_print(*args, **kwargs)


# For gathering results from different ranks
def all_gather(**kwargs):
    work = dist.all_gather(
        kwargs['opTensorList'], # tensor list
        kwargs['ipTensor'],
        async_op=True
    )
    work.wait()
