#!/bin/bash
#SBATCH --job-name=multiembed_gpu_perf   #The name you want the job to have
#SBATCH --nodes=4
#SBATCH --time=60:00    #The max minutes allowed for the job
#SBATCH --gres=gpu:volta:8
#SBATCH -p priority
#SBATCH --constraint volta32gb
#SBATCH --comment pyper

#SBATCH --ntasks-per-node=8
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
module purge
module load anaconda3
module load cuda/10.1 
module load cudnn/v7.6.5.32-cuda.10.1
module load NCCL/2.5.6-1-cuda.10.1
module load openmpi/4.0.2/gcc.7.4.0-cuda.10.1
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/pytorch-dper-dlrm-distributed-perf/lib/python3.7/site-packages/torch/lib/
# mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x HOROVOD_MPI_THREADS_DISABLE=1 --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_snn_forward.py small-weak
# mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x HOROVOD_MPI_THREADS_DISABLE=1 --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_snn_forward.py small-strong
# mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x HOROVOD_MPI_THREADS_DISABLE=1 --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_snn_forward.py large-weak
# mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x HOROVOD_MPI_THREADS_DISABLE=1 --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-



# Large model
mpirun -bind-to none -map-by slot --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_distributed_benchmark.py --batch-size 18432 --num-tables 576 --embedding-dim 64 --dense-features-dim 1024 --bag-size 192 --num-embeddings 10032 --iters 100
mpirun -bind-to none -map-by slot --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_distributed_benchmark.py --batch-size 480 --num-tables 576 --embedding-dim 64 --dense-features-dim 1024 --bag-size 192 --num-embeddings 10032 --iters 100 --weak-scaling

# # Small model
mpirun -bind-to none -map-by slot --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_distributed_benchmark.py --batch-size 31680 --num-tables 288 --embedding-dim 32 --dense-features-dim 512 --bag-size 96 --num-embeddings 100032 --iters 100
mpirun -bind-to none -map-by slot --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 /private/home/tulloch/.conda/envs/pytorch-dper-dlrm-distributed-perf/bin/python pytorch_distributed_benchmark.py --batch-size 480 --num-tables 288 --embedding-dim 32 --dense-features-dim 512 --bag-size 96 --num-embeddings 100032 --iters 100 --weak-scaling
