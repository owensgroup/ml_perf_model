#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
# echo $dlrm_extra_option

cpu=0
gpu=1
pt=1
ncores=8
nsockets="0"
ngpus="1" #"1 2 4"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm/dlrm_s_pytorch.py"

data=random #synthetic
print_freq=5 #100
rand_seed=727

# Model param

# B
mb_size=64 #2048 #1024 #512 #256
# num_batches
nbatches=10 #1000 #500 #100
# bottom_mlp
bot_mlp="13-512-256-64-16"
# top_mlp
top_mlp="512-256-128-1"
# E & T
emb="1461-584-10131227-2202608-306-24-12518-634-4-93146-5684-8351593-3195-28-14993-5461306-11-5653-2173-4-7046547-18-16-286181-105-142572"
# L
nindices=38 #100
# D
emb_size=16

interaction="dot"
tnworkers=0
tmb_size=256

_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"\
" --enable-profiling "


# CPU Benchmarking
if [ $cpu = 1 ]; then
  echo "--------------------------------------------"
  echo "CPU Benchmarking - running on $ncores cores"
  echo "--------------------------------------------"
  if [ $pt = 1 ]; then
    outf="data/model1_CPU_PT_$ncores.log"
    outp="dlrm_s_pytorch.prof"
    echo "-------------------------------"
    echo "Running PT (log file: $outf)"
    echo "-------------------------------"
    cmd="$numa_cmd $dlrm_pt_bin --mini-batch-size=$mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args $dlrm_extra_option > $outf"
    echo $cmd
    eval $cmd
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    echo "Min time per iteration = $min"
    # # move profiling file(s)
    # mv $outp ${outf//".log"/".prof"}
    # mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
  fi
fi

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    if [ $pt = 1 ]; then
      outf="data/model1_GPU_PT_${_ng}.log"
      outp="dlrm_s_pytorch.prof"
      echo "-------------------------------"
      echo "Running PT (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers ${_args} --use-gpu $dlrm_extra_option > $outf"
      echo $cmd
      eval $cmd
      min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file(s)
      mv $outp ${outf//".log"/".prof"}
      mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    fi
  done
fi
