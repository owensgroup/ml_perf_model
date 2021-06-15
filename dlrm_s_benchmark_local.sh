#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Get DLRM model name
model_name=${1:-DLRM_default}
list="DLRM_vipul DLRM_default MLPerf"
if [[ $list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi

# check if extra argument is passed to the test
if [[ $# == 2 ]]; then
    dlrm_extra_option=$2
else
    dlrm_extra_option=""
fi
# echo $dlrm_extra_option

cpu=0
gpu=1
ncores=8
nsockets="0"
ngpus="1" #"1 2 4"
common_args="   --use-gpu\
                --enable-profiling\
                --print-freq=5\
                --print-time\
                --batched-emb\
                --pin-memory "

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm/dlrm_s_pytorch.py" # fil-profile run

# Get GPU type
nvidia-smi --query-gpu=gpu_name --format=csv,noheader > /tmp/gpu_name.csv
if grep -q "V100" /tmp/gpu_name.csv
then
    export GPU_NAME="V100"
elif grep -q "P100" /tmp/gpu_name.csv
then
    export GPU_NAME="P100"
elif grep -q "Xp" /tmp/gpu_name.csv
then
    export GPU_NAME="Xp"
elif grep -q "1080" /tmp/gpu_name.csv
then
    export GPU_NAME="1080"
else
    echo "Unrecognized GPU name! Exit..."
    exit
fi

# ----------------------- Model param -----------------------
mb_size=2048
_args=""
if [[ $model_name == "DLRM_vipul" ]]; # From Vipul
then
    mb_size=256 #2048 #1024 #512 #256
    num_batches=50
    _args=" --data-generation=random\
            --arch-mlp-bot=13-512-256-64-16\
            --arch-mlp-top=512-256-128-1\
            --arch-sparse-feature-size=16\
            --arch-embedding-size=1461-584-10131227-2202608-306-24-12518-634-4-93146-5684-8351593-3195-28-14993-5461306-11-5653-2173-4-7046547-18-16-286181-105-142572\
            --num-indices-per-lookup=38\
            --arch-interaction-op=dot\
            --numpy-rand-seed=727\
            --num-workers=2 "
elif [[ $model_name == "DLRM_default" ]]; # DLRM original
then
    mb_size=2048 #1024 #512 #256
    num_batches=100
    _args=" --data-generation=random\
            --processed-data-file=/nvme/deep-learning/dlrm_random\
            --round-targets\
            --arch-mlp-bot=512-512-64\
            --arch-mlp-top=1024-1024-1024-1\
            --arch-sparse-feature-size=64\
            --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000\
            --num-indices-per-lookup=100\
            --num-indices-per-lookup-fixed\
            --arch-interaction-op=dot\
            --numpy-rand-seed=727\
            --num-worker=0 "
elif [[ $model_name == "MLPerf" ]]; # MLPerf
then
    mb_size=2048
    num_batches=100
    _args=" --data-generation=dataset\
            --data-set=terabyte\
            --raw-data-file=/nvme/deep-learning/criteo_terabyte/day\
            --processed-data-file=/nvme/deep-learning/criteo_terabyte/terabyte_processed.npz\
            --round-targets\
            --arch-mlp-bot=13-512-256-128\
            --arch-mlp-top=1024-1024-512-256-1\
            --arch-sparse-feature-size=128\
            --max-ind-range=40000000\
            --loss-function=bce\
            --learning-rate=1.0\
            --test-freq=102400\
            --memory-map\
            --mlperf-logging\
            --mlperf-auc-threshold=0.8025\
            --mlperf-bin-loader\
            --mlperf-bin-shuffle "
        # " --test-num-workers=16"
        # " --test-mini-batch-size=16384"
fi

interaction="dot"
tnworkers=0
tmb_size=-1 #256

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    rm -f /tmp/pytorch_execution_graph*
    # # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    outf="data/${GPU_NAME}/e2e/${model_name}_${_ng}.log"
    outp="dlrm_s_pytorch.prof"
    echo "-------------------------------"
    echo "Running PT (log file: $outf)"
    echo "-------------------------------"
    cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers ${common_args} ${_args} $dlrm_extra_option"
    if [ ! -f "data/${GPU_NAME}/e2e/${model_name}_${_ng}_graph.json" ];
    then
      echo "Execution graph doesn't exist! Extract it..."
      eval "$cmd --num-batches 1 --collect-execution-graph &> /dev/null" # Collect execution graph
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "data/${model_name}_${_ng}_graph.json"
    fi
    eval "$cmd --num-batches ${num_batches} > $outf"
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    echo "Min time per iteration = $min"
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
  done
fi
