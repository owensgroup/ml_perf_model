#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Get DLRM model name
model_name=$1
list="DLRM_vipul DLRM_default DLRM_MLPerf DLRM_DDP"
if [[ $list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi
mb_size=$2
ngpus=$3

# check if extra argument is passed to the test
if [[ $# == 4 ]]; then
    dlrm_extra_option=$2
else
    dlrm_extra_option=""
fi
# echo $dlrm_extra_option

tnworkers=0
tmb_size=-1 #256
num_batches=500
common_args="   --use-gpu\
                --print-freq=5\
                --batched-emb\
                --print-time\
                --pin-memory "

# ----------------------- Model param -----------------------
_args=""
if [[ $model_name == "DLRM_vipul" ]]; # From Vipul
then
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
elif [[ $model_name == "DLRM_DDP" ]]; # DLRM DDP example
then
    _args=" --data-generation=random\
            --processed-data-file=/nvme/deep-learning/dlrm_random\
            --round-targets\
            --arch-mlp-bot=128-128-128-128\
            --arch-mlp-top=512-512-512-256-1\
            --arch-sparse-feature-size=128\
            --arch-embedding-size=80000-80000-80000-80000-80000-80000-80000-80000\
            --max-ind-range=40000000\
            --num-indices-per-lookup-fixed\
            --loss-function=bce\
            --learning-rate=1.0 "
elif [[ $model_name == "DLRM_MLPerf" ]]; # DLRM_MLPerf
then
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

# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - ${model_name} running on $ngpus GPUs"
echo "--------------------------------------------"
for _ng in $ngpus
do
  rm -f /tmp/pytorch_execution_graph*
  # # weak scaling
  # _mb_size=$((mb_size*_ng))
  # strong scaling
  _mb_size=$((mb_size*1))
  _gpus=$(seq -s, 0 $((_ng-1)))
  if [ ${_ng} = 1 ];
  then
    dlrm_pt_bin="python ${PM_HOME}/3rdparty/dlrm/dlrm_s_pytorch.py"
    graph_filename_pattern="${_ng}_${_mb_size}_graph.json"
  else
    dlrm_pt_bin="python -m torch.distributed.run --nproc_per_node=${_ng} ${PM_HOME}/3rdparty/dlrm/dlrm_s_pytorch.py --dist-backend=nccl "
    graph_filename_pattern="${_ng}_${_mb_size}_distributed_[0-$((_ng-1))]_graph.json"
  fi
  cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
  echo "-------------------"
  echo "Using GPUS: $_gpus, batch size: $_mb_size"
  echo "-------------------"
  mkdir -p "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}"
  cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers ${common_args} ${_args} $dlrm_extra_option"
  if [[ ${_ng} != `ls ${PM_HOME}/data/${GPU_NAME}/e2e/${model_name} | grep -e $graph_filename_pattern | wc -l` ]];
  then
    echo "Execution graph doesn't exist! Extract it..."
    eval "$cmd --num-batches 1 --collect-execution-graph --enable-profiling --test-freq=-1 &> /dev/null" # Collect execution graph
    if [ ${_ng} = 1 ];
    then
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${_mb_size}_graph.json"
    else
      count=0
      for g in `ls /tmp/pytorch_execution_graph*`
      do
        cp $g "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${_mb_size}_distributed_${count}_graph.json"
        count=$((count+1))
      done
    fi
  fi
  eval "$cmd --num-batches ${num_batches} --enable-profiling &> /dev/null" # Profile to get trace
  # move profiling file(s)
  if [ ${_ng} = 1 ];
  then
    outf="${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${_mb_size}.log"
    outp="dlrm_s_pytorch.prof"
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
  else
    outf="${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${_mb_size}_distributed.log"
    count=0
    for g in `ls dlrm_s_pytorch*.prof`
    do
      mv $g "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${_mb_size}_distributed_${count}.prof"
      mv ${g//".prof"/".json"} "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${_mb_size}_distributed_${count}.json"
      count=$((count+1))
    done
  fi
  eval "$cmd --num-batches ${num_batches} > $outf" # No profile to get E2E time
  min=$(grep "Finished" $outf | awk 'BEGIN{best=999999} {if (best > $8) best=$8} END{print best}')
  echo "Min time per iteration = $min ms"
done
