#!/bin/bash

# Get model name
model_name=$1
list="ncf deepfm"
if [[ $list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi
mb_size=$2

gpu="1"
ngpus="1" #"1 2 4"

CORES=`lscpu | grep "Core(s)" | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

# GPU Benchmarking
if [ $gpu = 1 ];
then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - ${model_name} running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    rm -f /tmp/pytorch_execution_graph*
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    mkdir -p "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}"
    outf="${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}.log"
    outp="${model_name}_benchmark.prof"
    echo "-------------------------------"
    echo "Running benchmark (log file: $outf)"
    echo "-------------------------------"
    if [ $model_name = "ncf" ];
    then
        cd ${PM_HOME}/3rdparty/ncf/src
        cmd="python train.py --batch-size ${mb_size}"
        num_epoch=1
    else # DeepFM
        cd ${PM_HOME}/3rdparty/deepfm
        cmd="python main.py --batch-size ${mb_size} --embedding-dim 128 --mlp-hidden-size 64-64"
        num_epoch=500
    fi
    if [ ! -f "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}_graph.json" ];
    then
      echo "Execution graph doesn't exist! Extract it..."
      eval "$cmd --num-epoch 1 --collect-execution-graph --profile --num-batches 1 &> /dev/null" # Collect execution graph
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}_graph.json"
    fi
    eval "$cmd --num-epoch ${num_epoch} --profile --num-batches 500 > $outf" # Profile to get trace
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    eval "$cmd --num-epoch ${num_epoch} --num-batches 100 > $outf" # No profile to get E2E time
    cd "${PM_HOME}"
  done
fi
