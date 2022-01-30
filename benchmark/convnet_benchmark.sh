#!/bin/bash

# Get model name
model_name=$1
list="alexnet vgg11 inception_v3 resnet18 resnet50 resnext101 wide_resnet50_2 mnasnet_a1 mnasnet0_5 \
        squeezenet1_0 densenet121 mobilenet_v1 mobilenet_v2 mobilenet_v3_large shufflenet efficientnet_b7 unet unet3d"
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
num_steps="500"

CORES=`lscpu | grep "Core(s)" | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

# ----------------------- Model param -----------------------
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
    outp="convnet_benchmark.prof"
    echo "-------------------------------"
    echo "Running benchmark (log file: $outf)"
    echo "-------------------------------"
    cmd="python -u 3rdparty/convnet-benchmark-py/benchmark.py --arch ${model_name} --batch-size ${mb_size}"
    if [ ! -f "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}_graph.json" ];
    then
      echo "Execution graph doesn't exist! Extract it..."
      eval "$cmd --num-steps 2 --collect-execution-graph --profile &> /dev/null" # Collect execution graph
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}_graph.json"
    fi
    eval "$cmd --num-steps ${num_steps} --profile > $outf" # Profile to get trace
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    eval "$cmd --num-steps ${num_steps} > $outf" # No profile to get E2E time
    min=$(grep "Finished" $outf | awk 'BEGIN{best=999999} {if (best > $4) best=$4} END{print best}')
    echo "Min time per iteration = $min ms"
  done
fi
