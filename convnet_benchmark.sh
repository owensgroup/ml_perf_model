#!/bin/bash

# Get model name
model_name=${1:-resnet50}
list="alexnet vgg11 inception_v3 resnet18 resnet50 resnext101 wide_resnet50_2 mnasnet_a1 mnasnet0_5 \
        squeezenet1_0 densenet121 mobilenet_v1 mobilenet_v2 shufflenet unet unet3d"
if [[ $list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi

gpu="1"
ngpus="1" #"1 2 4"
num_steps="100"

CORES=`lscpu | grep "Core(s)" | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

# Get GPU type
./get_gpu_name.sh
export GPU_NAME=`cat /tmp/gpu_name.txt`

# ----------------------- Model param -----------------------
# GPU Benchmarking
if [ $gpu = 1 ];
then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    rm -f /tmp/pytorch_execution_graph*
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    outf="data/${GPU_NAME}/e2e/${model_name}_${_ng}.log"
    outp="convnet_benchmark.prof"
    echo "-------------------------------"
    echo "Running benchmark (log file: $outf)"
    echo "-------------------------------"
    cmd="python -u convnet-benchmark-py/benchmark.py --profile --arch ${model_name}"
    if [ ! -f "data/${GPU_NAME}/e2e/${model_name}_${_ng}_graph.json" ];
    then
      echo "Execution graph doesn't exist! Extract it..."
      eval "$cmd --num-steps 2 --collect-execution-graph &> /dev/null" # Collect execution graph
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "data/${GPU_NAME}/e2e/${model_name}_${_ng}_graph.json"
    fi
    eval "$cmd --num-steps ${num_steps} > $outf"
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    echo "Min time per iteration = $min"
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
  done
fi