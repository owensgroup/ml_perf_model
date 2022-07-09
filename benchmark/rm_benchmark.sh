#!/bin/bash
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
num_batches=200

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
        cmd="python train.py --batch-size ${mb_size} --num-epoch 1"
    else # DeepFM
        cd ${PM_HOME}/3rdparty/deepfm
        cmd="python main.py --batch-size ${mb_size} --embedding-dim 128 --mlp-hidden-size 64-64 --num-epoch 1"
    fi
    if [ ! -f "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}_graph.json" ];
    then
      echo "Execution graph doesn't exist! Extract it..."
      eval "$cmd --collect-execution-graph --profile --num-batches 1 &> /dev/null" # Collect execution graph
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/${_ng}_${mb_size}_graph.json"
    fi
    eval "$cmd --profile --num-batches ${num_batches} > $outf" # Profile to get trace
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    eval "$cmd --num-batches 100 > $outf" # No profile to get E2E time
    cd "${PM_HOME}"
  done
fi
