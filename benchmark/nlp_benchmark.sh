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


# Arguments
ngpus=1
bucket_size_mb=25
early_barrier=
aggregated_allreduce=
num_batches=200
while getopts b:m:g:ts:rad:h:e: flag
do
    case "${flag}" in
        b) mb_size=${OPTARG};;
        m) model_name=${OPTARG};;
        g) ngpus=${OPTARG};;
        s) bucket_size_mb=${OPTARG};;
        r) early_barrier="--early-barrier";;
        a) aggregated_allreduce="--aggregated-allreduce";;
    esac
done

model_list="bert gpt2"
if [[ $model_list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi

CORES=`lscpu | grep "Core(s)" | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - ${model_name} running on $ngpus GPUs"
echo "--------------------------------------------"
rm -f /tmp/pytorch_execution_graph*
_gpus=$(seq -s, 0 $((ngpus-1)))
nlp_pt_bin="accelerate launch --num_processes ${ngpus} nlp_transformers.py"
graph_filename_pattern="${ngpus}_${mb_size}_graph.json"
cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
echo "-------------------"
echo "Using GPUS: $_gpus, batch size: $mb_size"
echo "-------------------"
folder="${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}/"
if [ ${ngpus} -gt 1 ];
then
  if [[ $early_barrier == "--early-barrier" ]];
  then
      folder="${folder}/barrier"
  else
      folder="${folder}/no_barrier"
  fi
  if [[ $aggregated_allreduce == "--aggregated-allreduce" ]];
  then
      folder="${folder}_aggregated_allreduce"
  else
      folder="${folder}_bucketed_allreduce"
  fi
  folder="${folder}/${bucket_size_mb}"
fi
mkdir -p "${folder}"
cmd="$cuda_arg $nlp_pt_bin --model-name ${model_name} --batch-size=${mb_size} "
if [[ ${ngpus} != `ls ${folder} | grep -e $graph_filename_pattern | wc -l` ]];
then
  echo "Execution graph doesn't exist! Extract it..."
  eval "$cmd --num-batches 2 --collect-execution-graph --enable-profiling --profile-out-dir . &> /dev/null" # Collect execution graph
  if [ ${ngpus} = 1 ];
  then
    cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "${folder}/${ngpus}_${mb_size}_graph.json"
  else
    count=0
    for g in `ls /tmp/pytorch_execution_graph*`;
    do
      cp $g "${folder}/${ngpus}_${mb_size}_distributed_${count}_graph.json"
      count=$((count+1))
    done
  fi
fi
eval "$cmd --num-batches ${num_batches} --enable-profiling --profile-out-dir . &> /dev/null" # Profile to get trace
# move profiling file(s)
if [ ${ngpus} = 1 ];
then
  outf="${folder}/${ngpus}_${mb_size}.log"
  outp="nlp.prof"
  mv $outp ${outf//".log"/".prof"}
  mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
else
  outf="${folder}/${ngpus}_${mb_size}_distributed.log"
  count=0
  for g in `ls nlp*.prof`;
  do
    mv $g "${folder}/${ngpus}_${mb_size}_distributed_${count}.prof"
    mv ${g//".prof"/".json"} "${folder}/${ngpus}_${mb_size}_distributed_${count}.json"
    count=$((count+1))
  done
fi
eval "$cmd --num-batches ${num_batches} > $outf" # No profile to get E2E time
min=$(grep "Finished" $outf | awk 'BEGIN{best=999999} {if (best > $4) best=$4} END{print best}')
echo "Min time per iteration = $min ms"
