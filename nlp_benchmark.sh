#!/bin/bash

# Get model name
model_name=${1:-transformer}
list="transformer seq2seq"
if [[ $list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi

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

# Get GPU type
./get_gpu_name.sh
export GPU_NAME=`cat /tmp/gpu_name.txt`

if [ ${model_name} == "transformer" ];
then
    if [[ ! -d "transformer-pt/.data" ]]; # Data preprocessing
    then
        python -m spacy download en
        python -m spacy download de
        cd transformer-pt
        python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
        cd ..
    fi
fi

# GPU Benchmarking
if [ $gpu = 1 ];
then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    rm -f /tmp/pytorch_execution_graph*
    cd transformer-pt
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    outf="${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}_${_ng}.log"
    outp="${model_name}_benchmark.prof"
    echo "-------------------------------"
    echo "Running benchmark (log file: $outf)"
    echo "-------------------------------"
    cmd="python train.py -data_pkl m30k_deen_shr.pkl -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 5000 -profile -no_eval"
    if [ ! -f "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}_${_ng}_graph.json" ];
    then
      echo "Execution graph doesn't exist! Extract it..."
      eval "$cmd -epoch 1 -collect_execution_graph -truncate &> /dev/null" # Collect execution graph
      cp `ls -1t /tmp/pytorch_execution_graph* | tail -1` "${PM_HOME}/data/${GPU_NAME}/e2e/${model_name}_${_ng}_graph.json"
    fi
    eval "$cmd -epoch 1 > $outf"
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    cd ..
  done
fi
