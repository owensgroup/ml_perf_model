#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Arguments
ngpus=1
emb_type="--fbgemm-emb"
folder_emb_type="f" # FBGEMM
bucket_size_mb=25
early_barrier=
aggregated_allreduce=
table_indices="4-24-26-156-340-404" # Default tables
sharder="naive" # Default sharder scheme
dataset_suffix=2021
while getopts b:m:g:ts:rad:h:e:x: flag
do
    case "${flag}" in
        b) mb_size=${OPTARG};;
        m) model_name=${OPTARG};;
        g) ngpus=${OPTARG};;
        t) emb_type="--batched-emb"
            folder_emb_type="b";;
        s) bucket_size_mb=${OPTARG};;
        r) early_barrier="--early-barrier";;
        a) aggregated_allreduce="--aggregated-allreduce";;
        d) table_indices=${OPTARG};;
        h) sharder=${OPTARG};;
        e) dlrm_extra_option=${OPTARG};;
        x) dataset_suffix=${OPTARG};;
    esac
done

model_list="DLRM_test DLRM_default DLRM_MLPerf DLRM_DDP DLRM_open_source"
if [[ $model_list =~ (^|[[:space:]])$model_name($|[[:space:]]) ]];
then
    :;
else
    echo "Model name not supported!"
    exit
fi

tnworkers=0
tmb_size=-1 #256
num_batches=150
common_args="   --use-gpu\
                --print-freq=5\
                --print-time\
                --pin-memory\
                --sharder=${sharder}\
                ${emb_type}\
                --bucket-size-mb=${bucket_size_mb}\
                ${early_barrier}\
                ${aggregated_allreduce} "
model_name_year_indices=$model_name

# ----------------------- Model param -----------------------
_args=""
if [[ $model_name == "DLRM_test" ]]; # DLRM test
then
    _args=" --data-generation=random\
            --processed-data-file=/nvme/deep-learning/dlrm_random\
            --round-targets\
            --arch-mlp-bot=512-64\
            --arch-mlp-top=1024-1\
            --arch-sparse-feature-size=64\
            --arch-embedding-size=1000000\
            --num-indices-per-lookup=10\
            --num-indices-per-lookup-fixed\
            --arch-interaction-op=dot\
            --numpy-rand-seed=727\
            --num-worker=0 "
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
elif [[ $model_name == "DLRM_open_source" ]]; # DLRM using the open-source dataset
then
    model_name_year_indices="${model_name_year_indices}/${dataset_suffix}/${table_indices}"
    _args=" --data-generation=dataset\
            --data-set=dlrm_open_source\
            --processed-data-file=/nvme/deep-learning/dlrm_datasets/embedding_bag/${dataset_suffix}/merged_simple.pt\
            --arch-embedding-table-indices=${table_indices}\
            --arch-mlp-bot=512-1024-256-128\
            --arch-mlp-top=1024-512-256-1\
            --loss-function=bce\
            --learning-rate=1.0\
            --test-freq=102400\
            --memory-map "
fi

# GPU Benchmarking
echo "--------------------------------------------"
echo "GPU Benchmarking - ${model_name_year_indices} running on $ngpus GPUs"
echo "--------------------------------------------"
rm -f /tmp/pytorch_execution_graph*
_gpus=$(seq -s, 0 $((ngpus-1)))
if [ ${ngpus} = 1 ];
then
  dlrm_pt_bin="python ${PM_HOME}/3rdparty/dlrm/dlrm_s_pytorch.py"
  graph_filename_pattern="${ngpus}_${mb_size}_graph.json"
else
  dlrm_pt_bin="python -m torch.distributed.run --nproc_per_node=${ngpus} ${PM_HOME}/3rdparty/dlrm/dlrm_s_pytorch.py --dist-backend=nccl "
  graph_filename_pattern="${ngpus}_${mb_size}_distributed_[0-$((ngpus-1))]_graph.json"
fi
cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
echo "-------------------"
echo "Using GPUS: $_gpus, batch size: $mb_size"
echo "-------------------"
folder="${PM_HOME}/data/${GPU_NAME}/e2e/${model_name_year_indices}/${folder_emb_type}"
if [ ${ngpus} -gt 1 ];
then
  folder="${folder}/${sharder}"
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
cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers ${common_args} ${_args} $dlrm_extra_option"
if [[ ${ngpus} != `ls ${folder} | grep -e $graph_filename_pattern | wc -l` ]];
then
  echo "Execution graph doesn't exist! Extract it..."
  eval "$cmd --num-batches 2 --collect-execution-graph --enable-profiling --profile-out-dir . --test-freq=-1 &> /dev/null" # Collect execution graph
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
if [ ${ngpus} -eq 1 ];
then
  outf="${folder}/${ngpus}_${mb_size}.log"
  outp="dlrm_s_pytorch.prof"
  mv $outp ${outf//".log"/".prof"}
  mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
else
  outf="${folder}/${ngpus}_${mb_size}_distributed.log"
  count=0
  for g in `ls dlrm_s_pytorch*.prof`;
  do
    mv $g "${folder}/${ngpus}_${mb_size}_distributed_${count}.prof"
    mv ${g//".prof"/".json"} "${folder}/${ngpus}_${mb_size}_distributed_${count}.json"
    count=$((count+1))
  done
fi
eval "$cmd --num-batches ${num_batches} > $outf" # No profile to get E2E time
# move Ls and rfs file(s)
if [ ${ngpus} -eq 1 ];
then
  mv "Ls.txt" ${outf//".log"/"_Ls.txt"}
  mv "rfs.txt" ${outf//".log"/"_rfs.txt"}
else
  count=0
  for g in $(seq 1 $ngpus);
  do
    mv "Ls_${count}.txt" "${folder}/${ngpus}_${mb_size}_distributed_${count}_Ls.txt"
    mv "rfs_${count}.txt" "${folder}/${ngpus}_${mb_size}_distributed_${count}_rfs.txt"
    count=$((count+1))
  done
fi
min=$(grep "Finished" $outf | awk 'BEGIN{best=999999} {if (best > $8) best=$8} END{print best}')
echo "Min time per iteration = $min ms"
