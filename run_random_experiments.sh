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

dlrm_max_batch_size=4096
trimmed_iters=30 # Default iters 30
share_overheads= # Share overheads
emb_type= # FBGEMM for DLRM
bucket_size_mb=25 # Bucket size in MB
early_barrier= # Whether launch a barrier at the beginning of the iteration
aggregated_allreduce= # Whether extract an execution graph with aggregated allreduce for DDP (i.e. iteration 0)
dataset_suffix=2021
while getopts i:ots:rad:x: flag
do
    case "${flag}" in
        i) trimmed_iters=${OPTARG};;
        o) share_overheads="-o";;
        t) emb_type="-t";;
        s) bucket_size_mb=${OPTARG};;
        r) early_barrier="-r";;
        a) aggregated_allreduce="-a";;
        x) dataset_suffix=${OPTARG};;
    esac
done

# Get GPU memory size
GPU_memory=0
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader > /tmp/gpu_name.csv
for GPU_NAME in "A100" "V100" "P100" "Xp";
do
    if grep -q "$GPU_NAME" /tmp/gpu_name.csv
    then
        tmp=`grep -e "$GPU_NAME" /tmp/gpu_name.csv`
        read -a array <<< "$tmp"
        GPU_memory_MB="${array[${#array[@]}-2]}"
        GPU_memory="$( echo "$GPU_memory_MB * 1000 * 1000" | bc -l )"
        break
    fi
done
if [[ $GPU_memory == 0 ]];
then
    echo "Unrecognized GPU name! Exit..."
    exit
fi

ngpus="$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )"
cd benchmark

# Extract random tables from DLRM open-source dataset
python generate_random_dlrm_tasks.py --per-gpu-memory $GPU_memory

# Benchmark and trace analysis
while IFS= read -r table_indices
do
    for batch_size in 16 32 64 128;
    do
        for sharder in naive naive_chunk random dim_greedy size_greedy lookup_greedy norm_lookup_greedy size_lookup_greedy;
        do
            # size_greedy for 16, 32, 64
            if [ "$batch_size" != 128 ] && [ "$sharder" != "size_greedy" ];
            then
                continue
            fi

            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                trace_cmd="mpirun -np $ngpus -N $ngpus python"
            else
                trace_cmd="python"
            fi

            # Options for both benchmark and trace_stats
            options="   -m DLRM_open_source\
                        -g ${ngpus}\
                        ${emb_type}\
                        -s ${bucket_size_mb}\
                        ${early_barrier}\
                        ${aggregated_allreduce}\
                        -d ${table_indices}\
                        -h ${sharder}"

            trace_cmd="$trace_cmd \
                        trace_stats.py \
                        -i ${trimmed_iters} \
                        ${options}"

            # ./dlrm_benchmark.sh -b $((batch_size*32)) $options
            eval "$trace_cmd -b $((batch_size*32))" < /dev/null
        done
    done
done < "tasks_${dataset_suffix}_${ngpus}.txt"

# Create shared overheads
python create_shared_overheads.py --iters $trimmed_iters

# Run prediction
while IFS= read -r table_indices
do
    for batch_size in 16 32 64 128;
    do
        for sharder in size_greedy; # naive naive_chunk random dim_greedy size_greedy lookup_greedy norm_lookup_greedy size_lookup_greedy;
        do
            # size_greedy for 16, 32, 64
            if [ "$batch_size" != 128 ] && [ "$sharder" != "size_greedy" ];
            then
                continue
            fi

            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                cmd="mpirun -np $ngpus -N $ngpus python"
            else
                cmd="python"
            fi

            options="   -i ${trimmed_iters}\
                        -m DLRM_open_source\
                        -g ${ngpus}\
                        ${emb_type}\
                        -s ${bucket_size_mb}\
                        ${early_barrier}\
                        ${aggregated_allreduce}\
                        -d ${table_indices}\
                        -h ${sharder}\
                        ${share_overheads}"

            cmd="   $cmd\
                    e2e.py\
                    ${options}"

            eval "$cmd -b $((batch_size*32))" < /dev/null
        done
    done
done < "tasks_${dataset_suffix}_${ngpus}.txt"

cd ..
