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


runtime_batch_iters=30
metrics_bench_iters=3
warmup_iters=5
benchmark_metrics="0"

op_type=$1
is_forward=$2
is_big=${3:0}
fbgemm=${4:0}
benchmark_metrics=${5:0}
shmem="1"
sgd="1"
fc_test="0"
header=""
param_file_name=""
file_prefix="${PM_HOME}/data/${GPU_NAME}/kernel/${op_type}_${is_forward}"
BUS_ID="$( nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader )"

if [ ! -f nvprof_metrics.txt ] && [ "$benchmark_metrics" == "1" ];
then
    echo "no nvprof metrics file"
    exit
fi

if [ "$op_type" == "embedding_lookup" ];
then
    if [ "$is_big" == "1" ];
    then
        param_file_name="${PM_HOME}/bench_params/embedding_lookup_params_big.txt"
    else
        param_file_name="${PM_HOME}/bench_params/embedding_lookup_params.txt"
    fi
    if [ "$is_forward" == "0" ];
    then
        if [ "$sgd" == "1" ];
        then
            file_prefix="${file_prefix}_sgd"
        else
            file_prefix="${file_prefix}_adagrad"
        fi
    fi
    if [ "$fbgemm" == "1" ];
    then
        header="kernel_name,batch_size,num_embeddings,num_tables,bag_size,embedding_dim,rows_per_block"
        file_prefix="${file_prefix}_fbgemm"
    else
        header="kernel_name,batch_size,num_embeddings,num_tables,bag_size,embedding_dim"
        if [ "$shmem" == "1" ];
        then
            file_prefix="${file_prefix}_shmem"
        fi
    fi
elif [ "$op_type" == "fully_connected" ];
then
    header="kernel_name,batch_size,M,N,K"
    if [ "$is_big" == "1" ];
    then
        param_file_name="${PM_HOME}/bench_params/fc_params_big.txt"
    else
        param_file_name="${PM_HOME}/bench_params/fc_params.txt"
    fi
    if [ "$fc_test" == "1" ];
    then
        param_file_name="${PM_HOME}/bench_params/fc_test_params.txt"
        file_prefix="${file_prefix}_test"
    fi
elif [ "$op_type" == "conv2d" ];
then
    header="kernel_name,batch_size,H,W,IC,OC,stride,dilation,FH,FW,is_dw"
    if [ "$is_big" == "1" ];
    then
        param_file_name="${PM_HOME}/bench_params/conv2d_params_big.txt"
    else
        param_file_name="${PM_HOME}/bench_params/conv2d_only.txt"
    fi
elif [ "$op_type" == "conv1d" ];
then
    header="kernel_name,batch_size,L,IC,OC,groups"
    param_file_name="${PM_HOME}/bench_params/conv1d_params.txt"
elif [ "$op_type" == "concat" ];
then
    header="kernel_name,batch_size,M,N,K"
    param_file_name="${PM_HOME}/bench_params/concat_params.txt"
elif [ "$op_type" == "cross_entropy" ];
then
    header="kernel_name,batch_size"
    param_file_name="${PM_HOME}/bench_params/ce_params.txt"
elif [ "$op_type" == "transpose" ];
then
    header="kernel_name,batch_size,M,N,trans_type"
    param_file_name="${PM_HOME}/bench_params/transpose_params.txt"
elif [ "$op_type" == "tril" ];
then
    header="kernel_name,batch_size,M,N,diag"
    param_file_name="${PM_HOME}/bench_params/tril_params.txt"
elif [ "$op_type" == "bn" ];
then
    header="kernel_name,batch_size,H,W,OC"
    param_file_name="${PM_HOME}/bench_params/bn_params.txt"
else # memcpy
    header="kernel_name,batch_size,M,N"
    param_file_name="${PM_HOME}/bench_params/memcpy_params.txt"
fi
if [ "$is_big" == "1" ];
then
    file_prefix="${file_prefix}_big"
fi

header="${header},kernel_runtime,op_runtime"
if [ "$op_type" != "memcpy" ];
then
    header="${header},block_x,block_y,block_z,thread_x,thread_y,thread_z,regs,ssmem,dsmem"

    if [ "$benchmark_metrics" == "1" ];
    then
        file_prefix="${file_prefix}_big_with_metrics"
        metrics_args=""
        metrics=()
        while IFS= read -r line
        do
            metrics_args="$metrics_args --metrics $line"
            metrics+=( "$line" )
            header="$header,$line"
        done < "nvprof_metrics.txt"
    fi
else
    header="${header},throughput"
fi

file_name="${file_prefix}.csv"
if [ ! -f "$file_name" ];
then
    touch "$file_name"
    echo "${header}" >> "$file_name"
fi

# Benchmark operator
while IFS= read -r line
do
    IFS=', ' read -r -a array <<< "$line"
    bench_param=""
    last_array=""

    if [ "$op_type" == "embedding_lookup" ];
    then
        if ["$last_array" != ""];
        then
            tmp="${array[@]:0:5}"
            if [ "$tmp" == "$last_array" ];
            then
                continue
            fi
        fi
        last_array="${array[@]:0:5}"
        if [ "$fbgemm" == "1" ];
        then
            bench_param="--op-type $op_type --batch-size ${array[0]} --num-embeddings ${array[1]} --num-tables ${array[2]} --bag-size ${array[3]} --embedding-dim ${array[4]} --fbgemm"
        else
            bench_param="--op-type $op_type --batch-size ${array[0]} --num-embeddings ${array[1]} --num-tables ${array[2]} --bag-size ${array[3]} --embedding-dim ${array[4]} --rows-per-block ${array[5]}"
            if [ "${array[5]}" -gt "32" ] && [ "$is_forward" == "0" ]; # Skip when backward and rows_per_block too big
            then
                continue
            fi
            if [ "$shmem" == "1" ];
            then
                bench_param="${bench_param} --shmem"
            fi
        fi
        if [ "$sgd" == "1" ];
        then
            bench_param="${bench_param} --sgd"
        fi
    elif [ "$op_type" == "fully_connected" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "conv2d" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --H ${array[1]} --W ${array[2]} --IC ${array[3]} --OC ${array[4]} --stride ${array[5]} --dilation ${array[6]} --FH ${array[7]} --FW ${array[8]}"
        if [ "${array[9]}" == "1" ];
        then
            bench_param="${bench_param} --is-dw"
        fi
    elif [ "$op_type" == "conv1d" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --L ${array[1]} --IC ${array[2]} --OC ${array[3]} --groups ${array[4]}"
    elif [ "$op_type" == "concat" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "cross_entropy" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]}"
    elif [ "$op_type" == "transpose" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --trans-type ${array[3]}"
    elif [ "$op_type" == "tril" ]; # lower triangular after feature interaction
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --diag ${array[3]}"
    elif [ "$op_type" == "bn" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --H ${array[1]} --W ${array[2]} --OC ${array[3]}"
   else # Memcpy
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]}"
    fi
    if [ "$is_forward" == "0" ];
    then
        bench_param="${bench_param} --backward"
    fi
    echo "$bench_param"

    # Benchmark operator runtime: no nvprof
    python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iters --warmup-iters $warmup_iters >& "/tmp/${BUS_ID}_op.txt"
    op_time="$( < /tmp/${BUS_ID}_op.txt grep 'Time: ' | awk '{ x=gensub("    ","","G",$NF); x=gensub("us","","G",x); printf x }' )"

    # Benchmark general: get the major kernel names
    nvprof --openacc-profiling off --log-file "/tmp/${BUS_ID}_profile_results.txt" \
    python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $metrics_bench_iters \
    --warmup-iters $warmup_iters >& /dev/null

    # Get gpu trace
    echo "Get GPU trace of kernels ..."
    nvprof --openacc-profiling off --print-gpu-trace --log-file "/tmp/${BUS_ID}_kernel_trace.txt" \
    python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iters \
    --warmup-iters $warmup_iters >& /dev/null

    ./get_kernel_names.sh "$op_type"
    kernels=()
    while IFS= read -r line
    do
        kernels+=( "$line" )
    done < "/tmp/${BUS_ID}_kernel_names.txt"
    len=${#kernels[@]}

    for (( i=0; i<len; i++ ));
    do
        kernel=${kernels[i]}
        # Truncate concat kernel name
        if [ "$op_type" == "concat" ];
        then
            kernel=${kernel##*::}
        fi

        # Concat param
        stats_row="$kernel"
        array_len=${#array[@]}
        for (( j=0; j<array_len; j++ ));
        do
            stats_row="$stats_row,${array[j]}"
        done

        # Get trace info
        trace_values=""
        avg_kernel_time=0
        kernel_count=0
        while IFS= read -r line
        do
            # Skip the first few lines
            if [ "$( echo "$line" | grep "$kernel" )" == "" ];
            then
                continue
            fi

            # Get thread grid / block info
            kernel_count="$( echo "$kernel_count + 1" | bc )"
            IFS=', ' read -r -a x <<< "$line"
            if [ "$trace_values" == "" ];
            then
                if [ "$op_type" == "memcpy" ];
                then
                    trace_values="${x[8]}"
                else
                    trace_values="${x[2]/(/},${x[3]},${x[4]/)/},${x[5]/(/},${x[6]},${x[7]/)/},${x[8]},${x[9]},${x[10]}"
                fi
            fi

            # Get average kernel time excluding warmup runs, and convert to us
            kernel_time="${x[1]}"
            if [[ "$kernel_time" == *"us"* ]];
            then
                kernel_time="$( echo "$kernel_time" | tr --delete 'us' )"
            elif [[ "$kernel_time" == *"ms"* ]];
            then
                kernel_time="$( echo "$kernel_time" | tr --delete 'ms' )"
                kernel_time="$( echo "scale=4; $kernel_time * 1000.0" | bc )"
            else # in second
                kernel_time="$( echo "$kernel_time" | tr --delete 's' )"
                kernel_time="$( echo "scale=4; $kernel_time * 1000000.0" | bc )"
            fi
            if [ "$kernel_count" -gt "$warmup_iters" ];
            then
                avg_kernel_time="$( echo "scale=4; $avg_kernel_time + $kernel_time" | bc )"
            fi
        done < "/tmp/${BUS_ID}_kernel_trace.txt"
        avg_kernel_time="$( echo "scale=4; $avg_kernel_time / ($kernel_count - $warmup_iters) * ($kernel_count / ($runtime_batch_iters + $warmup_iters))" | bc )" # In case 1 op = multiple identical kernel calls, e.g. big transpose
        stats_row="${stats_row},${avg_kernel_time},${op_time},${trace_values}"

        if [ "$benchmark_metrics" == "1" ] && [ "$op_type" != "memcpy" ];
        then
            # Benchmark kernel metrics
            echo "Benchmark kernel $kernel"
            nvprof --openacc-profiling off --kernels $kernel \
            $metrics_args \
            --log-file "/tmp/${BUS_ID}_kernel.txt" \
            python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $metrics_bench_iters --warmup-iters 0 # >& /dev/null

            metric_values=()
            for (( j=0; j<${#metrics[@]}; j++ ));
            do
                value="$( < /tmp/${BUS_ID}_kernel.txt grep -m1 "${metrics[j]}" | awk '{ x=gensub("    ","","G",$NF); printf x }' )"
                metric_values+=( "$value" )
            done

            for (( j=0; j<${#metrics[@]}; j++ ));
            do
                stats_row="$stats_row,${metric_values[j]}"
            done
        fi
        echo "$stats_row" >> "$file_name"
    done
done < "$param_file_name"
