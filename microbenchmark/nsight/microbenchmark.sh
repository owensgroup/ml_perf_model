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
fbgemm_caching=${5:0}
benchmark_metrics=${6:0}
shmem="1"
sgd="1"
fc_test="0"
header=""
param_file_name=""
file_prefix="${PM_HOME}/data/${GPU_NAME}/kernel/${op_type}_${is_forward}"
BUS_ID="$( nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader )"

if [ ! -f nsight_metrics.txt ] && [ "$benchmark_metrics" == "1" ];
then
    echo "no nsight metrics file"
    exit
fi

if [ "$op_type" == "embedding_lookup" ];
then
    if [ "$fbgemm" == "2" ];
    then
        if [ "$is_big" == "1" ]; # Borrow is_big for dlrm dataset testing
        then
            param_file_name="${PM_HOME}/bench_params/embedding_lookup_params_dlrm_datasets_test.txt"
        else
            param_file_name="${PM_HOME}/bench_params/embedding_lookup_params_dlrm_datasets.txt"
        fi
    else 
        if [ "$is_big" == "1" ];
        then
            param_file_name="${PM_HOME}/bench_params/embedding_lookup_params_big.txt"
        else
            param_file_name="${PM_HOME}/bench_params/embedding_lookup_params.txt"
        fi
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
    if [ "$fbgemm" -ne "0" ];
    then
        header="kernel_name,batch_size,num_embeddings,num_tables,bag_size,embedding_dim"
        file_prefix="${file_prefix}_fbgemm"
        if [ "$fbgemm_caching" == "1" ];
        then
            file_prefix="${file_prefix}_caching"
        fi
        if [ "$fbgemm" == "2" ];
        then
            header="${header},dataset_path"
            file_prefix="${file_prefix}_dlrm_datasets"
        fi
    else
        header="kernel_name,batch_size,num_embeddings,num_tables,bag_size,embedding_dim,rows_per_block"
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
elif [ "$op_type" == "ln" ];
then
    header="kernel_name,batch_size,M,N"
    param_file_name="${PM_HOME}/bench_params/ln_params.txt"
elif [ "$op_type" == "gelu" ];
then
    header="kernel_name,batch_size,M,N"
    param_file_name="${PM_HOME}/bench_params/gelu_params.txt"
elif [ "$op_type" == "dropout" ];
then
    header="kernel_name,batch_size,M,N,p"
    param_file_name="${PM_HOME}/bench_params/dropout_params.txt"
else # memcpy
    header="kernel_name,batch_size,M,N"
    param_file_name="${PM_HOME}/bench_params/memcpy_params.txt"
fi
if [ "$is_big" == "1" ];
then
    if [[ "$op_type" == "embedding_lookup" && "$fbgemm" == "2" ]]; # Borrow is_big for dlrm dataset testing
    then
        file_prefix="${file_prefix}_test"
        reuse_factor_file_name="${file_prefix%/*}/embedding_lookup_fbgemm_dlrm_datasets_test_rf.csv"
        if [ ! -f "$reuse_factor_file_name" ];
        then
            touch "$reuse_factor_file_name"
            echo "batch_size,num_embeddings,num_tables,bag_size,embedding_dim,dataset_path,reuse_factors" >> "$reuse_factor_file_name"
        fi
    else
        file_prefix="${file_prefix}_big"
    fi
else
    if [[ "$op_type" == "embedding_lookup" && "$fbgemm" == "2" ]]; # Borrow is_big for dlrm dataset testing
    then
        reuse_factor_file_name="${file_prefix%/*}/embedding_lookup_fbgemm_dlrm_datasets_rf.csv"
        if [ ! -f "$reuse_factor_file_name" ];
        then
            touch "$reuse_factor_file_name"
            echo "batch_size,num_embeddings,num_tables,bag_size,embedding_dim,dataset_path,reuse_factors" >> "$reuse_factor_file_name"
        fi
    fi
fi

header="${header},kernel_runtime,op_runtime"
if [ "$op_type" != "memcpy" ];
then
    header="${header},block_x,block_y,block_z,thread_x,thread_y,thread_z,regs,ssmem,dsmem"

    if [ "$benchmark_metrics" == "1" ];
    then
        file_prefix="${file_prefix}_with_metrics"
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

# Benchmark operator
file_name="${file_prefix}.csv"
while IFS= read -r line
do
    IFS=', ' read -r -a array <<< "$line"
    bench_param="--op-type $op_type --batch-size ${array[0]}"
    last_array=""

    # Protection from unexpected absence of data file
    if [ ! -f "$file_name" ];
    then
        touch "$file_name"
        echo "${header}" >> "$file_name"
    fi

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
        if [ "$fbgemm" -ne "0" ];
        then
            bench_param="${bench_param} --num-embeddings ${array[1]} --num-tables ${array[2]} --bag-size ${array[3]} --embedding-dim ${array[4]} --fbgemm"
            if [ "$fbgemm_caching" == "1" ];
            then
                bench_param="${bench_param} --caching"
            fi
            if [ "$fbgemm" == "2" ]; # Dataset
            then
                bench_param="${bench_param} --dataset ${array[5]}"
            fi
        else
            bench_param="${bench_param} --num-embeddings ${array[1]} --num-tables ${array[2]} --bag-size ${array[3]} --embedding-dim ${array[4]} --rows-per-block ${array[5]}"
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
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "conv2d" ];
    then
        bench_param="${bench_param} --H ${array[1]} --W ${array[2]} --IC ${array[3]} --OC ${array[4]} --stride ${array[5]} --dilation ${array[6]} --FH ${array[7]} --FW ${array[8]}"
        if [ "${array[9]}" == "1" ];
        then
            bench_param="${bench_param} --is-dw"
        fi
    elif [ "$op_type" == "conv1d" ];
    then
        bench_param="${bench_param} --L ${array[1]} --IC ${array[2]} --OC ${array[3]} --groups ${array[4]}"
    elif [ "$op_type" == "concat" ];
    then
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "cross_entropy" ];
    then
        :
    elif [ "$op_type" == "transpose" ];
    then
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]} --trans-type ${array[3]}"
    elif [ "$op_type" == "tril" ]; # lower triangular after feature interaction
    then
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]} --diag ${array[3]}"
    elif [ "$op_type" == "bn" ];
    then
        bench_param="${bench_param} --H ${array[1]} --W ${array[2]} --OC ${array[3]}"
    elif [ "$op_type" == "ln" ];
    then
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]}"
    elif [ "$op_type" == "gelu" ];
    then
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]}"
    elif [ "$op_type" == "dropout" ];
    then
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]} --p ${array[3]}"
    else # Memcpy
        bench_param="${bench_param} --M ${array[1]} --N ${array[2]}"
    fi
    if [ "$is_forward" == "0" ];
    then
        bench_param="${bench_param} --backward"
    fi
    echo "$bench_param"

    # Benchmark operator runtime: no nsight
    rm /tmp/${BUS_ID}*
    python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iters --warmup-iters $warmup_iters >& "/tmp/${BUS_ID}_op.txt"
    op_time="$( < /tmp/${BUS_ID}_op.txt grep 'Time: ' | awk '{ x=gensub("    ","","G",$NF); x=gensub("us","","G",x); printf x }' )"

    if [[ "$op_type" == "embedding_lookup" && "$fbgemm" == "2" ]];
    then
        # Pooling factors
        tmp="$( grep 'L: ' /tmp/${BUS_ID}_op.txt | awk '{print $14}' )"
        if [[ "$tmp" == "" ]];
        then
            echo "==== Probably OOM! ===="
            echo "$bench_param"
            continue
        fi
        array[3]="${tmp::-1}"
        array_len=${#array[@]}
        rf_row=""
        for (( j=0; j<array_len; j++ ));
        do
            rf_row="$rf_row,${array[j]}"
        done
        # Reuse factors
        tmp="$( grep 'Reuse ' /tmp/${BUS_ID}_op.txt | awk '{print $3}' )"
        echo "${rf_row:1},$tmp" >> "$reuse_factor_file_name"
    fi

    # Get gpu trace
    echo "Get GPU trace of kernels ..."
    nsys profile --trace=cuda --force-overwrite true --output "/tmp/${BUS_ID}_profile_results.nsys-rep" \
    python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iters \
    --warmup-iters $warmup_iters >& /dev/null

    # Extract stats
    if [ "$op_type" != "memcpy" ];
    then
        nsys stats --report gpukernsum --format csv --output . --force-overwrite true /tmp/${BUS_ID}_profile_results.nsys-rep >& /dev/null
    else
        nsys stats --report gpumemtimesum --format csv --output . --force-overwrite true /tmp/${BUS_ID}_profile_results.nsys-rep >& /dev/null
    fi
    nsys stats --report gputrace --format csv --output . --force-overwrite true /tmp/${BUS_ID}_profile_results.nsys-rep >& /dev/null
    mv /tmp/${BUS_ID}_profile_results_gputrace.csv /tmp/${BUS_ID}_kernel_trace.csv

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

        if [ "$op_type" == "memcpy" ];
        then
            result_file="/tmp/${BUS_ID}_profile_results_gpumemtimesum.csv"
        else
            result_file="/tmp/${BUS_ID}_profile_results_gpukernsum.csv"
        fi

        # Get kernel time
        first_line=$( head -n 1 $result_file )
        IFS=',' read -r -a x <<< "$first_line"
        duration_unit="$( echo ${x[1]} | awk '{gsub(/^[^(]*\(|\)[^)]*$/,"",$0);print $0}' )" # Extract unit inside parentheses
        related_line=$( cat $result_file | grep "$kernel" )
        IFS=',' read -r -a x <<< "$related_line"
        avg_kernel_time="${x[4]}"
        if [[ "$duration_unit" == "ns" ]]; # Convert to us
        then
            avg_kernel_time="$( echo "scale=4; $avg_kernel_time / 1000.0" | bc )"
        elif [[ "$duration_unit" == "ms" ]];
        then
            avg_kernel_time="$( echo "scale=4; $avg_kernel_time * 1000.0" | bc )"
        else # in second
            avg_kernel_time="$( echo "scale=4; $avg_kernel_time * 1000000.0" | bc )"
        fi

        # Get trace info
        trace_values=""
        first_line=$( head -n 1 "/tmp/${BUS_ID}_kernel_trace.csv" )
        IFS=',' read -r -a x <<< "$first_line"
        throughput_unit="$( echo ${x[13]} | awk '{gsub(/^[^(]*\(|\)[^)]*$/,"",$0);print $0}' )" # Ditto
        related_line=$( cat "/tmp/${BUS_ID}_kernel_trace.csv" | grep "$kernel" | head -n 1 )
        IFS=',' read -r -a x <<< "$related_line"
        if [ "$op_type" == "memcpy" ];
        then
            trace_values="${x[13]}"
            # Convert to GB/s
            if [[ "$throughput_unit" == "MB/s" ]];
            then
                trace_values="$( echo "scale=4; $trace_values / 1000.0" | bc )"
            fi
            trace_values="${trace_values}GB/s"
        else
            trace_values="${x[3]},${x[4]},${x[5]},${x[6]},${x[7]},${x[8]},${x[9]},${x[10]}B,${x[11]}B"
        fi
        stats_row="${stats_row},${avg_kernel_time},${op_time},${trace_values}"

        # TODO: Fix this for ncu/nsys
        # if [ "$benchmark_metrics" == "1" ];
        # then
        #     # Benchmark kernel metrics
        #     echo "Benchmark kernel $kernel"
        #     nvprof --openacc-profiling off --kernels $kernel \
        #     $metrics_args \
        #     --log-file "/tmp/${BUS_ID}_kernel.txt" \
        #     python ${PM_HOME}/3rdparty/sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $metrics_bench_iters --warmup-iters $warmup_iters >& /dev/null

        #     metric_values=()
        #     for (( j=0; j<${#metrics[@]}; j++ ));
        #     do
        #         value="$( < /tmp/${BUS_ID}_kernel.txt grep -m1 "${metrics[j]}" | awk '{ x=gensub("    ","","G",$NF); printf x }' )"
        #         metric_values+=( "$value" )
        #     done

        #     for (( j=0; j<${#metrics[@]}; j++ ));
        #     do
        #         stats_row="$stats_row,${metric_values[j]}"
        #     done
        # fi
        echo "$stats_row" >> "$file_name"
    done
done < "$param_file_name"
