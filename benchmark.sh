if [ ! -f perf_params.txt ];
then
    echo "no parameter file"
    exit
fi

if [ ! -f nvprof_metrics.txt ];
then
    echo "no parameter file"
    exit
fi

op_type=$1
is_forward=$2
shmem="0"
sgd="0"
fc_test="0"
header=""
param_file_name=""
file_prefix="./data/${op_type}_${is_forward}"
if [ "${CUDA_VISIBLE_DEVICES}" == "" ];
then
    CUDA_VISIBLE_DEVICES="0"
fi

if [ "$op_type" == "embedding_lookup" ];
then
    header="kernel_name,batch_size,num_embeddings,num_tables,bag_size,embedding_dim,rows_per_block"
    param_file_name="./bench_params/embedding_lookup_params.txt"
    if [ "$is_forward" == "1" ] && [ "$shmem" == "1" ];
    then
        file_prefix="$file_prefix"
    fi
    if [ "$is_forward" == "0" ];
    then
        if [ "$sgd" == "0" ];
        then
            file_prefix="${file_prefix}_sgd"
        else
            file_prefix="${file_prefix}_adagrad"
        fi
    fi
elif [ "$op_type" == "fully_connected" ];
then
    header="kernel_name,batch_size,M,N,K"
    param_file_name="./bench_params/fc_params.txt"
    if [ "$fc_test" == "1" ];
    then
        param_file_name="./bench_params/fc_test_params.txt"
        file_prefix="${file_prefix}_test"
    fi
elif [ "$op_type" == "concat" ];
then
    header="kernel_name,batch_size,M,N,K"
    param_file_name="./bench_params/cat_params.txt"
elif [ "$op_type" == "cross_entropy" ];
then
    header="kernel_name,batch_size"
    param_file_name="./bench_params/ce_params.txt"
else
    header="kernel_name,batch_size"
    param_file_name="./bench_params/memcpy_params.txt"
fi
file_name="${file_prefix}_results.csv"

header="${header},kernel_runtime,op_runtime"
if [ "$op_type" != "memcpy" ];
then
    metrics_args=""
    metrics=()

    while IFS= read -r line
    do
        metrics_args="$metrics_args --metrics $line"
        metrics+=( "$line" )
        header="$header,$line"
    done < "nvprof_metrics.txt"
    if [ "$op_type" == "fully_connected" ];
    then
        header="${header},block_x,block_y,block_z,thread_x,thread_y,thread_z,regs,smem"
    fi
fi
if [ ! -f "$file_name" ];
then
    touch "$file_name"
    echo "${header}" >> "$file_name"
fi

runtime_batch_iter=30
metrics_bench_iter=3

Tsp="$( < perf_params.txt grep 'GFLOPS - SP' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
Tdp="$( < perf_params.txt grep 'DP' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
Tint="$( < perf_params.txt grep 'MAD' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
Tadd="$( < perf_params.txt grep 'ADD' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
Tldst="$( < perf_params.txt grep 'SHMEM' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
Bmem="$( < perf_params.txt grep 'GBSEC - DRAM' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
Bcache="$( < perf_params.txt grep 'GBSEC - L2' | awk '{ x=gensub("    ","","G",$4); printf x }' )"
echo "Device params:"
echo "    GFLOPS (SP): $Tsp"
echo "    GFLOPS (DP): $Tdp"
echo "    GFLOPS (MAD): $Tint"
echo "    GFLOPS (ADD): $Tadd"
echo "    SHMEM BW: $Tldst"
echo "    DRAM BW: $Bmem"
echo "    L2 BW: $Bcache"

# Benchmark operator
while IFS= read -r line
do
    IFS=', ' read -r -a array <<< "$line"
    bench_param=""

    if [ "$op_type" == "embedding_lookup" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --num-embeddings ${array[1]} --num-tables ${array[2]} --bag-size ${array[3]} --embedding-dim ${array[4]} --rows-per-block ${array[5]}"
        if [ "${array[0]}" -gt "32" ] && [ "$is_forward" == "0" ]; # Skip when backward and rows_per_block too big
        then
            continue
        fi
    elif [ "$op_type" == "fully_connected" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "concat" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "cross_entropy" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]}"
    else # Memcpy
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]}"
    fi
    if [ "$is_forward" == "0" ];
    then
        bench_param="${bench_param} --backward"
    fi
    if [ "$shmem" == "1" ];
    then
        bench_param="${bench_param} --shmem"
    fi
    if [ "$sgd" == "1" ];
    then
        bench_param="${bench_param} --sgd"
    fi
    echo "$bench_param"

    # Benchmark general
    nvprof --log-file "/tmp/${CUDA_VISIBLE_DEVICES}_profile_results.txt" python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iter >& "/tmp/${CUDA_VISIBLE_DEVICES}_op.txt"
    op_time="$( < /tmp/${CUDA_VISIBLE_DEVICES}_op.txt grep 'Time: ' | awk '{ x=gensub("    ","","G",$NF); x=gensub("us","","G",x); printf x }' )"

    if [ "$op_type" == "fully_connected" ];
    then
        echo "Get GPU trace of kernels ..."
        nvprof --print-gpu-trace --log-file "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_trace.txt" \
        python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters 1 >& /dev/null
    fi

    ./get_kernel_names.sh
    kernels=()
    while IFS= read -r line
    do
        kernels+=( "$line" )
    done < "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"
    len=${#kernels[@]}

    for (( i=0; i<len; i++ ));
    do
        kernel=${kernels[i]}
        if [ "$op_type" == "concat" ];
        then
            kernel=${kernel##*::}
        fi

        measured_time="$( < "/tmp/${CUDA_VISIBLE_DEVICES}_profile_results.txt" grep "$kernel" | awk '{ x=gensub("    ","","G",$6); printf x }' )"
        if [[ "$measured_time" == *"us"* ]];
        then
            measured_time="$( echo "$measured_time" | tr --delete 'us' )"
        else
            measured_time="$( echo "$measured_time" | tr --delete 'ms' )"
            measured_time="$( echo "scale=4; $measured_time * 1000.0" | bc )"
        fi
        echo $measured_time

        echo "Benchmark kernel $kernel"
        nvprof --kernels $kernel \
        $metrics_args \
        --log-file "/tmp/${CUDA_VISIBLE_DEVICES}_kernel.txt" \
        python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $metrics_bench_iter >& /dev/null

        metric_values=()
        for (( j=0; j<${#metrics[@]}; j++ ));
        do
            value="$( < /tmp/${CUDA_VISIBLE_DEVICES}_kernel.txt grep -m1 "${metrics[j]}" | awk '{ x=gensub("    ","","G",$NF); printf x }' )"
            metric_values+=( "$value" )
        done

        stats_row="$kernel"
        array_len=${#array[@]}
        for (( j=0; j<array_len; j++ ));
        do
            stats_row="$stats_row,${array[j]}"
        done
        stats_row="$stats_row,$measured_time,$op_time"
        for (( j=0; j<${#metrics[@]}; j++ ));
        do
            stats_row="$stats_row,${metric_values[j]}"
        done
        if [ "$op_type" == "fully_connected" ];
        then
            trace_values=""
            while IFS= read -r line
            do
                if [ "$( echo "$line" | grep "$kernel" )" == ""];
                then
                    continue
                fi

                IFS=', ' read -r -a x <<< "$line"
                trace_values="${x[2]/(/},${x[3]},${x[4]/)/},${x[5]/(/},${x[6]},${x[7]/)/},${x[8]},${x[9]}"
                break
            done < "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_trace.txt"
            stats_row="${stats_row},${trace_values}"
        fi
        echo "$stats_row" >> "$file_name"
    done
    exit
done < "$param_file_name"
