if [ ! -f nvprof_metrics.txt ];
then
    echo "no nvprof metrics file"
    exit
fi

# Lock GPU frequency
sudo nvidia-smi –lgc 1297 # V100

# Make the data dir
mkdir -p data

runtime_batch_iters=30
metrics_bench_iters=3
warmup_iters=5
benchmark_metrics="0"

op_type=$1
is_forward=$2
shmem="0"
sgd="1"
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
        file_prefix="${file_prefix}_shmem"
    fi
    if [ "$is_forward" == "0" ];
    then
        if [ "$sgd" == "1" ];
        then
            file_prefix="${file_prefix}_sgd"
        else
            file_prefix="${file_prefix}_adagrad"
        fi
	if [ "$shmem" == "1" ];
	then
	    file_prefix="${file_prefix}_shmem"
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
elif [ "$op_type" == "conv" ];
then
    header="kernel_name,batch_size,H,W,IC,OC"
    param_file_name="./bench_params/conv_params.txt"
elif [ "$op_type" == "concat" ];
then
    header="kernel_name,batch_size,M,N,K"
    param_file_name="./bench_params/cat_params.txt"
elif [ "$op_type" == "cross_entropy" ];
then
    header="kernel_name,batch_size"
    param_file_name="./bench_params/ce_params.txt"
elif [ "$op_type" == "reshape" ];
then
    header="kernel_name,batch_size,M,N,trans_type"
    param_file_name="./bench_params/transpose_params.txt"
else # memcpy
    header="kernel_name,batch_size,M,N"
    param_file_name="./bench_params/memcpy_params.txt"
fi
file_name="${file_prefix}.csv"

header="${header},kernel_runtime,op_runtime"
if [ "$op_type" != "memcpy" ];
then
    header="${header},block_x,block_y,block_z,thread_x,thread_y,thread_z,regs,smem"

    if [ "$benchmark_metrics" == "1" ];
    then
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

    if [ "$op_type" == "embedding_lookup" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --num-embeddings ${array[1]} --num-tables ${array[2]} --bag-size ${array[3]} --embedding-dim ${array[4]} --rows-per-block ${array[5]}"
        if [ "${array[5]}" -gt "32" ] && [ "$is_forward" == "0" ]; # Skip when backward and rows_per_block too big
        then
            continue
        fi
        if [ "$shmem" == "1" ];
        then
            bench_param="${bench_param} --shmem"
        fi
        if [ "$sgd" == "1" ];
        then
            bench_param="${bench_param} --sgd"
        fi
    elif [ "$op_type" == "fully_connected" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "conv" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --H ${array[1]} --W ${array[2]} --IC ${array[3]} --OC ${array[4]}"
    elif [ "$op_type" == "concat" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --K ${array[3]}"
    elif [ "$op_type" == "cross_entropy" ];
    then
        bench_param="--op-type $op_type --batch-size ${array[0]}"
    elif [ "$op_type" == "reshape" ]; # reshape
    then
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]} --trans-type ${array[3]}"
    else # Memcpy
        bench_param="--op-type $op_type --batch-size ${array[0]} --M ${array[1]} --N ${array[2]}"
    fi
    if [ "$is_forward" == "0" ];
    then
        bench_param="${bench_param} --backward"
    fi
    echo "$bench_param"

    # Benchmark operator runtime: no nvprof
    python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iters --warmup-iters $warmup_iters >& "/tmp/${CUDA_VISIBLE_DEVICES}_op.txt"
    op_time="$( < /tmp/${CUDA_VISIBLE_DEVICES}_op.txt grep 'Time: ' | awk '{ x=gensub("    ","","G",$NF); x=gensub("us","","G",x); printf x }' )"

    # Benchmark general: get the major kernel names
    nvprof --openacc-profiling off --log-file "/tmp/${CUDA_VISIBLE_DEVICES}_profile_results.txt" python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $metrics_bench_iters --warmup-iters $warmup_iters >& /dev/null

    # Get gpu trace
    echo "Get GPU trace of kernels ..."
    nvprof --openacc-profiling off --print-gpu-trace --log-file "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_trace.txt" \
    python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $runtime_batch_iters --warmup-iters $warmup_iters >& /dev/null

    ./get_kernel_names.sh "$op_type"
    kernels=()
    while IFS= read -r line
    do
        kernels+=( "$line" )
    done < "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"
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
                    trace_values="${x[2]/(/},${x[3]},${x[4]/)/},${x[5]/(/},${x[6]},${x[7]/)/},${x[8]},${x[9]}"
                fi
            fi

            # Get average kernel time excluding warmup runs
            kernel_time="${x[1]}"
            if [[ "$kernel_time" == *"us"* ]];
            then
                kernel_time="$( echo "$kernel_time" | tr --delete 'us' )"
            else
                kernel_time="$( echo "$kernel_time" | tr --delete 'ms' )"
                kernel_time="$( echo "scale=4; $kernel_time * 1000.0" | bc )"
            fi
            if [ "$kernel_count" -gt "$warmup_iters" ];
            then
                avg_kernel_time="$( echo "scale=4; $avg_kernel_time + $kernel_time / $runtime_batch_iters" | bc )"
            fi
        done < "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_trace.txt"
        stats_row="${stats_row},${avg_kernel_time},${op_time},${trace_values}"

        if [ "$benchmark_metrics" == "1" ];
        then
            # Benchmark kernel metrics
            echo "Benchmark kernel $kernel"
            nvprof --openacc-profiling off --kernels $kernel \
            $metrics_args \
            --log-file "/tmp/${CUDA_VISIBLE_DEVICES}_kernel.txt" \
            python sparse-ads-baselines/kernel_benchmark.py $bench_param --iters $metrics_bench_iters --warmup-iters $warmup_iters >& /dev/null

            metric_values=()
            for (( j=0; j<${#metrics[@]}; j++ ));
            do
                value="$( < /tmp/${CUDA_VISIBLE_DEVICES}_kernel.txt grep -m1 "${metrics[j]}" | awk '{ x=gensub("    ","","G",$NF); printf x }' )"
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
