#!/bin/bash

dlrm_max_batch_size=4096

for batch_size in 16 32 64 128;
do
    for ngpus in 1 2 4 8;
    do
        # Has enough GPUs?
        num_gpus="$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )"
        if [ "$ngpus" -gt "$num_gpus" ];
        then
            continue
        fi

        for model in DLRM_default DLRM_MLPerf DLRM_DDP
        do
            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                cmd="mpirun -np $ngpus -N $ngpus python"
            else
                cmd="python"
            fi

            # Strong scaling
            ./dlrm_benchmark.sh ${model} $((batch_size*32)) ${ngpus}
            eval "$cmd trace_stats.py --model-name ${model} --batch-size $((batch_size*32)) --iters 100"
            eval "$cmd e2e.py --model-name ${model} --batch-size $((batch_size*32))"

            # Weak scaling
            if [ "$num_gpus" -gt 1 ] && (( "$( echo "$batch_size * 32 * $ngpus > $dlrm_max_batch_size" | bc -l )" )) ;
            then
                ./dlrm_benchmark.sh ${model} $((batch_size*32*ngpus)) ${ngpus}
                eval "$cmd trace_stats.py --model-name ${model} --batch-size $((batch_size*32*ngpus)) --iters 100 --num-gpus $ngpus"
                eval "$cmd e2e.py --model-name ${model} --batch-size $((batch_size*32*ngpus)) --num-gpus $ngpus"
            fi
        done
    done

    # for model in resnet50 inception_v3;
    # do
    #     ./convnet_benchmark.sh ${model} ${batch_size}
    #     python trace_stats.py --model-name ${model} --batch-size ${batch_size} --iters 100
    #     python e2e.py --model-name ${model} --batch-size ${batch_size}
    # done

    # for model in transformer
    # do
    #     ./nlp_benchmark.sh transformer $((batch_size*4))
    #     python trace_stats.py --model-name ${model} --batch-size $((batch_size*4)) --iters 100
    #     python e2e.py --model-name ${model} --batch-size $((batch_size*4))
    # done

    # for model in ncf deepfm
    # do 
    #     ./rm_benchmark.sh ${model} $((batch_size*32))
    #     python trace_stats.py --model-name ${model} --batch-size $((batch_size*32)) --iters 100
    #     python e2e.py --model-name ${model} --batch-size $((batch_size*32))
    # done
done
