#!/bin/bash

for batch_size in 16 32 64 128;
do
    for ngpus in 1 2 4;
    do
        for model in DLRM_default DLRM_MLPerf DLRM_DDP
        do
            ./dlrm_benchmark.sh ${model} $((batch_size*32)) ${ngpus}
            python trace_stats.py --model-name ${model} --batch-size $((batch_size*32)) --iters 100
            python e2e.py --model-name ${model} --batch-size $((batch_size*32))
        done
    done

    for model in resnet50 inception_v3;
    do
        ./convnet_benchmark.sh ${model} ${batch_size}
        python trace_stats.py --model-name ${model} --batch-size ${batch_size} --iters 100
        python e2e.py --model-name ${model} --batch-size ${batch_size}
    done

    for model in transformer
    do
        ./nlp_benchmark.sh transformer $((batch_size*4))
        python trace_stats.py --model-name ${model} --batch-size $((batch_size*4)) --iters 100
        python e2e.py --model-name ${model} --batch-size $((batch_size*4))
    done

    for model in ncf deepfm
    do 
        ./rm_benchmark.sh ${model} $((batch_size*32))
        python trace_stats.py --model-name ${model} --batch-size $((batch_size*32)) --iters 100
        python e2e.py --model-name ${model} --batch-size $((batch_size*32))
    done
done
