#!/bin/bash

for batch_size in 16 32 64 128;
do
    for model in DLRM_default DLRM_MLPerf
    do
        ./dlrm_benchmark.sh ${model} $((batch_size*32))
        python trace_stats.py --model-name ${model} --batch-size $((batch_size*32))
        python e2e.py --model-name ${model} --batch-size $((batch_size*32))
    done

    for model in resnet50 inception_v3;
    do
        ./convnet_benchmark.sh ${model} ${batch_size}
        python trace_stats.py --model-name ${model} --batch-size ${batch_size}
        python e2e.py --model-name ${model} --batch-size ${batch_size}
    done

    for model in transformer
    do
        ./nlp_benchmark.sh transformer $((batch_size*4))
        python trace_stats.py --model-name ${model} --batch-size $((batch_size*4))
        python e2e.py --model-name ${model} --batch-size $((batch_size*4))
    done
done
