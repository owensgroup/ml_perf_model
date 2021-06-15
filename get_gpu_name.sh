#!/bin/bash
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader > /tmp/tmp_gpu_name.csv
for GPU_NAME in "V100" "P100" "Xp" "1080";
do
    if grep -q "$GPU_NAME" /tmp/tmp_gpu_name.csv
    then
        echo "$GPU_NAME" > /tmp/gpu_name.txt
        break
    fi
done
