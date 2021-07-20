#!/bin/bash
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader > /tmp/tmp_gpu_name.csv
gpu_name_get=0
for GPU_NAME in "V100" "P100" "Xp" "1080";
do
    if grep -q "$GPU_NAME" /tmp/tmp_gpu_name.csv
    then
        echo "$GPU_NAME" > /tmp/gpu_name.txt
        gpu_name_get=1
        break
    fi
done
if [ $gpu_name_get == "0" ];
then
    echo "Unrecognized GPU name! Exit..."
    exit
fi
