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


# Add the GPU memory size in bytes here
# GPU_memory=42949672960 # A100 (40GB)
# GPU_memory=16777216000 # V100
# GPU_memory=12788432896 # XP
# GPU_memory=16777216000 # P100

# Get GPU memory size
GPU_memory=0
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader > /tmp/gpu_name.csv
for GPU_NAME in "A100" "GV100" "V100" "P100" "Xp";
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


if [ ! -f embedding_lookup_params.txt ];
then
    touch embedding_lookup_params.txt
    for B in 128 256 512;
    do
        for E in 1000 2000 5000 7500 10000 20000 50000 75000 100000 200000 500000 750000 1000000 2000000 5000000 7500000 10000000 20000000 50000000;
        do
            for T in 32 64 128 256;
            do
                for L in 8 16 32 38 64 128;
                do
                    for D in 32 64 128 256;
                    do
                        tmp=1
                        if [ "$( echo "(1024 / $D) > 1" | bc -l )" ];
                        then
                            tmp="$( echo "(1024 / $D)" | bc -l )"
                        fi
                        rows_per_block="$( echo "$tmp * 4 / 1" | bc )"

                        table_offsets_size="$( echo "$T * 4" | bc -l )"
                        offsets_size="$( echo "($B * $T + 1) * 4" | bc -l )"
                        indices_size="$( echo "$B * $T * $L * 4" | bc -l )"
                        weights_size="$( echo "$E * $T * $D * 4" | bc -l )"
                        outputs_size="$( echo "$B * $T * $D * 4" | bc -l )"
                        total_size="$( echo "$table_offsets_size + $offsets_size + $indices_size + $weights_size + $outputs_size" | bc -l )"

                        if [ "$total_size" -lt "$GPU_memory" ];
                        then
                            echo "$B $E $T $L $D $rows_per_block" >> embedding_lookup_params.txt
                        fi
                    done
                done
            done
        done
    done
fi


if [ ! -f embedding_lookup_params_big.txt ];
then
    touch embedding_lookup_params_big.txt
    for B in 1024 2048 4096;
    do
        for E in 1000 2000 5000 7500 10000 20000 50000 75000 100000 200000 500000 750000 1000000 2000000 5000000 7500000 10000000 20000000 50000000;
        do
            for T in 32 64 128 256;
            do
                for L in 8 16 32 38 64 128;
                do
                    for D in 32 64 128 256;
                    do
                        tmp=1
                        if [ "$( echo "(1024 / $D) > 1" | bc -l )" ];
                        then
                            tmp="$( echo "(1024 / $D)" | bc -l )"
                        fi
                        rows_per_block="$( echo "$tmp * 4 / 1" | bc )"

                        table_offsets_size="$( echo "$T * 4" | bc -l )"
                        offsets_size="$( echo "($B * $T + 1) * 4" | bc -l )"
                        indices_size="$( echo "$B * $T * $L * 4" | bc -l )"
                        weights_size="$( echo "$E * $T * $D * 4" | bc -l )"
                        outputs_size="$( echo "$B * $T * $D * 4" | bc -l )"
                        total_size="$( echo "$table_offsets_size + $offsets_size + $indices_size + $weights_size + $outputs_size" | bc -l )"

                        if [ "$total_size" -lt "$GPU_memory" ];
                        then
                            echo "$B $E $T $L $D $rows_per_block" >> embedding_lookup_params_big.txt
                        fi
                    done
                done
            done
        done
    done
fi


if [ ! -f embedding_lookup_params_dlrm_datasets.txt ];
then
    python generate_el_table_configs_from_datasets.py --dataset-path /nvme/deep-learning/dlrm_datasets/embedding_bag/2021
    python generate_el_table_configs_from_datasets.py --dataset-path /nvme/deep-learning/dlrm_datasets/embedding_bag/2022
    python sample_batches_from_dataset.py --per-gpu-memory $GPU_memory --num-samples 10000
    python sample_batches_from_dataset.py --per-gpu-memory $GPU_memory --num-samples 5000 --is-test
fi


# Addmm and Linear F&B
if [ ! -f fc_params.txt ];
then
    touch fc_params.txt
    python generate_fc_params.py --per-gpu-memory $GPU_memory
fi


# Bmm F&B
if [ ! -f fc_params_big.txt ];
then
    touch fc_params_big.txt
    python generate_fc_params.py --is-big --per-gpu-memory $GPU_memory
fi


if [ ! -f concat_params.txt ];
then
    touch concat_params.txt
    for batch_size in 1 128 256 512 1024 2048 4096;
    do
        for M in 64 198 256 512 1024 32768;
        do
            for N in 1 16 64 198 225 256 287 512;
            do
                for K in 15 64 197 256 512 1024 1281 2048 3482 4096 32768;
                do
                    A_size="$( echo "$batch_size * $M * $K * 4" | bc -l )"
                    B_size="$( echo "$batch_size * $N * $K * 4" | bc -l )"
                    C_size="$( echo "$batch_size * ($M + $N) * $K * 4" | bc -l )"
                    total_size="$( echo "$A_size + $B_size + $C_size" | bc -l )"

                    if [ "$total_size" -lt "$GPU_memory" ];
                    then
                        echo "$batch_size $M $N $K" >> concat_params.txt
                    fi
                done
            done
        done
    done
fi


if [ ! -f memcpy_params.txt ];
then
    touch memcpy_params.txt
    for batch_size in 1 128 256 512 1024 2048 4096;
    do
        for M in 64 128 256 512 1024 2048 4096 8192 16384 32768 65536;
        do
            for N in 64 128 256 512 1024 2048 4096 16384 32768 65536;
            do
                A_size="$( echo "$batch_size * $M * $N * 4" | bc -l )"

                if [ "$A_size" -lt "$GPU_memory" ];
                then
                    echo "$batch_size $M $N" >> memcpy_params.txt
                fi
            done
        done
    done
fi


if [ ! -f transpose_params.txt ];
then
    touch transpose_params.txt
    for batch_size in 1 128 256 512 1024 2048 4096;
    do
        for M in 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 48652 65536;
        do
            for N in 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 48652 65536;
            do
                A_size="$( echo "$batch_size * $M * $N * 4" | bc -l )"

                if [ "$A_size" -lt "$GPU_memory" ];
                then
                    echo "$batch_size $M $N" 0 >> transpose_params.txt
                    # echo "$batch_size $M $N" 1 >> transpose_params.txt
                    # echo "$batch_size $M $N" 2 >> transpose_params.txt
                fi
            done
        done
    done
fi


if [ ! -f tril_params.txt ];
then
    touch tril_params.txt
    for batch_size in 1 64 128 256 512 1024 2048 4096;
    do
        for MN in {5..40};
        do
            for diag in 0 1;
            do
                A_size="$( echo "$batch_size * $MN * $MN * 4" | bc -l )"

                if [ "$A_size" -lt "$GPU_memory" ];
                then
                    echo "$batch_size $MN $MN $diag" >> tril_params.txt
                fi
            done
        done
    done
fi


# Conv1d for parallel multi-head mm: stride = 1, padding = 0, dilation = 1, groups = num of mm groups
if [ ! -f conv1d_params.txt ];
then
    touch conv1d_params.txt
    for batch_size in 1 128 256 512 1024 2048 4096;
    do
        for L in 1 2 4 8 16 24 32 40 48 56 64;
        do
            for OC in 32 64 128 256;
            do
                for groups in 2 5 8 11 14 17 20 23 26 29 32;
                do
                    # IC = 1 for now
                    input_size="$( echo "$batch_size * 1 * $groups * $L * 4" | bc -l )"
                    filter_size="$( echo "$OC * 1 * $groups * 4" | bc -l )"
                    output_size="$( echo "$batch_size * $OC * $groups * $L * 4" | bc -l )"
                    total_size="$( echo "$input_size + $filter_size + $output_size" | bc -l )"

                    if [ "$total_size" -lt "$GPU_memory" ];
                    then
                        echo "$batch_size $L 1 $OC $groups" >> conv1d_params.txt
                    fi
                done
            done
        done
    done
fi


if [ ! -f conv2d_params.txt ];
then
    touch conv2d_params.txt
    for batch_size in 1 8 16 32;
    do
        for HW in 7 8 14 17 28 35 56 71 73 112 147 149 224 299;
        do
            for IC in 3 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048;
            do
                for OC in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048;
                do
                    for stride in 1 2;
                    do
                        for dilation in 1 2;
                        do
                            for FHW in 1 3 5;
                            do
                                for is_dw in 0 1;
                                do
                                    if [[ $stride == "2" && $FHW == "1" ]]; # 1x1 conv2d only has stride = 1
                                    then
                                        continue
                                    fi
                                    if [[ $is_dw == "1" && $FHW == "1" ]]; # 1x1 dw-conv2d doesn't exist
                                    then
                                        continue
                                    fi
                                    if [[ $is_dw == "1" && $IC != $OC ]]; # IC = OC in dw-conv2d
                                    then
                                        continue
                                    fi
                                    if [[ $is_dw == "1" && $dilation != "2" ]]; # No dilation for in dw-conv2d
                                    then
                                        continue
                                    fi
                                    if [[ $IC == "3" ]];
                                    then
                                        if [[ $HW != "299" && $HW != "224" ]]; # Infeasible input sizes
                                        then
                                            continue
                                        fi
                                    fi
                                    if [[ $HW == "299" || $HW == "224" ]];
                                    then
                                        if [[ $IC != "3" ]]; # Infeasible input sizes
                                        then
                                            continue
                                        fi
                                    fi
                                    ic_hw_prod="$( echo "$IC * $HW" | bc -l )" # Infeasible input sizes
                                    if [[ $ic_hw_prod -lt 1500 || $ic_hw_prod -gt 15000 ]];
                                    then
                                        if [[ $IC != "3" ]];
                                        then
                                            continue
                                        fi
                                    fi
                                    ic_oc_ratio="$( echo "scale=4; $IC / $OC" | bc )"
                                    if [[ $IC == "3" ]];
                                    then
                                        if [[ $( echo "$ic_oc_ratio > 0.03" | bc ) -eq 0 ]]; # Infeasible channel lengths
                                        then
                                            continue
                                        fi
                                    else
                                        if [[ $( echo "$ic_oc_ratio > 0.125" | bc ) -eq 0 || $( echo "$ic_oc_ratio < 8" | bc ) -eq 0 ]]; # Infeasible channel lengths
                                        then
                                            continue
                                        fi
                                    fi

                                    input_size="$( echo "$batch_size * $HW * $HW * $IC * 4" | bc -l )"
                                    filter_size=0
                                    if [ $is_dw == "1" ];
                                    then
                                        filter_size="$( echo "$FHW * $FHW * $OC * 4" | bc -l )"
                                    else
                                        filter_size="$( echo "$FHW * $FHW * $IC * $OC * 4" | bc -l )"
                                    fi
                                    output_size="$( echo "$batch_size * $HW * $HW * $OC * 4" | bc -l )"
                                    total_size="$( echo "$input_size + $filter_size + $output_size" | bc -l )"

                                    if [ "$total_size" -lt "$GPU_memory" ];
                                    then
                                        echo "$batch_size $HW $HW $IC $OC $stride $dilation $FHW $is_dw" >> conv2d_params.txt
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
fi


if [ ! -f conv2d_params_big.txt ];
then
    touch conv2d_params_big.txt
    for batch_size in 64 128;
    do
        for HW in 7 8 14 17 28 35 56 71 73 112 147 149 224 299;
        do
            for IC in 3 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048;
            do
                for OC in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048;
                do
                    for stride in 1 2;
                    do
                        for dilation in 1 2;
                        do
                            for FHW in 1 3 5;
                            do
                                for is_dw in 0 1;
                                do
                                    if [[ $stride == "2" && $FHW == "1" ]]; # 1x1 conv2d only has stride = 1
                                    then
                                        continue
                                    fi
                                    if [[ $is_dw == "1" && $FHW == "1" ]]; # 1x1 dw-conv2d doesn't exist
                                    then
                                        continue
                                    fi
                                    if [[ $is_dw == "1" && $IC != $OC ]]; # IC = OC in dw-conv2d
                                    then
                                        continue
                                    fi
                                    if [[ $is_dw == "1" && $dilation != "2" ]]; # No dilation for in dw-conv2d
                                    then
                                        continue
                                    fi
                                    if [[ $IC == "3" ]];
                                    then
                                        if [[ $HW != "299" && $HW != "224" ]]; # Infeasible input sizes
                                        then
                                            continue
                                        fi
                                    fi
                                    if [[ $HW == "299" || $HW == "224" ]];
                                    then
                                        if [[ $IC != "3" ]]; # Infeasible input sizes
                                        then
                                            continue
                                        fi
                                    fi
                                    ic_hw_prod="$( echo "$IC * $HW" | bc -l )" # Infeasible input sizes
                                    if [[ $ic_hw_prod -lt 1500 || $ic_hw_prod -gt 15000 ]];
                                    then
                                        if [[ $IC != "3" ]];
                                        then
                                            continue
                                        fi
                                    fi
                                    ic_oc_ratio="$( echo "scale=4; $IC / $OC" | bc )"
                                    if [[ $IC == "3" ]];
                                    then
                                        if [[ $( echo "$ic_oc_ratio > 0.03" | bc ) -eq 0 ]]; # Infeasible channel lengths
                                        then
                                            continue
                                        fi
                                    else
                                        if [[ $( echo "$ic_oc_ratio > 0.125" | bc ) -eq 0 || $( echo "$ic_oc_ratio < 8" | bc ) -eq 0 ]]; # Infeasible channel lengths
                                        then
                                            continue
                                        fi
                                    fi

                                    input_size="$( echo "$batch_size * $HW * $HW * $IC * 4" | bc -l )"
                                    filter_size=0
                                    if [ $is_dw == "1" ];
                                    then
                                        filter_size="$( echo "$FHW * $FHW * $OC * 4" | bc -l )"
                                    else
                                        filter_size="$( echo "$FHW * $FHW * $IC * $OC * 4" | bc -l )"
                                    fi
                                    output_size="$( echo "$batch_size * $HW * $HW * $OC * 4" | bc -l )"
                                    total_size="$( echo "$input_size + $filter_size + $output_size" | bc -l )"

                                    if [ "$total_size" -lt "$GPU_memory" ];
                                    then
                                        echo "$batch_size $HW $HW $IC $OC $stride $dilation $FHW $is_dw" >> conv2d_params_big.txt
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
fi


if [ ! -f bn_params.txt ];
then
    touch bn_params.txt
    for batch_size in 1 16 32 64 128;
    do
        for HW in 7 8 14 17 28 35 56 71 73 112 147 149 224 299;
        do
            for OC in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048;
            do
                oc_hw_prod="$( echo "$OC * $HW" | bc -l )" # Infeasible input sizes
                if [[ $oc_hw_prod -lt 1500 || $oc_hw_prod -gt 15000 ]];
                then
                    continue
                fi

                input_size="$( echo "$batch_size * $HW * $HW * $OC * 4" | bc -l )"
                if [ "$input_size" -lt "$GPU_memory" ];
                then
                    echo "$batch_size $HW $HW $OC" >> bn_params.txt
                fi
            done
        done
    done
fi


if [ ! -f gelu_params.txt ];
then
    touch gelu_params.txt
    for batch_size in 1 8 16 32 64 128;
    do
        for M in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048 2560 3072 3840 4096;
        do
            for N in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048 2560 3072 3840 4096;
            do
                input_size="$( echo "$batch_size * $M * $N * 4" | bc -l )"
                if [ "$input_size" -lt "$GPU_memory" ];
                then
                    echo "$batch_size $M $N" >> gelu_params.txt
                fi
            done
        done
    done
fi


if [ ! -f ln_params.txt ];
then
    touch ln_params.txt
    for batch_size in 1 8 16 32 64 128;
    do
        for M in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048 2560 3072 3840 4096;
        do
            for N in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048 2560 3072 3840 4096;
            do
                input_size="$( echo "$batch_size * $M * $N * 4" | bc -l )"
                if [ "$input_size" -lt "$GPU_memory" ];
                then
                    echo "$batch_size $M $N" >> ln_params.txt
                fi
            done
        done
    done
fi


if [ ! -f dropout_params.txt ];
then
    touch dropout_params.txt
    for batch_size in 1 8 16 32 64;
    do
        for M in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048 2560 3072 3840 4096;
        do
            for N in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048 2560 3072 3840 4096;
            do
                for p in 10 20 30 50 80;
                do
                    input_size="$( echo "$batch_size * $M * $N * 4" | bc -l )"
                    if [ "$input_size" -lt "$GPU_memory" ];
                    then
                        echo "$batch_size $M $N $( echo "scale=4; $p / 100.0" | bc -l )" >> dropout_params.txt
                    fi
                done
            done
        done
    done
fi


if [ ! -f pool_params.txt ];
then
    touch pool_params.txt
    for batch_size in 1 8 16 32 64 128;
    do
        for HW in 7 8 14 17 28 35 56 71 73 112 147 149 224 299;
        do
            for OC in 16 24 32 64 96 128 160 192 256 320 448 512 768 1024 1280 2048;
            do
                for stride in 1 2;
                do
                    for dilation in 1 2;
                    do
                        for FHW in 2 3;
                        do
                            for is_maxpool in 0 1;
                            do
                                if [[ $is_maxpool == "0" && $dilation != "1" ]]; # No dilation for avg pool
                                then
                                    continue
                                fi
                                oc_hw_prod="$( echo "$OC * $HW" | bc -l )" # Infeasible input sizes
                                if [[ $oc_hw_prod -lt 1500 || $oc_hw_prod -gt 15000 ]];
                                then
                                    continue
                                fi

                                input_size="$( echo "$batch_size * $HW * $HW * $OC * 4" | bc -l )"
                                if [ "$input_size" -lt "$GPU_memory" ];
                                then
                                    echo "$batch_size $HW $HW $OC $stride $dilation $FHW $is_dw" >> pool_params.txt
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
fi


if [ ! -f a2a_2_params.txt ];
then
    touch a2a_2_params.txt
    python generate_a2a_params.py --num-gpus 2 --per-gpu-memory $GPU_memory --num-samples -1 # All combinations
fi

if [ ! -f a2a_4_params.txt ];
then
    touch a2a_4_params.txt
    python generate_a2a_params.py --num-gpus 4 --per-gpu-memory $GPU_memory
fi

if [ ! -f a2a_8_params.txt ];
then
    touch a2a_8_params.txt
    python generate_a2a_params.py --num-gpus 8 --per-gpu-memory $GPU_memory
fi
