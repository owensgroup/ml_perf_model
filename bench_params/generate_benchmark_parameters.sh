#!/bin/bash
GPU_memory=16777216000

if [ ! -f embedding_lookup_params.txt ];
then
    touch embedding_lookup_params.txt
    for B in 128 256 512 1024 2048 4096;
    do
        for E in 1000 2000 5000 7500 10000 20000 50000 75000 100000 200000 500000 10000000 50000000;
        do
            for T in 32 64 128 197;
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


if [ ! -f fc_params.txt ];
then
    touch fc_params.txt
    for batch_size in 1 128 256 512;
    do
        for M in 64 198 256 512 1024 32768;
        do
            for N in 1 16 64 198 225 256 287 512;
            do
                for K in 15 64 197 256 512 1024 1281 2048 3482 4096 32768;
                do
                    A_size="$( echo "$batch_size * $M * $K * 4" | bc -l )"
                    B_size="$( echo "$batch_size * $N * $K * 4" | bc -l )"
                    C_size="$( echo "$batch_size * $M * $N * 4" | bc -l )"
                    total_size="$( echo "$A_size + $B_size + $C_size" | bc -l )"

                    if [ "$total_size" -lt "$GPU_memory" ];
                    then
                        echo "$batch_size $M $N $K" >> fc_params.txt
                    fi
                done
            done
        done
    done
fi

if [ ! -f cat_params.txt ];
then
    touch cat_params.txt
    for batch_size in 1 128 256 512;
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
                        echo "$batch_size $M $N $K" >> cat_params.txt
                    fi
                done
            done
        done
    done
fi