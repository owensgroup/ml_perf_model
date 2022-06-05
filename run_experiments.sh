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

dlrm_max_batch_size=4096
trimmed_iters=${1:-30} # Default iters 30
share_overheads=${2:-1} # Share overheads by default

cd benchmark

# Benchmark and trace analysis
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
            eval "$cmd trace_stats.py --model-name ${model} --batch-size $((batch_size*32)) --iters $trimmed_iters --num-gpus $ngpus"
            # Weak scaling
            if [ "$num_gpus" -gt 1 ] && (( "$( echo "$batch_size * 32 * $ngpus > $dlrm_max_batch_size" | bc -l )" )) ;
            then
                ./dlrm_benchmark.sh ${model} $((batch_size*32*ngpus)) ${ngpus}
                eval "$cmd trace_stats.py --model-name ${model} --batch-size $((batch_size*32*ngpus)) --iters $trimmed_iters --num-gpus $ngpus"
            fi
        done
    done

    for model in resnet50 inception_v3;
    do
        ./convnet_benchmark.sh ${model} ${batch_size}
        python trace_stats.py --model-name ${model} --batch-size ${batch_size} --iters $trimmed_iters
    done

    # for model in transformer
    # do
    #     ./nlp_benchmark.sh transformer $((batch_size*4))
    #     python trace_stats.py --model-name ${model} --batch-size $((batch_size*4)) --iters $trimmed_iters
    # done

    for model in ncf deepfm
    do
        ./rm_benchmark.sh ${model} $((batch_size*32))
        python trace_stats.py --model-name ${model} --batch-size $((batch_size*32)) --iters $trimmed_iters
    done
done

# Create shared overheads
python create_shared_overheads.py --iters $trimmed_iters

# Run prediction
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
                cmd="mpirun -np $ngpus -N $ngpus python e2e.py"
            else
                cmd="python e2e.py"
            fi

            # Share overheads?
            if [[ "$share_overheads" -ne "1" ]];
            then
                cmd="$cmd --use-independent-overheads"
            fi
            cmd="$cmd --model-name ${model} --iters $trimmed_iters --num-gpus $ngpus"

            # Strong scaling
            eval "$cmd --batch-size $((batch_size*32))"
            # Weak scaling
            if [ "$num_gpus" -gt 1 ] && (( "$( echo "$batch_size * 32 * $ngpus > $dlrm_max_batch_size" | bc -l )" )) ;
            then
                eval "$cmd --batch-size $((batch_size*32*ngpus))"
            fi
        done
    done

    for model in resnet50 inception_v3;
    do
        python e2e.py --model-name ${model} --batch-size ${batch_size} --iters $trimmed_iters
    done

    # for model in transformer
    # do
    #     ./nlp_benchmark.sh transformer $((batch_size*4))
    #     python trace_stats.py --model-name ${model} --batch-size $((batch_size*4)) --iters $trimmed_iters
    #     python e2e.py --model-name ${model} --batch-size $((batch_size*4)) --iters $trimmed_iters
    # done

    for model in ncf deepfm
    do
        python e2e.py --model-name ${model} --batch-size $((batch_size*32)) --iters $trimmed_iters
    done
done

cd ..
