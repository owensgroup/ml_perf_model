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
trimmed_iters=30 # Default iters 30
share_overheads= # Share overheads
emb_type= # FBGEMM for DLRM
bucket_size_mb=25 # Bucket size in MB
early_barrier= # Whether launch a barrier at the beginning of the iteration
aggregated_allreduce= # Whether extract an execution graph with aggregated allreduce for DDP (i.e. iteration 0)
while getopts i:ots:ra flag
do
    case "${flag}" in
        i) trimmed_iters=${OPTARG};;
        o) share_overhead="-o";;
        t) emb_type="-t";;
        s) bucket_size_mb=${OPTARG};;
        r) early_barrier="-r";;
        a) aggregated_allreduce="-a";;
    esac
done

cd benchmark

# Benchmark and trace analysis
for batch_size in 8 16 32 64;
do
    for ngpus in 1 2 4 8;
    do
        # Has enough GPUs?
        num_gpus="$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )"
        if [ "$ngpus" -gt "$num_gpus" ];
        then
            continue
        fi

        for model in DLRM_open_source DLRM_default DLRM_MLPerf DLRM_DDP
        do
            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                trace_cmd="mpirun -np $ngpus -N $ngpus python"
            else
                trace_cmd="python"
            fi

            # Options for both benchmark and trace_stats
            options="   -m ${model}\
                        -g ${ngpus}\
                        ${emb_type}\
                        -s ${bucket_size_mb}\
                        ${early_barrier}\
                        ${aggregated_allreduce}"

            trace_cmd="$trace_cmd \
                        trace_stats.py \
                        -i ${trimmed_iters} \
                        ${options}"

            # Strong scaling
            ./dlrm_benchmark.sh -b $((batch_size*64)) ${options}
            eval "$trace_cmd -b $((batch_size*64))"

            # Weak scaling
            if [ "$num_gpus" -gt 1 ] && (( "$( echo "$batch_size * 64 * $ngpus > $dlrm_max_batch_size" | bc -l )" )) ;
            then
                ./dlrm_benchmark.sh -b $((batch_size*64*ngpus)) ${options}
                eval "$trace_cmd -b $((batch_size*64*ngpus))"
            fi
        done

        for model in bert gpt2;
        do
            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                trace_cmd="mpirun -np $ngpus -N $ngpus python"
            else
                trace_cmd="python"
            fi

            options="   -m ${model}\
                        -g ${ngpus}\
                        -s ${bucket_size_mb}\
                        ${early_barrier}\
                        ${aggregated_allreduce}"

            trace_cmd="$trace_cmd \
                        trace_stats.py \
                        -i ${trimmed_iters} \
                        ${options}"

            ./nlp_benchmark.sh -b ${batch_size} ${options}
            eval "$trace_cmd -b ${batch_size}"
        done
    done

    for model in resnet50 inception_v3;
    do
        ./convnet_benchmark.sh ${model} $((batch_size*2))
        python trace_stats.py -m ${model} -i ${trimmed_iters} -b $((batch_size*2))
    done

    for model in ncf deepfm;
    do
        ./rm_benchmark.sh ${model} $((batch_size*64))
        python trace_stats.py -m ${model} -i ${trimmed_iters} -b $((batch_size*64))
    done
done


# Create shared overheads
python create_shared_overheads.py --iters $trimmed_iters


# Run prediction
for batch_size in 8 16 32 64;
do
    for ngpus in 1 2 4 8;
    do
        # Has enough GPUs?
        num_gpus="$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )"
        if [ "$ngpus" -gt "$num_gpus" ];
        then
            continue
        fi

        for model in DLRM_open_source DLRM_default DLRM_MLPerf DLRM_DDP
        do
            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                cmd="mpirun -np $ngpus -N $ngpus python"
            else
                cmd="python"
            fi

            options="   -i ${trimmed_iters}\
                        -m ${model}\
                        -g ${ngpus}\
                        ${emb_type}\
                        -s ${bucket_size_mb}\
                        ${early_barrier}\
                        ${aggregated_allreduce}\
                        ${share_overheads}"

            cmd="   $cmd\
                    e2e.py\
                    ${options}"

            # Strong scaling
            eval "$cmd -b $((batch_size*64))"

            # Weak scaling
            if [ "$num_gpus" -gt 1 ] && (( "$( echo "$batch_size * 64 * $ngpus > $dlrm_max_batch_size" | bc -l )" )) ;
            then
                eval "$cmd -b $((batch_size*64*ngpus))"
            fi
        done

        for model in bert gpt2;
        do
            # Multi-GPU?
            if [ "$ngpus" -gt "1" ];
            then
                cmd="mpirun -np $ngpus -N $ngpus python"
            else
                cmd="python"
            fi

            options="   -i ${trimmed_iters}\
                        -m ${model}\
                        -g ${ngpus}\
                        -s ${bucket_size_mb}\
                        ${early_barrier}\
                        ${aggregated_allreduce}\
                        ${share_overheads}"

            cmd="   $cmd\
                    e2e.py\
                    ${options}"

            eval "$cmd -b ${batch_size}"
        done
    done

    for model in resnet50 inception_v3;
    do
        python e2e.py -m ${model} -i ${trimmed_iters} -b $((batch_size*2))
    done

    for model in ncf deepfm;
    do
        python e2e.py -m ${model} -i ${trimmed_iters} -b $((batch_size*64))
    done
done

cd ..
