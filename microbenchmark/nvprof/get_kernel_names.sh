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

BUS_ID="$( nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader )"

rm -f "/tmp/${BUS_ID}_kernel_names.txt"
op_type=$1
threshold="0.98"
sum_perc="0.0"
touch "/tmp/${BUS_ID}_kernel_names.txt"

< "/tmp/${BUS_ID}_profile_results.txt" awk '/GPU activities: /,/API calls:/' | grep -v "API calls:" > "/tmp/${BUS_ID}_all_names.txt"
sum_runtime="0.0"
while IFS= read -r line
do
    line="${line/GPU activities: /}"
    IFS=', ' read -r -a array <<< "$line"

    kernel_name="${array[6]/<*/}"
    kernel_name="${kernel_name/(*/}"
    num_calls="${array[2]}"

    if [ "$op_type" != "memcpy" ];
    then
        if [ "$kernel_name" == "[CUDA" ] || [ "$num_calls" -lt "5" ];
        then
            continue
        fi
    fi

    sum_kernel="${array[1]}"
    if [[ "$sum_kernel" == *"us"* ]];
    then
        sum_kernel="$( echo "$sum_kernel" | tr --delete 'us' )"
    elif [[ "$sum_kernel" == *"ms"* ]];
    then
        sum_kernel="$( echo "$sum_kernel" | tr --delete 'ms' )"
        sum_kernel="$( echo "scale=4; $sum_kernel * 1000.0" | bc )"
    else # in second
        sum_kernel="$( echo "$sum_kernel" | tr --delete 's' )"
        sum_kernel="$( echo "scale=4; $sum_kernel * 1000000.0" | bc )"
    fi

    sum_runtime="$( echo "scale=4; $sum_runtime + $sum_kernel" | bc )"
done < "/tmp/${BUS_ID}_all_names.txt"

echo "Sum of kernel runtime: $sum_runtime us."

# Get dominating kernels
while IFS= read -r line
do
    line="${line/GPU activities: /}"
    IFS=', ' read -r -a array <<< "$line"

    num_calls="${array[2]}"

    kernel_name="${array[6]/<*/}"
    kernel_name="${kernel_name/(*/}"
    if [ "$op_type" != "memcpy" ];
    then
        if [ "$kernel_name" == "[CUDA" ] || [ "$num_calls" -lt "5" ];
        then
            continue
        fi
        if [ "$kernel_name" == "void" ] || [ "$kernel_name" == "int" ] || [ "$kernel_name" == "float" ];
        then
            kernel_name="${array[7]/<*/}"
        fi
    else # memcpy
        mem_type="${array[8]/]/}"
        kernel_name="CUDA memcpy ${mem_type}"
        echo "$kernel_name"
    fi

    sum_kernel="${array[1]}"
    if [[ "$sum_kernel" == *"us"* ]];
    then
        sum_kernel="$( echo "$sum_kernel" | tr --delete 'us' )"
    elif [[ "$sum_kernel" == *"ms"* ]];
    then
        sum_kernel="$( echo "$sum_kernel" | tr --delete 'ms' )"
        sum_kernel="$( echo "scale=4; $sum_kernel * 1000.0" | bc )"
    else # in second
        sum_kernel="$( echo "$sum_kernel" | tr --delete 's' )"
        sum_kernel="$( echo "scale=4; $sum_kernel * 1000000.0" | bc )"
    fi
    perc="$( echo "scale=4; $sum_kernel / $sum_runtime" | bc )"
    sum_perc="$( echo "scale=4; $sum_perc + $perc" | bc )"

    echo "Kernel name: $kernel_name"
    echo "Percentage: $perc"
    echo "$kernel_name" >> "/tmp/${BUS_ID}_kernel_names.txt"

    if (( "$( echo "$sum_perc > $threshold" | bc -l )" ));
    then
        break
    fi
done < "/tmp/${BUS_ID}_all_names.txt"
