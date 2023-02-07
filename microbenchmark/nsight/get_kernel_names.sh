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
threshold="98" # In percentage
sum_perc="0.0"
touch "/tmp/${BUS_ID}_kernel_names.txt"

if [ "$op_type" == "memcpy" ];
then
    result_file="/tmp/${BUS_ID}_profile_results_gpumemtimesum.csv"
else
    result_file="/tmp/${BUS_ID}_profile_results_gpukernsum.csv"
fi

awk -F, '$3 > 5' $result_file > /tmp/${BUS_ID}_filtered_profile_results.csv # Filter all non-related kernels
mv /tmp/${BUS_ID}_filtered_profile_results.csv $result_file
perc_sum=$( awk '{ sum+=$1 } END { print sum }' $result_file ) # Sum up perc for related kernels

while IFS= read -r line
do
    if [ "$( echo "$line" | grep "Time" )" == "" ];
    then
        IFS=',' read -r -a array <<< "$line"
        perc="${array[0]}"
        kernel_name=$( echo "${array[6]/<*/}" | awk '{ sub("\"void ","",$0); printf $0 }' ) # Strip everything after <
        kernel_name="${kernel_name/*::/}" # Strip everything before (the last) ::
        kernel_name="${kernel_name/(*/}" # Strip everything after (
        if [ "$op_type" == "memcpy" ];
        then
            kernel_name=$( echo $kernel_name | sed 's/[][]//g' )
        fi
        sum_perc="$( echo "scale=4; $sum_perc + $perc / $perc_sum" | bc )"

        echo "Kernel name: $kernel_name"
        echo "Percentage: $( echo "scale=4; $perc / $perc_sum " | bc )"
        echo "$kernel_name" >> "/tmp/${BUS_ID}_kernel_names.txt"

        if (( "$( echo "$sum_perc > $threshold" | bc -l )" ));
        then
            break
        fi
    fi
done < $result_file
