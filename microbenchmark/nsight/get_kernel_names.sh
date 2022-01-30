#!/bin/bash
if [ "${CUDA_VISIBLE_DEVICES}" == "" ];
then
    CUDA_VISIBLE_DEVICES="0"
fi

rm -f "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"
op_type=$1
threshold="98" # In percentage
sum_perc="0.0"
touch "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"

if [ "$op_type" == "memcpy" ];
then
    result_file="/tmp/${CUDA_VISIBLE_DEVICES}_profile_results_gpumemtimesum.csv"
else
    result_file="/tmp/${CUDA_VISIBLE_DEVICES}_profile_results_gpukernsum.csv"
fi

awk -F, '$3 > 5' $result_file > /tmp/${CUDA_VISIBLE_DEVICES}_filtered_profile_results.csv # Filter all non-related kernels
mv /tmp/${CUDA_VISIBLE_DEVICES}_filtered_profile_results.csv $result_file
perc_sum=$( awk '{ sum+=$1 } END { print sum }' $result_file ) # Sum up perc for related kernels

while IFS= read -r line
do
    if [ "$( echo "$line" | grep "Time" )" == "" ];
    then
        IFS=',' read -r -a array <<< "$line"
        perc="${array[0]}"
        kernel_name=$( echo "${array[6]/<*/}" | awk '{ sub("\"void ","",$0); printf $0 }' )
        if [ "$op_type" == "memcpy" ];
        then
            kernel_name=$( echo $kernel_name | sed 's/[][]//g' )
        fi
        sum_perc="$( echo "scale=4; $sum_perc + $perc / $perc_sum" | bc )"

        echo "Kernel name: $kernel_name"
        echo "Percentage: $( echo "scale=4; $perc / $perc_sum " | bc )"
        echo "$kernel_name" >> "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"

        if (( "$( echo "$sum_perc > $threshold" | bc -l )" ));
        then
            break
        fi
    fi
done < $result_file
