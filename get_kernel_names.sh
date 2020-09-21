#!/bin/bash
if [ "${CUDA_VISIBLE_DEVICES}" == "" ];
then
	CUDA_VISIBLE_DEVICES="0"
fi

< "/tmp/${CUDA_VISIBLE_DEVICES}_profile_results.txt" awk '/GPU activities: /,/API calls:/' | grep -v "API calls:" > "/tmp/${CUDA_VISIBLE_DEVICES}_all_names.txt"
rm -f "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"

threshold="0.98"
sum_runtime="0.0"
while IFS= read -r line
do
	line="${line/GPU activities: /}"
	IFS=', ' read -r -a array <<< "$line"

	kernel_name="${array[6]/<*/}"
	num_calls="${array[2]}"

	if [ "$kernel_name" == "[CUDA" ] || [ "$num_calls" -lt "10" ];
	then
		continue
	fi

	sum_kernel="${array[1]}"
	if [[ "$sum_kernel" == *"us"* ]];
	then
		sum_kernel="$( echo "$sum_kernel" | tr --delete 'us' )"
	else
		sum_kernel="$( echo "$sum_kernel" | tr --delete 'ms' )"
		sum_kernel="$( echo "scale=4; $sum_kernel * 1000.0" | bc )"
	fi

	sum_runtime="$( echo "scale=4; $sum_runtime + $sum_kernel" | bc )"
done < "/tmp/${CUDA_VISIBLE_DEVICES}_all_names.txt"

echo "Sum of kernel runtime: $sum_runtime"

# Get dominating kernels

sum_perc="0.0"
touch "/tmp/${CUDA_VISIBLE_DEVICES}_kernel_names.txt"
while IFS= read -r line
do
	line="${line/GPU activities: /}"
	IFS=', ' read -r -a array <<< "$line"

	kernel_name="${array[6]/<*/}"
	num_calls="${array[2]}"

	if [ "$kernel_name" == "[CUDA" ] || [ "$num_calls" -lt "10" ];
	then
		continue
	fi

	if [ "$kernel_name" == "void" ] || [ "$kernel_name" == "int" ] || [ "$kernel_name" == "float" ];
	then
		kernel_name="${array[7]/<*/}"
	fi

	sum_kernel="${array[1]}"
	if [[ "$sum_kernel" == *"us"* ]];
	then
		sum_kernel="$( echo "$sum_kernel" | tr --delete 'us' )"
	else
		sum_kernel="$( echo "$sum_kernel" | tr --delete 'ms' )"
		sum_kernel="$( echo "scale=4; $sum_kernel * 1000.0" | bc )"
	fi
	perc="$( echo "scale=4; $sum_kernel / $sum_runtime" | bc )"
	sum_perc="$( echo "scale=4; $sum_perc + $perc" | bc )"

	if (( "$( echo "$sum_perc > $threshold" | bc -l )" ));
	then
		break
	fi
done < "/tmp/${CUDA_VISIBLE_DEVICES}_all_names.txt"
















