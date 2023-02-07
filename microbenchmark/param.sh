# Run PARAM benchmark for communication collectives (a2a, all_reduce)
cd ${PM_HOME}/3rdparty/param/train/comms/pt/
source ./init.sh
num_gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo $num_gpus
./comms_collective_bench.sh $num_gpus
cd ../../../../../microbenchmark

python generate_collective_params.py