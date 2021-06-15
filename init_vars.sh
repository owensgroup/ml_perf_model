export PM_HOME=`pwd`
mkdir -p data
conda activate zhongyi

echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo &> /dev/null
for i in {0..39}
do
  echo performance | sudo tee /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor &> /dev/null
done

if [[ `pwd` == *"daisy"* ]];
then
  sudo nvidia-smi -lgc 1297
elif [[ `pwd` == *"mario"* ]];
then
  sudo nvidia-smi -ac 5505,1404
fi # Add any new conda env and lock GPU frequency here

./get_gpu_name.sh
export GPU_NAME=`cat /tmp/gpu_name.txt`
mkdir -p analysis/ml_predictors/${GPU_NAME}
mkdir -p data/${GPU_NAME}/e2e
mkdir -p data/${GPU_NAME}/kernel
