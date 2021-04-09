echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo &> /dev/null
for i in {0..39}
do
  echo performance | sudo tee /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor &> /dev/null
done

if [[ `pwd` == *"daisy"* ]];
then
  source activate zhongyi
  sudo nvidia-smi -lgc 1297
elif [[ `pwd` == *"mario"* ]];
then
  source activate zhongyi_mario
  sudo nvidia-smi -ac 5505,1404
fi # Add any new conda env and lock GPU frequency here
