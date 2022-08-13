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

export PM_HOME=`pwd`
export PYTHONPATH=${PM_HOME}:${PYTHONPATH}
export NCU_BIN=/usr/local/cuda-11.3/bin/ncu # For PARAM
mkdir -p data
conda activate zhongyi
echo "localhost" > 3rdparty/param/train/comms/pt/hfile.txt

echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo &> /dev/null
for i in {0..39}
do
  echo performance | sudo tee /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor &> /dev/null
done

./get_gpu_name.sh
export GPU_NAME=`cat /tmp/gpu_name.txt`
mkdir -p analysis/ml_predictors/${GPU_NAME}
mkdir -p data/${GPU_NAME}/e2e
mkdir -p data/${GPU_NAME}/kernel

if [[ $GPU_NAME == "A100" ]];
then
  sudo nvidia-smi -ac 1215,1095
elif [[ $GPU_NAME == "V100" ]];
then
  sudo nvidia-smi -ac 877,1297
elif [[ $GPU_NAME == "P100" ]];
then
  sudo nvidia-smi -ac 715,1189
elif [[ $GPU_NAME == "Xp" ]];
then
  sudo nvidia-smi -ac 5505,1404
fi # Add any new conda env and lock GPU frequency here
