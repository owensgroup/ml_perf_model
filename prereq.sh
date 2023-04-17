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


pip install h5py tqdm scikit-learn matplotlib tensorboard tensorboardX torchviz onnx gputil pandas scikit-build pydot mpi4py # General dependencies
pip install transformers accelerate datasets # Transformers
git clone --recursive https://github.com/owensgroup/ml_perf_model.git

cd ml_perf_model/3rdparty/bench_params
./generate_benchmark_parameters.sh # Generate benchmark parameters.
cd ../sparse-ads-baselines
python setup.py install # Install table batched embedding lookup kernel.
cd ../mlperf-logging
python setup.py install # Install MLPerf logging for DLRM
cd ../param/train/comms/pt
./init.sh # Initialization for PARAM
cd ../../../../../
source ./init_vars.sh # Turn off turbo, turn on performance, lock frequency, etc.

# Torchvision for ConvNet benchmark
cd ..
git clone https://github.com/pytorch/vision.git torchvision
cd torchvision
python setup.py clean --all install
cd ../ml_perf_model
