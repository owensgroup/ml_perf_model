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

import argparse, json, re, os
from analysis.exec_graph_utils import ExecutionGraph # Not commited to Pytorch yet
import analysis.extend_distributed as ext_dist
from analysis.utils import PM_HOME, GPU_NAME
from analysis.inference import get_e2e_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict end-to-end training time of DLRM models.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--use-independent-overheads", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.num_gpus > 1:
        ext_dist.init_distributed(use_gpu=False) # Don't need GPU for E2E prediction.
    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
        print("======= {}, {} GPU(s), batch size: {}, iters: {} =======".format(
            args.model_name, args.num_gpus, args.batch_size, args.iters))

    prefix = "{}/data/{}/e2e/{}/{}_{}{}".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size, "_distributed" if args.num_gpus > 1 else "")
    module_marker = "DLRM " if "DLRM" in args.model_name else "## Forward ##"

    exec_graph_file = "{}{}_graph.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if args.num_gpus > 1 else "")
    with open(exec_graph_file) as f:
        graph = ExecutionGraph(json.load(f))
    if args.use_independent_overheads:
        overheads_file = "{}_overhead_stats_{}.json".format(prefix, args.iters)
    else:
        overheads_file = "{}/data/{}/e2e/shared_overheads.json".format(PM_HOME, GPU_NAME)
    if not os.path.exists(overheads_file):
        print("Overheads file doesn't exist! Please run the trace analysis first.")
        exit(1)
    with open(overheads_file) as f:
        overheads = json.load(f)

    real_e2e_time, real_gpu_active_time = -1, -1
    log_file = "{}.log".format(prefix)
    if os.path.exists(log_file):
        for line in open(log_file, 'r'):
            if re.search("Overall per-batch", line):
                real_e2e_time = float(line.split(' ')[4]) * 1000 # In us
    assert real_e2e_time != -1
    summary_file = "{}{}_summary_{}.log".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "", args.iters)
    if os.path.exists(summary_file):
        for line in open(summary_file, 'r'):
            if re.search("Total per-batch GPU time", line):
                real_gpu_active_time = float(line.split(' ')[-3]) # In us
    assert real_gpu_active_time != -1

    e2e_time, gpu_active_time = get_e2e_time(graph, overheads, module_marker, debug=args.debug)
    # Only rank 0 prints
    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
        st = "E2E time: {:.2f}, GPU time: {:.2f}".format(e2e_time, gpu_active_time)
        if real_e2e_time != -1:
            st += "\nReference time: {:.2f}, {:.2f}".format(real_e2e_time, real_gpu_active_time)
            st += "\nPrediction error: {:.2f}%, {:.2f}%, {:.2f}%".format(
                (gpu_active_time / real_gpu_active_time - 1) * 100,
                (e2e_time / real_e2e_time - 1) * 100,
                (gpu_active_time / real_e2e_time - 1) * 100,
            )
        print(st)
        prediction_name = "{}{}_prediction_{}.log".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "", args.iters)
        with open(prediction_name, 'w') as f:
            f.write(st)
