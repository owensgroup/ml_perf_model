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
import analysis.extend_distributed as ext_dist
from analysis.utils import PM_HOME, GPU_NAME
from analysis.inference import get_e2e_time
from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict end-to-end training time of DLRM models.", add_help=False)
    parser.add_argument("-i", "--iters", type=int, default=30)
    parser.add_argument("-b", "--batch-size", type=int, default=2048)
    parser.add_argument("-m", "--model-name", type=str, required=True)
    parser.add_argument("-g", "--num-gpus", type=int, default=1)
    parser.add_argument("-t", "--is-batched-emb", action="store_true", default=False)
    parser.add_argument("-s", "--bucket-size-mb", type=int, default=25)
    parser.add_argument("-r", "--early-barrier", action="store_true", default=False)
    parser.add_argument("-a", "--aggregated-allreduce", action="store_true", default=False)
    parser.add_argument("-o", "--use-shared-overheads", action="store_true", default=False)
    parser.add_argument("-d", "--table-indices", type=str, default="4-24-26-156-340-404")
    parser.add_argument("-h", "--sharder", type=str, default="naive")
    parser.add_argument("-x", "--year", type=str, default="2021")
    parser.add_argument("-u", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.num_gpus > 1:
        ext_dist.init_distributed(use_gpu=False) # Don't need GPU for E2E prediction.
    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
        tmp_str = ""
        if "DLRM" in args.model_name:
            tmp_str = "{}{}{}{}{}{}".format(
                ", batched_emb" if args.is_batched_emb else ", FBGEMM",
                ", {} sharder".format(args.sharder),
                ", bucket size: {}".format(args.bucket_size_mb) if args.bucket_size_mb != 25 else "",
                ", early barrier" if args.early_barrier else "",
                ", aggregated allreduce" if args.aggregated_allreduce else ", bucketed allreduce",
                " ({}: {})".format(args.year, args.table_indices) if args.model_name == "DLRM_open_source" else "",
            )
        print("======= [Training time prediction] {}, {} GPU(s), batch size: {}, iters: {}{} =======".format(
            args.model_name, args.num_gpus, args.batch_size, args.iters, tmp_str))

    dlrm_folder_str = ""
    if "DLRM" in args.model_name:
        dlrm_folder_str += "b/" if args.is_batched_emb else "f/"
        if args.num_gpus > 1:
            dlrm_folder_str += "{}/{}_{}/{}/".format(
                args.sharder,
                "barrier" if args.early_barrier else "no_barrier",
                "aggregated_allreduce" if args.aggregated_allreduce else "bucketed_allreduce",
                args.bucket_size_mb,
            )
    prefix = "{}/data/{}/e2e/{}{}{}/{}{}_{}{}".format(
        PM_HOME,
        GPU_NAME,
        args.model_name,
        ("/" + args.year) if args.model_name == "DLRM_open_source" else "",
        ("/" + args.table_indices) if args.model_name == "DLRM_open_source" else "",
        dlrm_folder_str,
        args.num_gpus,
        args.batch_size,
        "_distributed" if args.num_gpus > 1 else ""
    )
    per_device_prefix = "{}{}".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "")
    exec_graph_file = "{}_graph.json".format(per_device_prefix)
    with open(exec_graph_file) as f:
        graph = ExecutionGraph(json.load(f))
    if args.use_shared_overheads:
        overheads_file = "{}/data/{}/e2e/shared_overheads.json".format(PM_HOME, GPU_NAME)
    else:
        overheads_file = "{}_overhead_stats_{}.json".format(prefix, args.iters)
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
    summary_file = "{}_summary_{}.log".format(per_device_prefix, args.iters)
    if os.path.exists(summary_file):
        for line in open(summary_file, 'r'):
            if re.search("Total per-batch GPU time", line):
                real_gpu_active_time = float(line.split(' ')[-3]) # In us
    assert real_gpu_active_time != -1

    Ls_file = "{}_Ls.txt".format(per_device_prefix) if "DLRM" in args.model_name else None
    embedding_rfs_file = "{}_rfs.txt".format(per_device_prefix) if "DLRM" in args.model_name else None
    e2e_time, gpu_active_time = get_e2e_time(
        graph, overheads, iters=args.iters,
        ls_file=Ls_file,
        embedding_rfs_file=embedding_rfs_file,
        debug=args.debug)
    # Only rank 0 prints
    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
        st = "E2E time: {:.2f}, GPU time: {:.2f}".format(e2e_time, gpu_active_time)
        if real_e2e_time != -1:
            st += "\nReference time: {:.2f}, {:.2f}".format(real_e2e_time, real_gpu_active_time)
            st += "\nPrediction error: {:.2f}%, {:.2f}%, {:.2f}%\n".format(
                (gpu_active_time / real_gpu_active_time - 1) * 100,
                (e2e_time / real_e2e_time - 1) * 100,
                (gpu_active_time / real_e2e_time - 1) * 100,
            )
        print(st)
        prediction_name = "{}_prediction_{}{}.log".format(
            prefix, args.iters, '_shared' if args.use_shared_overheads else '')
        with open(prediction_name, 'w') as f:
            f.write(st)
