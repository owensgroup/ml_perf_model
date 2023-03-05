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

import argparse, glob, json, os, re
import analysis.extend_distributed as ext_dist
from analysis.trace_utils import *
from analysis.utils import PM_HOME, GPU_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Process trace files and get stats.", add_help=False)
    parser.add_argument("-i", "--iters", type=int, default=30)
    parser.add_argument("-b", "--batch-size", type=int, default=2048)
    parser.add_argument("-m", "--model-name", type=str, required=True)
    parser.add_argument("-g", "--num-gpus", type=int, default=1)
    parser.add_argument("-t", "--is-batched-emb", action="store_true", default=False)
    parser.add_argument("-s", "--bucket-size-mb", type=int, default=25)
    parser.add_argument("-r", "--early-barrier", action="store_true", default=False)
    parser.add_argument("-a", "--aggregated-allreduce", action="store_true", default=False)
    parser.add_argument("-d", "--table-indices", type=str, default="4-24-26-156-340-404")
    parser.add_argument("-h", "--sharder", type=str, default="naive")
    args = parser.parse_args()

    if args.num_gpus > 1:
        ext_dist.init_distributed(use_gpu=False) # Don't need GPU for E2E
    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
        tmp_str = ""
        if "DLRM" in args.model_name:
            tmp_str = "{}{}{}{}{}{}".format(
                ", batched_emb" if args.is_batched_emb else ", FBGEMM",
                ", {} sharder".format(args.sharder),
                ", bucket size: {}".format(args.bucket_size_mb) if args.bucket_size_mb != 25 else "",
                ", early barrier" if args.early_barrier else "",
                ", aggregated allreduce" if args.aggregated_allreduce else ", bucketed allreduce",
                " ({})".format(args.table_indices) if args.model_name == "DLRM_open_source" else "",
            )
        print("======= [Trace analysis] {}, {} GPU(s), batch size: {}, iters: {}{} =======".format(
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
    prefix = "{}/data/{}/e2e/{}{}/{}{}_{}{}".format(
        PM_HOME,
        GPU_NAME,
        args.model_name,
        ("/" + args.table_indices) if args.model_name == "DLRM_open_source" else "",
        dlrm_folder_str,
        args.num_gpus,
        args.batch_size,
        "_distributed" if args.num_gpus > 1 else ""
    )
    per_device_prefix = "{}{}".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "")
    trace_file = "{}.json".format(per_device_prefix)
    if not os.path.exists(trace_file):
        print("Trace file doesn't exist! Please run the benchmark first.")
        exit(1)
    trimmed_trace_file="{}_trimmed_{}.json".format(per_device_prefix, args.iters)
    if not os.path.exists(trimmed_trace_file):
        trimmed_trace_file = trim_trace_by_num_iter(trace_file, trimmed_trace_file, iters=args.iters)
    with open(trimmed_trace_file) as f:
        trace = json.load(f)

    # Build the event tree
    roots, cc, streams, corrected_start_time, corrected_end_time, sum_skipped_intervals = process_event_hierarchy(trace['traceEvents'], skip_module=False)
    host_runtime = corrected_end_time - corrected_start_time - sum_skipped_intervals
    device_runtime = host_runtime
    ops = []
    get_operators(roots, ops)
    QPS = 1e6 / host_runtime * args.iters * args.batch_size

    # Extract and save overheads
    overhead_stats, overhead_raw = get_overheads(ops)
    overhead_stats_name = "{}_overhead_stats_{}.json".format(per_device_prefix, args.iters)
    overhead_raw_name = "{}_overhead_raw_{}.csv".format(per_device_prefix, args.iters)
    print("Rank {}: export overheads to JSON...".format(ext_dist.my_local_rank if ext_dist.my_size > 1 else 0))
    with open(overhead_stats_name, "w") as f:
        json.dump(overhead_stats, f)
    save_raw_overhead_data(overhead_raw, overhead_raw_name, args.model_name, args.batch_size)
    ext_dist.barrier() # Wait for overhead data to be saved on all processes
    # Merge raw overhead and stats from all processes
    if ext_dist.my_size > 1 and ext_dist.my_local_rank == 0:
        print("Rank {}: merge raw overhead and stats...".format(ext_dist.my_local_rank if ext_dist.my_size > 1 else 0))
        overhead_stats_files = glob.glob("{}*_overhead_stats_{}.json".format(prefix, args.iters))
        overhead_raw_files = glob.glob("{}*_overhead_raw_{}.csv".format(prefix, args.iters))

        shared_overhead, shared_df = create_shared_overhead(overhead_raw_files, overhead_stats_files, return_df=True)

        # Save shared raw overhead and stats
        with open("{}_overhead_stats_{}.json".format(prefix, args.iters), 'w') as f:
            json.dump(shared_overhead, f)
        shared_df.to_csv("{}_overhead_raw_{}.csv".format(prefix, args.iters), index=False)

    # Get overall per-batch time
    runtime_total_us = -1
    log_file = "{}.log".format(prefix)
    if os.path.exists(log_file):
        for line in open(log_file, 'r'):
            if re.search("Overall per-batch", line):
                runtime_total_us = float(line.split(' ')[4]) * 1000 * args.iters # us
    assert runtime_total_us != -1

    # Get GPU stream stats
    gpu_time = get_gpu_stream_stats(cc)

    # Print trace stats
    st = f"""Number of iterations: {args.iters}
Num of events: {len(trace['traceEvents'])}, num of root events: {len(roots)}, num of caller/callee pairs: {len(cc)}
Average per-batch dataloading time: {(sum_skipped_intervals / args.iters):.2f} us
Average per-batch host runtime: {(host_runtime / args.iters):.2f} us
QPS: {QPS:.2f}
{len(streams)} GPU stream(s) in total."""

    # Each stream and total GPU time
    for s in streams:
        active_time_perc = min(gpu_time[s], runtime_total_us) / runtime_total_us
        idle_time_perc = 1 - active_time_perc
        st += "\n    Stream {}: average per-batch time: {:.2f} us, active perc {:.2f}%, idle perc {:.2f}%".format(
            s,
            min(gpu_time[s], runtime_total_us) / args.iters,
            active_time_perc * 100,
            idle_time_perc * 100)
    st += "\nTotal per-batch GPU time and percentage: {:.2f} us ({:.2f}%)".format(
        min(gpu_time['total'], runtime_total_us) / args.iters,
        min(gpu_time['total'], runtime_total_us) / runtime_total_us * 100
    )

    # eg_imbalance and eg_comm for multi-GPU
    if ext_dist.my_size > 1:
        eg_imbalance = get_eg_imbalance(cc, runtime_total_us)
        eg_comm = get_eg_comm(cc, runtime_total_us)
        st += "\neg_imbalance: {:.2f}, eg_comm: {:.2f}".format(
            eg_imbalance,
            eg_comm
        )

    summary_file = "{}_summary_{}.log".format(per_device_prefix, args.iters)
    with open(summary_file, 'w') as f:
        f.write(st)
