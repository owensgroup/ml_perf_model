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
    parser = argparse.ArgumentParser("Process trace files and get stats.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--not-fbgemm", action="store_true", default=False)
    args = parser.parse_args()

    if args.num_gpus > 1:
        ext_dist.init_distributed(use_gpu=False) # Don't need GPU for E2E
    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
        print("======= {}, {} GPU(s), batch size: {}, iters: {} =======".format(
                args.model_name, args.num_gpus, args.batch_size, args.iters))

    prefix = "{}/data/{}/e2e/{}/{}/{}_{}{}".format(PM_HOME, GPU_NAME, args.model_name, 'b' if args.not_fbgemm else 'f', args.num_gpus, args.batch_size, "_distributed" if args.num_gpus > 1 else "")
    trace_file = "{}{}.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "")
    if not os.path.exists(trace_file):
        print("Trace file doesn't exist! Please run the benchmark first.")
        exit(1)
    trimmed_trace_file="{}{}_trimmed_{}.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "", args.iters)
    if not os.path.exists(trimmed_trace_file):
        trimmed_trace_file = trim_trace_by_num_iter(trace_file, iters=args.iters, trimmed_file=trimmed_trace_file)
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
    overhead_stats_name = "{}{}_overhead_stats_{}.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "", args.iters)
    overhead_raw_name = "{}{}_overhead_raw_{}.csv".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "", args.iters)
    print("Rank {}: export overheads to JSON...".format(ext_dist.my_local_rank if ext_dist.my_size > 1 else 0))
    with open(overhead_stats_name, "w") as f:
        json.dump(overhead_stats, f)
    save_raw_overhead_data(overhead_raw, overhead_raw_name, args.model_name, args.batch_size)
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
    runtime_no_pf = -1
    log_file = "{}.log".format(prefix)
    if os.path.exists(log_file):
        for line in open(log_file, 'r'):
            if re.search("Overall per-batch", line):
                runtime_no_pf = float(line.split(' ')[4]) * 1000 * args.iters # us
    assert runtime_no_pf != -1

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
        active_time_perc = min(gpu_time[s], runtime_no_pf) / runtime_no_pf
        idle_time_perc = 1 - active_time_perc
        st += "\n    Stream {}: average per-batch time: {:.2f} us, active perc {:.2f}%, idle perc {:.2f}%".format(
            s,
            min(gpu_time[s], runtime_no_pf) / args.iters,
            active_time_perc * 100,
            idle_time_perc * 100)
    st += "\nTotal per-batch GPU time and percentage: {:.2f} us ({:.2f}%)".format(
        min(gpu_time['total'], runtime_no_pf) / args.iters,
        min(gpu_time['total'], runtime_no_pf) / runtime_no_pf * 100
    )
    summary_file = "{}{}_summary_{}.log".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "", args.iters)
    with open(summary_file, 'w') as f:
        f.write(st)
