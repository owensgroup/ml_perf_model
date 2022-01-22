import argparse, json, os
import analysis.extend_distributed as ext_dist
from analysis.trace_utils import *
from analysis.utils import PM_HOME, GPU_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Process trace files and get stats.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    print("======= {}, {} GPU(s), batch size: {} =======".format(args.model_name, args.num_gpus, args.batch_size))
    if args.num_gpus > 1:
        ext_dist.init_distributed(use_gpu=False) # Don't need GPU for E2E
    prefix = "{}/data/{}/e2e/{}/{}_{}{}".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size, "_distributed" if args.num_gpus > 1 else "")
    module_marker = "DLRM " if "DLRM" in args.model_name else "## Forward ##"

    trace_file = "{}{}.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "")
    trimmed_trace_file="{}{}_trimmed.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "")
    if not os.path.exists(trace_file):
        print("Trace file doesn't exist! Please run the benchmark first.")
        exit(1)
    if not os.path.exists(trimmed_trace_file):
        trimmed_trace_file = trim_trace_by_num_iter(trace_file, iters=args.iters, trimmed_file=trimmed_trace_file)
    with open(trimmed_trace_file) as f:
        trace = json.load(f)

    # Build the event tree
    roots, cc, streams, corrected_start_time, corrected_end_time, sum_skipped_intervals = process_event_hierarchy(trace['traceEvents'], skip_module=False, module_marker=module_marker)
    host_runtime = corrected_end_time - corrected_start_time - sum_skipped_intervals
    device_runtime = host_runtime
    ops = []
    get_operators(roots, ops)
    QPS = 1000000 / host_runtime * args.iters * args.batch_size

    # Extract and save overheads
    overhead = get_overheads(ops)
    overhead_name = "{}{}_overheads.json".format(prefix, ("_" + str(ext_dist.my_local_rank)) if ext_dist.my_size > 1 else "")
    print("Rank {}: export overheads to JSON...".format(ext_dist.my_local_rank))
    with open(overhead_name, "w") as f:
        json.dump(overhead, f)

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
    print(
        f"""Number of iterations: {args.iters}
Num of events: {len(trace['traceEvents'])}, num of root events: {len(roots)}, num of caller/callee pairs: {len(cc)}
Sum of dataloading time: {sum_skipped_intervals} us
Average per-batch host runtime: {host_runtime / args.iters} us
QPS: {QPS:.2f}
Totally {len(streams)} GPU stream(s)."""
    )

    # Each stream and total GPU time
    for s in streams:
        print("    Stream {}: average per-batch time: {:.2f} us".format(s, gpu_time[s] / args.iters))
        active_time_perc = gpu_time[s] / runtime_no_pf
        idle_time_perc = 1 - gpu_time[s] / runtime_no_pf
        print("    Stream {}: active time perc {:.2f}%, idle time perc {:.2f}%".format(
            s, active_time_perc * 100, idle_time_perc * 100))
    print("Total per-batch GPU time: {:.2f} us".format(gpu_time['total'] / args.iters))
