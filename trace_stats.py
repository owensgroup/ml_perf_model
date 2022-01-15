import argparse, json, os
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
    prefix = "{}/data/{}/e2e/{}/{}_{}{}".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size, "_distributed" if args.num_gpus > 1 else "")
    module_marker = "DLRM " if "DLRM" in args.model_name else "## Forward ##"

    trace_file = "{}_0.json".format(prefix) # TODO: Replace 0 with local rank
    trimmed_trace_file="{}_0_trimmed.json".format(prefix)
    if not os.path.exists(trace_file):
        print("Trace file doesn't exist! Please run the benchmark first.")
        exit(1)
    if not os.path.exists(trimmed_trace_file):
        trimmed_trace_file = trim_trace_by_num_iter(trace_file, iters=args.iters, trimmed_file=trimmed_trace_file)
    with open(trimmed_trace_file) as f:
        trace = json.load(f)

    roots, cc, streams, corrected_start_time, corrected_end_time, sum_skipped_intervals = process_event_hierarchy(trace['traceEvents'], skip_module=False, module_marker=module_marker)
    print("Number of iterations: {}".format(args.iters))
    print('Num of events: {}, num of root events: {}, num of caller/callee pairs: {}'.format(len(trace['traceEvents']), len(roots), len(cc)))
    print('Sum of dataloading time: {} us'.format(sum_skipped_intervals))
    # print("Corrected start time: {}, corrected end time: {}".format(corrected_start_time, corrected_end_time))
    host_runtime = corrected_end_time - corrected_start_time - sum_skipped_intervals
    # ---
    # device_runtime, device_start_delay = get_device_runtime_and_start_delay(cc, corrected_start_time)
    # print("Device start delay: ", device_start_delay)
    # ---
    device_runtime = host_runtime
    # ---
    print("Average per-batch host runtime: {} us".format(host_runtime / args.iters))
    # print("Device runtime: {} us".format(device_runtime))
    ops = []
    get_operators(roots, ops)
    QPS = 1000000 / host_runtime * args.iters * args.batch_size
    print(f"QPS: {QPS:.2f}")
    print("Totally {} GPU stream(s).".format(len(streams)))

    # Get GPU stream stats
    gpu_time = get_gpu_stream_stats(cc)

    # Get overall per-batch time
    runtime_no_pf = -1
    log_file = "{}.log".format(prefix)
    if os.path.exists(log_file):
        for line in open(log_file, 'r'):
            if re.search("Overall per-batch", line):
                runtime_no_pf = float(line.split(' ')[4]) * 1000 * args.iters # us
    assert runtime_no_pf != -1

    # Each stream and total GPU time
    for s in streams:
        print("  Stream {}: average per-batch time: {:.2f} us".format(s, gpu_time[s] / args.iters))
        active_time_perc = gpu_time[s] / runtime_no_pf
        idle_time_perc = 1 - gpu_time[s] / runtime_no_pf
        print("  Stream {}: active time perc {:.2f}%, idle time perc {:.2f}%".format(
            s, active_time_perc * 100, idle_time_perc * 100))
    print("Total per-batch GPU time: {:.2f} us".format(gpu_time['total'] / args.iters))

    # Extract and save overheads
    overhead = get_overheads(ops)
    overhead_name = "{}_overheads.json".format(prefix)
    print("Export overheads to JSON...")
    with open(overhead_name, "w") as f:
        json.dump(overhead, f)
