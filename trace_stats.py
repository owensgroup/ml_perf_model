import argparse, json, os, re
import numpy as np
from analysis.trace_utils import *
from analysis.utils import PM_HOME, GPU_NAME, KERNEL_LAUNCH_LENGTH, CPU_EVENT_OVERHEAD, GPU_EVENT_OVERHEAD

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Process trace files and get stats.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    print("======= {}, {} GPU(s), batch size: {} =======".format(args.model_name, args.num_gpus, args.batch_size))
    prefix = "{}/data/{}/e2e/{}/{}_{}".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size)
    module_marker = "DLRM " if "DLRM" in args.model_name else "## Forward ##"

    trace_file = "{}.json".format(prefix)
    trimmed_trace_file="{}_trimmed.json".format(prefix)
    if not os.path.exists(trace_file):
        print("Trace file doesn't exist! Please run the benchmark first.")
        exit(1)
    if not os.path.exists(trimmed_trace_file):
        trimmed_trace_file = trim_trace_by_num_iter(trace_file, iters=args.iters, trimmed_file=trimmed_trace_file)
    with open(trimmed_trace_file) as f:
        trace = json.load(f)

    roots, cc, corrected_start_time, corrected_end_time, sum_skipped_intervals = process_event_hierarchy(trace['traceEvents'], skip_module=False, module_marker=module_marker)
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

    op_device_runtime = get_device_runtime(ops, cc)
    dt_breakdown = device_runtime_breakdown(roots, op_device_runtime, depth=0)
    flatten = {}
    print("Totally {} GPU streams.".format(len(dt_breakdown.items())))
    gpu_time = 0
    for stream, v in dt_breakdown.items():
        flatten[stream] = {}
        get_major_device_results(device_runtime, dt_breakdown[stream], flatten[stream])
        print("  Stream {}: average per-batch time: {:.2f} us".format(stream, flatten[stream]["total"]["runtime"] / args.iters))
        runtime_no_pf = -1
        log_file = "{}/data/{}/e2e/{}/1_{}.log".format(PM_HOME, GPU_NAME, args.model_name, args.batch_size)
        if os.path.exists(log_file):
            for line in open(log_file, 'r'):
                if re.search("Overall per-batch", line):
                    runtime_no_pf = float(line.split(' ')[4]) * 1000 * args.iters # us

            per_op = {}
            total = 0.0
            for k, vv in flatten[stream].items():
                if k == 'total' or 'DLRM ' in k[0] or 'module' in k[0]: # Skip all labels
                    continue
                k0 = k[0] if '#' not in k[0] else k[0].split('#')[0]
                if k0 not in per_op.keys():
                    per_op[k0] = 0.0
                per_op[k0] += vv['runtime']
                total += vv['runtime']

            tmp = sorted(per_op.items(), key=lambda x: x[1], reverse=True)
            op = [x[0] for x in tmp]
            p = [x[1] / total for x in tmp]

            active_time_perc = flatten[stream]['total']['runtime'] / runtime_no_pf
            idle_time_perc = 1 - flatten[stream]['total']['runtime'] / runtime_no_pf
            print("  Stream {}: active time perc {:.2f}%, idle time perc {:.2f}%".format(
                stream, active_time_perc * 100, idle_time_perc * 100))
        
        gpu_time += flatten[stream]["total"]["runtime"] # TODO: Deal with stream overlapping.
    print("Total per-batch GPU time: {:.2f} us".format(gpu_time / args.iters))

    # Type 1 overhead: between two op calls
    # Type 2 overhead: before the first device call, op-specific
    # Type 3 overhead: after the last device call, op-specific
    # Type 4 overhead: kernel launches themselves, kernel-launch-type-specific
    # Type 5 overhead: sum of gaps between kernel launches, op-specific
    overheads = {'independent': {}}
    overheads['independent']['t1'] = [] # Independent from names
    overheads['independent']['t4'] = {} # Independent from names
    launches_dict = {}

    for i, op in enumerate(ops):
        name = op.name()
        if name not in overheads.keys():
            overheads[name] = {}

        if 't2' not in overheads[name].keys():
            overheads[name]['t2'] = []
        if 't3' not in overheads[name].keys():
            overheads[name]['t3'] = []
        if 't5' not in overheads[name].keys():
            overheads[name]['t5'] = []

        sub_event_count = get_sub_event_count(op)
        # Get the number of events before each kernel launch (to subtract corresponding amount of CPU overheads from estimated time)
        tmp_launches = get_event_all_kernel_launches(op)
        launches = []
        count = 0
        for x, y in tmp_launches:
            count += y
            if x.name() in ["cudaLaunchKernel", "cudaMemcpyAsync", "cudaStreamSynchronize"]:
                launches.append((x, count))
                count = 0

        if len(launches) > 0:
            overheads[name]['t2'].append(launches[0][0].start_time() - op.start_time() - launches[0][1] * CPU_EVENT_OVERHEAD) # T2 has all overheads before the first launch
            trailing_sub_event_count = sub_event_count - sum([y+1 for _, y in launches]) # And kernel launches themselves
            overheads[name]['t3'].append(max(op.end_time() - launches[-1][0].end_time() - trailing_sub_event_count * CPU_EVENT_OVERHEAD, 0)) # T3 has all overheads after the last launch
            t3 = op.end_time() - launches[-1][0].end_time() - trailing_sub_event_count * CPU_EVENT_OVERHEAD
            if len(launches) > 1:
                overheads[name]['t5'].extend([launches[i][0].start_time() - launches[i-1][0].end_time() - launches[i][1] * CPU_EVENT_OVERHEAD for i in range(1, len(launches))]) # T5 has all overheads between each pair of launches
            else:
                overheads[name]['t5'].append(0)

            # T4 is launch-type-dependent
            for x, _ in launches:
                # Skip T4 for synchronization
                if x.name() in ["cudaStreamSynchronize"]:
                    continue
                if x.name() not in overheads['independent']['t4']:
                    overheads['independent']['t4'][x.name()] = []
                overheads['independent']['t4'][x.name()].append(KERNEL_LAUNCH_LENGTH - CPU_EVENT_OVERHEAD - GPU_EVENT_OVERHEAD) # T4 has 1 overhead
            
            if op.name() not in launches_dict.keys():
                launches_dict[op.name()] = []
                for x, _ in launches:
                    # Don't record synchronization
                    if x.name() not in ["cudaStreamSynchronize"]:
                        launches_dict[op.name()].append(x.name())
        else:
            if name not in overheads.keys():
                overheads[name] = {}
            # If an op doesn't have kernel calls it has only one T5 overhead representing its CPU duration
            if 't5' not in overheads[name].keys():
                overheads[name]['t5'] = []
            if name == "aten::to":
                continue # Some aten::to doesn't have children
            else:
                overheads[name]['t5'].append(op.duration() - sub_event_count * CPU_EVENT_OVERHEAD) # Remove cpu overhead for all sub events

        if i > 0:
            prev_op = ops[i-1]
            
            # Only consider adjacent ops under the SAME MODULE
            if prev_op.parent != op.parent:
                continue
                
            gap = op.start_time() - prev_op.end_time()
            if gap < 200: # Skip dataloading gaps
                overheads['independent']['t1'].append(gap - CPU_EVENT_OVERHEAD) # Some pairs of ops are actually inserted by a runtime call which has been filtered from ops. TODO: fix it.

    # # T1: mean ~= 21, std ~= 20
    # from analysis.utils import histogram
    # histogram(overheads['independent']['t1'], perc=False, bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 100000])
    # print(np.mean(overheads['independent']['t1']), np.std(overheads['independent']['t1']))

    # T2, T3, T5
    t2 = {k: (np.mean(v['t2']), np.std(v['t2']), len(v['t2'])) for k, v in overheads.items() if k != 'independent' and len(v['t2']) > 0}
    # pprint(t2)
    t3 = {k: (np.mean(v['t3']), np.std(v['t3']), len(v['t3'])) for k, v in overheads.items() if k != 'independent' and len(v['t3']) > 0}
    # pprint(t3)
    t5 = {k: (np.mean(v['t5']), np.std(v['t5']), len(v['t5'])) for k, v in overheads.items() if k != 'independent' and len(v['t5']) > 0}
    # pprint(t5)

    # # T4
    # for t, l in overheads['independent']['t4'].items():
        # print(t, np.mean(l), np.std(l), len(l))

    o = {
        "t1": (np.mean(overheads['independent']['t1']), np.std(overheads['independent']['t1']), len(overheads['independent']['t1'])),
        "t2": t2,
        "t3": t3,
        "t4": {
            t: (np.mean(l), np.std(l), len(l)) for t, l in overheads['independent']['t4'].items()
        },
        "t5": t5,
        "launches": launches_dict
    }

    overhead_name = "{}/data/{}/e2e/{}/{}_{}_overheads.json".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size)
    print("Export overheads to JSON...")
    with open(overhead_name, "w") as f:
        json.dump(o, f)
