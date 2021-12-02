from __future__ import absolute_import, division, print_function, unicode_literals
import json, sys
from itertools import compress

from analysis.utils import SKIP

# Label markers
LABEL_MARKERS = ["##", "__", "module::", "DLRM "]

# Launches to be skip
SKIP_RUNTIME_EVENTS = ["cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", \
                    "cudaPeakAtLastError", "cudaStreamGetCaptureInfo", \
                    "cudaEventQuery", "cudaEventRecord"]


def is_module(op):
    return any([op.name().startswith(x) for x in LABEL_MARKERS])


# From Louis. Trim a long trace so that it eases the ATC processing
def trim_trace_by_percentage(file_name, start, end, trimmed_file=None):
    assert (0 <= start and start <= 1 and 0 <= end and end <= 1 and start <= end)
    if trimmed_file is None:
        trimmed_file = "./{}_trimmed.json".format(file_name.split('/')[-1].split('.json')[0])

    with open(file_name) as trace_file:
        trace = json.load(trace_file)
        min_time = sys.maxsize
        max_time = 0

        for event in trace["traceEvents"]:
            min_time = min(min_time, event['ts'])
            max_time = max(max_time, event['ts'])

        print("time range: {} {}".format(min_time, max_time))
        time_range = max_time - min_time
        offset_start = start * time_range
        offset_end = end * time_range
        # offset from the start to the trimmed end
        max_time = min_time + offset_end
        # move the min time to the offset start
        min_time += offset_start
        print("trimmed time range: {} {}".format(min_time, max_time))
        trimmed_trace = [x for x in trace["traceEvents"] if x['ts'] > min_time and x['ts'] < max_time]
        with open(trimmed_file, 'w') as out_file:
            trace = {
                'schemaVersion': trace['schemaVersion'],
                'traceEvents': trimmed_trace
            }
            json.dump(trace, out_file)

    return trimmed_file


def trim_trace_by_time(file_name, start, end, trimmed_file=None):
    if trimmed_file is None:
        trimmed_file = "./{}_trimmed.json".format(file_name.split('/')[-1].split('.json')[0])

    with open(file_name) as trace_file:
        trace = json.load(trace_file)
        
        min_time = sys.maxsize
        max_time = 0
        for event in trace["traceEvents"]:
            min_time = min(min_time, event['ts'])
            max_time = max(max_time, event['ts'])
        assert (min_time <= start and start <= max_time and min_time <= end and end <= max_time and start <= end)
        
        trimmed_trace = [x for x in trace["traceEvents"] if x['ts'] >= start and x['ts'] <= end]
        with open(trimmed_file, 'w') as out_file:
            trace = {
                'schemaVersion': trace['schemaVersion'],
                'traceEvents': trimmed_trace
            }
            json.dump(trace, out_file)

    return trimmed_file


def trim_trace_by_num_iter(file_name, iters=10, skip_iters=50, trimmed_file=None):
    if trimmed_file is None:
        trimmed_file = "./{}_trimmed.json".format(file_name.split('/')[-1].split('.json')[0])

    with open(file_name) as trace_file:
        trace = json.load(trace_file)
        start_idx, end_idx = 0, -1
        marker_count = 0
        t = trace["traceEvents"] if isinstance(trace, dict) else trace

        if 'DLRM' in file_name:
            marker = 'DataLoader'
        else:
            marker = '## Forward ##'
        for idx, x in enumerate(t):
            if marker in x['name']:
                if marker_count == skip_iters:
                    start_idx = idx
                if marker_count == skip_iters + iters:
                    end_idx = idx
                    break
                marker_count += 1
        assert end_idx != -1, "Trace too short!"
        trimmed_trace = [x for x in t if x['ts'] >= t[start_idx]["ts"] and x['ts'] < t[end_idx]["ts"]] # Don't include the last Dataloader

        with open(trimmed_file, 'w') as out_file:
            trace = {
                'schemaVersion': None,
                'traceEvents': trimmed_trace
            }
            json.dump(trace, out_file)

    return trimmed_file


def list_to_tuple(lst):
    return tuple(list_to_tuple(l) if isinstance(l, list) else l for l in lst) if lst is not None else None


class Event:
    def __init__(self, e, dummy=False):
        if dummy:
            self.event = {
                "name": "dummy",
                "ts": -1,
                "dur": -1,
                "cat": "Runtime",
                "args": {}
            }
        else:
            assert (type(e) == dict)
            self.event = e
        self.parent = None
        self.children = []
        self.has_device_calls = False
    def __str__(self):
        return json.dumps(self.event, sort_keys=True, indent=4, separators=(',', ': '))
#     def __repr__(self):
#         return json.dumps(self.event, sort_keys=True, indent=4, separators=(',', ': '))
    def start_time(self):
        if "ts" not in self.event.keys():
            return None
        return self.event["ts"]
    def duration(self):
        if "dur" not in self.event.keys():
            return None
        return self.event["dur"]
    def end_time(self):
        if "ts" not in self.event.keys() or "dur" not in self.event.keys():
            return None
        return self.event["ts"] + self.event["dur"]
    def category(self):
        if "cat" not in self.event.keys():
            raise TypeError("Unknown event type!")
        return self.event["cat"]
    def name(self):
        if "name" not in self.event.keys():
            raise TypeError("Name lost!")
        return self.event["name"]
    def is_sub_of(self, other):
        assert (self.start_time() is not None and \
                self.duration() is not None and \
                other.start_time() is not None and \
                other.duration() is not None)
        ls = other.start_time()
        le = other.start_time() + other.duration()
        es = self.start_time()
        ee = self.start_time() + self.duration()
        return ls <= es and le >= ee
    def input_shape(self):
        if "args" not in self.event.keys() or "Input Dims" not in self.event["args"].keys():
            return (-1,)
        return list_to_tuple(self.event["args"]["Input Dims"])
    def output_shape(self):
        if "args" not in self.event.keys() or "Output dims" not in self.event["args"].keys():
            return (-1,)
        return list_to_tuple(self.event["args"]["Output dims"])
    def external_id(self):
        if "args" not in self.event.keys():
            return None

        if ("External id" not in self.event["args"].keys() and \
             "external id" not in self.event["args"].keys()):
            raise TypeError("External id lost!")
        
        if self.category() == "Operator":
            return self.event["args"]["External id"]
        else:
            return self.event["args"]["external id"]
    def correlation_id(self):
        if "args" not in self.event.keys() or self.category() == "Operator":
            return None

        if ("correlation" not in self.event["args"].keys()):
            raise TypeError("Correlation id lost!")
        return self.event["args"]["correlation"]
    def pid(self):
        assert "pid" in self.event.keys(), "Illegal trace!"
        return self.event["pid"]
    def tid(self):
        assert "tid" in self.event.keys(), "Illegal trace!"
        return self.event["tid"]
    def device(self):
        if "args" not in self.event.keys() or \
            ("Device" not in self.event["args"].keys() and \
            "device" not in self.event["args"].keys()):
            return None
        if "Device" in self.event["args"].keys():
            return self.event["args"]["Device"]
        else:
            return self.event["args"]["device"]
    def stream(self):
        if "args" not in self.event.keys() or "stream" not in self.event["args"].keys():
            return None
        return self.event["args"]["stream"]


# Construct a forest to represent the event hierarchy as well as a data structure to hold the relation between ops and device calls
########## cc #########
# {
#     ex_id1 : {
#         caller: - (an op that has one or multiple device calls)
#         callees: {
#             cr_id1: {
#                 launcher: - (cudaKernelLaunch)
#                 executor: - (device kernel)
#             }
#             ...
#         }
#     }
#     ...
# }
def process_event_hierarchy(raw_trace, skip_module=False, module_marker="## "):
    
    # Get the "grandest child" event of a given leaf
    # e.g. |------------ A --------------| The leaf event in the frontier currently being accessed
    #         |------------B-----------|
    #            |-----C------| The current "grandest child" of A, since D hasn't been added as A's child yet
    #               |---D---| The event currently being processed
    def get_grandest_child_event(leaf, current_event, depth=1):
        if not current_event.is_sub_of(leaf):
            return None
        ret = leaf
        for c in leaf.children:
            grandest = get_grandest_child_event(c, current_event, depth+1)
            if grandest is not None and current_event.tid() == grandest.tid(): # The root leaf e.g. the module won't be the grandest here
                ret = grandest
                break
        return ret

    roots = [] # All the root events that have no parents
    leaves = [] # The event frontier of the processing
    unaccounted = [] # Unaccounted events (not being used now)
    cc = {} # caller / callee: key = external id, value = { caller event, callee events }
    main_tid = -1 # ID of the main thread that executes data loading, module events, etc

    # Remove all events without a duration and sort the event lists by start time (increasing order) and duration (decreasing order)
    sorted_events = [Event(e) for e in raw_trace if "dur" in e.keys()]
    sorted_events = sorted(sorted_events, key=lambda x: (x.start_time(), -x.duration()))
    
    # Remove all leftovers from the last iteration and next iteration
    start_idx = 0
    end_idx = len(sorted_events) - 1
    corrected_start_time = sorted_events[0].start_time()
    corrected_end_time = sorted_events[-1].start_time()

    # Start the analysis from the first module detected, if module is not to be skipped
    for idx, x in enumerate(sorted_events):
        ######## IMPORTANT ########
        # Find the start of an iteration started with "##" without ":". The first module should be "## zero_grad ##" though, 
        # but the current ATC code couldn't start the extraction exactly at there. 
        # Change TORCH_AUTOGRAD_GRAPHROOT in ATC's trace_utils.py does the trick
        if not skip_module and x.name().startswith(module_marker) and ":" not in x.name():
            # The actual start time is the start time of the profiler enter call right before "zero_grad"
            for idy, y in enumerate(reversed(sorted_events[:idx])):
                if y.name() == "profiler::_record_function_enter":
                    start_idx = idx - idy
                    corrected_start_time = y.start_time()
                    break
            break

    # End the analysis at the last event that has a duration. Set the corrected end time later.
    for idx, x in enumerate(reversed(sorted_events)):
        if x.duration() is not None:
            end_idx = idx
            break
    sorted_events = sorted_events[start_idx:(len(sorted_events) - 1 - end_idx)]
    skipped_intervals = []
    skipped_intervals_loader = []
    skipped_intervals_distribute = []

    # Skip data-loading ops and DLRM distribute emb data
    for x in sorted_events:
        event_start = x.start_time()
        event_duration = x.duration()
        external_id = x.external_id()
        correlation_id = x.correlation_id()
        if 'DataLoader' in x.name() and event_duration > 100:
            skipped_intervals_loader.append((event_start, event_start+event_duration))
        if 'distribute emb data' in x.name():
            skipped_intervals_distribute.append((event_start, event_start+event_duration))
        # Find the thread ID of the main thread btw
        if main_tid == -1 and x.name().startswith(module_marker):
            main_tid = x.tid()
    assert main_tid != -1
    for x, y in zip(skipped_intervals_loader, skipped_intervals_distribute):
        skipped_intervals.append((x[0], y[1]))

    for x in sorted_events:
        # Get start, duration and end time of the current event
        event_start = x.start_time()
        event_duration = x.duration()
        external_id = x.external_id()
        correlation_id = x.correlation_id()

        # Skip all events in skipped intervals
        should_skip = False
        for s, e in skipped_intervals:
            if event_start >= s and event_start <= e:
                should_skip = True
                break
        if should_skip:
            continue

        # Runtime events e.g. cudaLaunchKernel counted as host events
        if x.category() == "Operator" or x.category() == "Runtime":
            if event_start is None or event_duration is None:
                print("Unaccounted event: {}".format(x.event))
                unaccounted.append(x)
                continue
            # Put all OPERATOR events with no device info into unaccounted (0 means None in the trace file)
            # This usually work for events like aten::pin_memory, etc
            # if x.device() == 0:
            #     unaccounted.append(x)
            #     continue

            event_end = event_start + event_duration
            corrected_end_time = max(event_end, corrected_end_time)
            # Find parent of the current event from the frontier
            parent_found = False
            add_to_root = None
            add_to_leaf = None
            active_leaves = [] # Boolean markers for leaves that are not outdated yet
            for idx, l in enumerate(leaves):
                if parent_found:
                    active_leaves.extend([True] * (len(leaves) - len(active_leaves))) # Fill up the filter list
                    break
                leaf_start = l.start_time()
                leaf_end = leaf_start + l.duration()

                # The current event has no overlap with leaf
                if event_start > leaf_end:
                    active_leaves.append(False) # Mark this leaf as outdated
                    continue
                # The current event is sub to leaf
                if event_end <= leaf_end:
                    # Only search of the children when the current leaf is on the main thread or the same thread of the current event
                    if l.tid() == main_tid or l.tid() == x.tid():
                        # Add this event to the GRANDEST CHILD of the leaf that can sub it
                        grandest = get_grandest_child_event(l, x)
                        x.parent = grandest
                        grandest.children.append(x)
                    # Add to leaf anyway
                    add_to_leaf = x
                    parent_found = True
                    active_leaves.append(True) # Mark this leaf as active
                # Crossover shouldn't happen
                else:
                    raise ValueError("\tCrossover happens to {}!".format(str(x)))
            # Delete all outdated leaves
            leaves = list(compress(leaves, active_leaves))

            # New root and leaf
            if not parent_found:
                add_to_root = x
                add_to_leaf = x
            if add_to_root:
                roots.append(add_to_root)
            if add_to_leaf:
                leaves.append(add_to_leaf)
            
            # Add op to caller or unaccounted
            if x.category() == "Operator":
                if external_id != 0:
                    if external_id not in cc.keys():
                        cc[external_id] = {}  
                    cc[external_id]["caller"] = x
                    cc[external_id]["callees"] = {}
            else: # Runtime
                if external_id != 0 and correlation_id != 0: # Not consider some events without ex_id and cr_id, e.g. cudaEventCreateWithFlags
                    if external_id not in cc.keys():
                        cc[external_id] = {}
                    if "caller" not in cc[external_id].keys():
                        cc[external_id]["caller"] = None
                    if "callees" not in cc[external_id].keys():
                        cc[external_id]["callees"] = {}
                    if correlation_id not in cc[external_id]["callees"].keys():
                        cc[external_id]["callees"][correlation_id] = {}
                        cc[external_id]["callees"][correlation_id]["launcher"] = None
                        cc[external_id]["callees"][correlation_id]["executor"] = None
                    cc[external_id]["callees"][correlation_id]["launcher"] = x
        else:
            # Skip modules if needed
            if (skip_module and x.name().startswith(module_marker)):
                continue
            else: # "cat" = "Memcpy" or "Kernel", i.e. callee
                if external_id != 0 and correlation_id != 0: # Doesn't consider some events without ex_id and cr_id, e.g. cudaEventCreateWithFlags
                    if external_id not in cc.keys():
                        cc[external_id] = {}
                    if "caller" not in cc[external_id].keys():
                        cc[external_id]["caller"] = None
                    if "callees" not in cc[external_id].keys():
                        cc[external_id]["callees"] = {}
                    if correlation_id not in cc[external_id]["callees"].keys():
                        cc[external_id]["callees"][correlation_id] = {}
                        cc[external_id]["callees"][correlation_id]["launcher"] = None
                        cc[external_id]["callees"][correlation_id]["executor"] = None
                    cc[external_id]["callees"][correlation_id]["executor"] = x

    # Update 'has_device_calls' for all events in the tree
    def update_has_device_calls(roots):
        for r in roots:
            ex_id = r.external_id()
            if len(r.children) == 0:
                if ex_id in cc.keys() and len(cc[ex_id]["callees"].keys()) != 0:
                    for k, v in cc[ex_id]["callees"].items():
                        if v["executor"] is not None:
                            r.has_device_calls = True
            else:
                update_has_device_calls(r.children)
                for c in r.children:
                    if c.has_device_calls:
                        r.has_device_calls = True
    update_has_device_calls(roots)
    
    sum_skipped_intervals = sum([e-s for s, e in skipped_intervals])

    return roots, cc, corrected_start_time, corrected_end_time, sum_skipped_intervals


def get_event_all_kernel_launches(event):
    count = 0 # Count of leading events of a launch
    def get_launches(event, lst):
        nonlocal count
        if len(event.children) == 0:
            if event.category() == 'Runtime' and event.name() not in SKIP_RUNTIME_EVENTS:
                lst.append((event, count-1))
                count = 0
            return
        for r in event.children:
            count += 1
            get_launches(r, lst)

    lst = []
    get_launches(event, lst)
    return lst


def get_sub_event_count(event):
    count = 0 # Count of sub events
    def get_count(event):
        nonlocal count
        count += 1
        for r in event.children:
            # Skip a few runtime events
            if r.name() not in SKIP_RUNTIME_EVENTS:
                get_count(r)

    get_count(event)
    return count


def get_device_runtime_and_start_delay(cc, corrected_start_time):
    device_runtime = 0
    device_start_delay = 1000000000000000

    for ex_id, v in cc.items():
        for cr_id, vv in v["callees"].items():
            if vv["executor"] is not None:
                device_start_delay = min(device_start_delay, vv["executor"].start_time() - corrected_start_time)

                if "batched_embedding_forward_kernel" in vv["executor"].name():
                    if device_runtime == 0:
                        device_runtime = vv["executor"].start_time()
                    else:
                        device_runtime = vv["executor"].start_time() - device_runtime
                        device_runtime *= 2
                        if device_runtime < 0:
                            device_runtime = 0 - device_runtime
    
    return device_runtime, device_start_delay


def get_host_runtime_breakdown(events, cc, total_time):
    runtime_breakdown = {}

    def construct_rb(events, rb):
        for e in events:
            name = e.name()
            shape = (e.input_shape(), e.output_shape())
            ex_id = e.external_id()
            if name not in rb.keys():
                rb[name] = {}
                rb[name]["subs"] = {} # All the children
                rb[name]["runtime"] = 0.0 # Event runtime seen on the host
                rb[name]["stats"] = {}
            rb[name]["runtime"] += e.duration()
            
            if shape not in rb[name]["stats"].keys():
                rb[name]["stats"][shape] = {}
                rb[name]["stats"][shape]["count"] = 0
            rb[name]["stats"][shape]["count"] += 1

            # DFS for children
            construct_rb(e.children, rb[name]["subs"])

    construct_rb(events, runtime_breakdown)
    runtime_breakdown = {
        "runtime": total_time,
        "subs": runtime_breakdown
    }

    return runtime_breakdown


def print_host_results(rb, depth_limit=sys.maxsize, truncate_count=100, depth=0):
    t = rb["runtime"]
    if depth == 0:
        print(f"Two iteration runtime: {t:>20} (in us, same below)")

    module_perc_sum = 0
    dominated_perc_sum = 0
    dominated_count = 0
    space_padding = " " * (depth + 1) * 5
    for k, v in sorted(rb["subs"].items(), key=lambda x: x[1]["runtime"], reverse=True):
        vt = str(v["runtime"])
        perc = v["runtime"] / t
        module_perc_sum += perc
        # Truncate results for brevity
        if dominated_count < truncate_count:
            dominated_perc_sum += perc
            count = 0
            for kk, vv in v["stats"].items():
                count += vv["count"]
            print(f"{space_padding}{(k+':'):<40} {('(' + vt):>{(depth+2) * 5}}, {(perc * 100):.2f}%, {count})")

            # DFS and print
            if depth < depth_limit and len(v["subs"].keys()) != 0:
                print_host_results(v, depth_limit, truncate_count, depth=depth+1)
            dominated_count += 1

    # If there's still remaining time, print it under "Others"
    if abs(module_perc_sum - dominated_perc_sum) > 1e-4:
        other_time = "{:.1f}".format((module_perc_sum - dominated_perc_sum) * t)
        print(f"{space_padding}{'Others:':<40} {('(' + other_time):>{(depth+2) * 5}}, {((module_perc_sum - dominated_perc_sum) * 100):.2f}%)")

    # Unaccounted time
    unaccounted_time = "{:.1f}".format((1 - module_perc_sum) * t)
    print(f"{space_padding}{'Unaccounted:':<40} {('(' + unaccounted_time):>{(depth+2) * 5}}, {((1 - module_perc_sum) * 100):.2f}%)")


# Get and flatten root operators, not including modules
def get_operators(roots, ops):
    for r in roots:
        # Is an operator, and
        # Not a module or submodule, and
        # (Parent is a module, or, is simply a root operator)
        if r.category() == "Operator" and\
            (not is_module(r)) and ((\
            r.parent is not None
        ) or (\
            r.parent is None\
        )) :
            ops.append(r)
        else:
            get_operators(r.children, ops)


# Shorten name for some heavily templated kernels
def shorten_kernel_name(name):
    if '<' in name:
        name = name.split('<')[0]
    if '::' in name:
        name = name.split('::')[-1]
    return name


#######################
# {
#    ex_id1: {
#         stream1 : {
#             (executor_name1, shape1): {
#                 count: - (cudaKernelLaunch)
#                 runtime: - (device kernel)
#             }
#             ...
#         }
#         ...
#    }
#    ...
# }
def get_device_runtime(roots, cc, depth=0):
    result = {}
    for r in roots:
        if not r.has_device_calls:
            continue
        external_id = r.external_id()
        tmp_result = {}
        if len(r.children) == 0: # No children: either cudaLaunchKernel or host ops that have no device calls
            tmp_stats = {}
            correlation_id = r.correlation_id()
            v = cc[external_id]["callees"][correlation_id]
            lc, ex = v["launcher"], v["executor"]
            if ex is None:
                # Dummy: no executor runtime
                pass
            elif lc is not None:
                lc_dur, ex_dur = lc.duration(), ex.duration()

                ##############################################################
                # TODO: Maybe there's a better way to model the device time, e.g. involving overheads of some operators
                device_total = ex_dur
                ##############################################################

                # Important stats: type (name): { shapes: count & time }
                # Shorten executor name too long
                executor_name = shorten_kernel_name(ex.name())
                cl = cc[external_id]["caller"]
                if cl is None:
                    shape = (-1,)
                else:
                    shape = cl.input_shape()
                key = (executor_name, shape)
                stream = ex.stream()
                assert (stream is not None)
                if stream not in tmp_stats.keys():
                    tmp_stats[stream] = {}
                if key not in tmp_stats[stream].keys():
                    tmp_stats[stream][key] = {}
                    tmp_stats[stream][key]["count"] = 0
                    tmp_stats[stream][key]["runtime"] = 0.0
                tmp_stats[stream][key]["count"] += 1
                tmp_stats[stream][key]["runtime"] += device_total
            tmp_result = {-1: tmp_stats}
        else: # If has children, go to the next level
            tmp_result = get_device_runtime(r.children, cc, depth=depth+1)

        if external_id not in result.keys():
            result[external_id] = {}
        dict_pointer = result[external_id]

        # Merge results from lower levels
        for _, tmp_stats in tmp_result.items():
            for stream, v in tmp_stats.items():
                if stream not in dict_pointer.keys():
                    dict_pointer[stream] = {}
                for key, vv in v.items():
                    if key not in dict_pointer[stream].keys():
                        dict_pointer[stream][key] = {}
                        dict_pointer[stream][key]["count"] = 0
                        dict_pointer[stream][key]["runtime"] = 0.0
                    dict_pointer[stream][key]["count"] += vv["count"]
                    dict_pointer[stream][key]["runtime"] += vv["runtime"]

    # Remove all entries without correlation ids
    if depth == 0:
        to_be_deleted = []
        for ex_id, v in result.items():
            delete = True
            for stream, vv in v.items():
                if len(list(vv.keys())) != 0:
                    delete = False
            if delete:
                to_be_deleted.append(ex_id)
        for ex_id in to_be_deleted:
            del result[ex_id]
    return result


def print_all_device_results(roots, odr, total_time, depth=0):
    space_padding = " " * depth * 4
    tmp_space_padding = " " * (depth + 1) * 4
    for r in roots:
        ex_id = r.external_id()
        if r.has_device_calls:
            print(f"{space_padding}{r.name()}")
            if ex_id in odr.keys():
                result = odr[ex_id]
                for _, d in result.items():
                    for key, v in d.items():
                        kernel_name = key[0]
                        shape = key[1]
                        kernel_count = v["count"]
                        kernel_time = v["runtime"]
                        print(f"{tmp_space_padding}{(kernel_name+':'):<44} {('( ' + str(shape)):>{(depth+2) * 4}}, {kernel_count}, {kernel_time} )")
            else:
                print_all_device_results(r.children, odr, -1, depth=depth+1)


########### device runtime breakdown ############
# {
#    stream1: {
#         runtime: -
#         (module_name1, ()): {
#             runtime: -
#             count: -
#             occ: [
#                 {
#                     runtime: -
#                     (op_name1, shape1): {
#                     }
#                     ...
#                 }
#                 ...
#             ]
#         }
#         ...
#    }
#    ...
# }
def device_runtime_breakdown(ops, odr, depth=0):
    result = {}
    for r in ops:
        ex_id = r.external_id()
        if r.has_device_calls:
            key = (r.name(), r.input_shape())
            if ex_id in odr.keys(): # The current event is a caller (STILL AN OPERATOR!)
                tmp_stats = {}
                for stream, v in odr[ex_id].items():
                    if stream not in tmp_stats.keys():
                        tmp_stats[stream] = {}
                        tmp_stats[stream]["runtime"] = 0.0
                    for tmp_key, vv in v.items():
                        kernel_count = vv["count"]
                        kernel_time = vv["runtime"]
                        if tmp_key not in tmp_stats[stream].keys():
                            tmp_stats[stream][tmp_key] = {}
                            tmp_stats[stream][tmp_key]["count"] = 0
                            tmp_stats[stream][tmp_key]["runtime"] = 0.0
                        tmp_stats[stream][tmp_key]["count"] += kernel_count
                        tmp_stats[stream][tmp_key]["runtime"] += kernel_time
                        tmp_stats[stream]["runtime"] += kernel_time
            else:
                tmp_stats = device_runtime_breakdown(r.children, odr, depth=depth+1)

            for stream, v in tmp_stats.items():
                if stream not in result.keys():
                    result[stream] = {}
                    result[stream]["runtime"] = 0.0
                if key not in result[stream].keys():
                    result[stream][key] = {}
                    result[stream][key]["count"] = 0
                    result[stream][key]["runtime"] = 0.0
                    result[stream][key]["occ"] = []
                result[stream]["runtime"] += v["runtime"]
                result[stream][key]["count"] += 1
                result[stream][key]["runtime"] += v["runtime"]
                result[stream][key]["occ"].append(v)
    return result


def get_major_device_results(device_runtime, drb, flatten, parent_name="total"):
    t = drb["runtime"]
    if t == 0.0:
        return
    idle = device_runtime - t
        
    module_perc_sum = 0
    dominated_perc_sum = 0
    dominated_count = 0
    
    if parent_name not in flatten.keys():
        flatten[parent_name] = {}
        flatten[parent_name]["runtime"] = 0.0
        flatten[parent_name]["subs"] = {}
    flatten[parent_name]["runtime"] += t
    
    for key, v in sorted(drb.items(), key=lambda x: x[1]["runtime"] if (isinstance(x[1], dict) and "runtime" in x[1].keys()) else -1, reverse=True):
        if key == "runtime":
            continue
        perc = v["runtime"] / t
        module_perc_sum += perc
        dominated_perc_sum += perc
            
        # DFS and print
        if "occ" in v.keys(): # Only ops and modules have 'occ', saving all occurences of this op or module with roughly the same runtime
            get_major_device_results(device_runtime, v["occ"][0], flatten, parent_name=key)

        if parent_name not in flatten:
            flatten[parent_name] = {}
            flatten[parent_name]["subs"] = {}
            flatten[parent_name][key] = {}
        if key not in flatten[parent_name]["subs"].keys():
            flatten[parent_name]["subs"][key] = 0.0
        flatten[parent_name]["subs"][key] += v["runtime"]

    # If there's still remaining time, print it under "Others"
    if abs(module_perc_sum - dominated_perc_sum) > 1e-4:
        other_time = "{:.1f}".format((module_perc_sum - dominated_perc_sum) * t)
        if "others" not in flatten[parent_name]["subs"].keys():
            flatten[parent_name]["subs"]["others"] = 0.0
        flatten[parent_name]["subs"]["others"] += float(other_time)
        
    # Unaccounted time
    if abs(1 - module_perc_sum) > 1e-4:
        unaccounted_time = "{:.1f}".format((1 - module_perc_sum) * t)
        if "unaccounted" not in flatten[parent_name]["subs"].keys():
            flatten[parent_name]["subs"]["unaccounted"] = 0.0
        flatten[parent_name]["subs"]["unaccounted"] += float(unaccounted_time)


def print_major_device_results(device_runtime, drb, flatten, parent_name="total", truncate_count=100, depth=0):
    t = drb["runtime"]
    t_perc = t / device_runtime * 100.0
    if t == 0.0:
        return
    idle = device_runtime - t
    idle_perc = idle / device_runtime * 100.0
    
    if depth == 0:
        print(f"    Total device time: {(device_runtime)} (in us, same below)")
        print(f"    Device idle time: {idle} ({idle_perc:.2f}%)")
        print(f"    Device active time: {t} ({t_perc:.2f}%)")
        
    module_perc_sum = 0
    dominated_perc_sum = 0
    dominated_count = 0
    space_padding = " " * ((depth + 1) * 2 + 4)
    
    for key, v in sorted(drb.items(), key=lambda x: x[1]["runtime"] if (isinstance(x[1], dict) and "runtime" in x[1].keys()) else -1, reverse=True):
        if key == "runtime":
            continue
        vt = str(v["runtime"])
        perc = v["runtime"] / t
        module_perc_sum += perc
        count = v["count"]
        # Truncate results for brevity
        if dominated_count < truncate_count:
            dominated_perc_sum += perc
            name = key[0]
            shape = str(key[1])
            if shape == '()' or shape == "((),)":
                print(f"{space_padding}{(name+':'):<52} {('(' + vt):>{(depth+2) * 3}}, {(perc * 100):.2f}%, {count})")
            else:
                print(f"{space_padding}{(name+':'):<52} {('(' + vt):>{(depth+2) * 3}}, {(perc * 100):.2f}%, {count}) {(shape):>60}")
                
            # DFS and print
            if "occ" in v.keys(): # Only ops and modules have 'occ', saving all occurences of this op or module with roughly the same runtime
                print_major_device_results(device_runtime, v["occ"][0], flatten, parent_name=key, truncate_count=truncate_count, depth=depth+1)
            dominated_count += 1

    # If there's still remaining time, print it under "Others"
    if abs(module_perc_sum - dominated_perc_sum) > 1e-4:
        other_time = "{:.1f}".format((module_perc_sum - dominated_perc_sum) * t)
        print(f"{space_padding}{'Others:':<52} {('(' + other_time):>{(depth+2) * 5}}, {((module_perc_sum - dominated_perc_sum) * 100):.2f}%)")
        
    # Unaccounted time
    if abs(1 - module_perc_sum) > 1e-4:
        unaccounted_time = "{:.1f}".format((1 - module_perc_sum) * t)
        print(f"{space_padding}{'Unaccounted:':<52} {('(' + unaccounted_time):>{(depth+2) * 5}}, {((1 - module_perc_sum) * 100):.2f}%)")
