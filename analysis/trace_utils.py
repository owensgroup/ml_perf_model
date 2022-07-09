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

from __future__ import absolute_import, division, print_function, unicode_literals
from itertools import compress
from collections import defaultdict
from analysis.utils import DUMMY_SHAPES, KERNEL_LAUNCH_LENGTH, CPU_EVENT_OVERHEAD, GPU_EVENT_OVERHEAD, remove_outliers
import numpy as np
import pandas as pd
import json, sys

# Label markers
LABEL_MARKERS = ["##", "__", "module::", "DLRM ", "DistributedDataParallel"]

# Launches to be skip
SKIP_RUNTIME_EVENTS = ["cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", \
                    "cudaPeakAtLastError", "cudaStreamGetCaptureInfo", \
                    "cudaEventQuery", "cudaEventRecord"]


def is_module(op):
    return any([op.name().startswith(x) for x in LABEL_MARKERS])


def is_backward(op):
    return op.name().startswith("autograd::engine::evaluate_function:") or \
            'Backward' in op.name() # Old version of Pytorch


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
        marker = 'DataLoader' # Workaround: doesn't work with ConvNets. TODO: fix this.
        marker_count = 0
        t = trace["traceEvents"] if isinstance(trace, dict) else trace
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


# Tell if the tid is normal
def abnormal_tid(e):
    try:
        int(e.tid())
    except ValueError:
        return False
    return abs(int(e.tid()) / int(e.pid())) > 2


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
    def start_time(self):
        if "ts" in self.event.keys():
            return self.event["ts"]
        return None
    def duration(self):
        if "dur" in self.event.keys():
            return self.event["dur"]
        return None
    def end_time(self):
        if "ts" not in self.event.keys() or "dur" not in self.event.keys():
            return None
        return self.event["ts"] + self.event["dur"]
    def category(self):
        if "cat" in self.event.keys():
            return self.event["cat"]
        raise TypeError("Unknown event type!")
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
    def is_annotation(self):
        return self.category() == "user_annotation"
    def is_cpu_op(self):
        # Old and new versions of trace
        return self.category() == "Operator" or self.category() == "cpu_op"
    def is_runtime(self):
        # Ditto
        return self.category() == "Runtime" or self.category() == "cuda_runtime"
    def input_shape(self):
        if "args" not in self.event.keys() or "Input dims" not in self.event["args"].keys():
            return ((-1,),)
        shape = list_to_tuple(self.event["args"]["Input dims"])
        if not shape or all([not x for x in shape]): # Empty
            return ((-1,),)
        return tuple([x for x in shape if x])
    def output_shape(self):
        if "args" not in self.event.keys() or "Output dims" not in self.event["args"].keys():
            return ((-1,),)
        shape = list_to_tuple(self.event["args"]["Output dims"])
        if not shape or all([not x for x in shape]): # Empty
            return ((-1,),)
        return tuple([x for x in shape if x])
    def external_id(self):
        if "args" not in self.event.keys() or ("External id" not in self.event["args"].keys() and "external id" not in self.event["args"].keys()):
            return -1
        if "External id" in self.event["args"].keys():
            return self.event["args"]["External id"]
        return self.event["args"]["external id"]
    def correlation_id(self):
        if "args" not in self.event.keys() or "correlation" not in self.event["args"].keys():
            return -1
        return self.event["args"]["correlation"]
    def pid(self):
        assert "pid" in self.event.keys(), "Illegal trace without pid!"
        return self.event["pid"]
    def tid(self):
        assert "tid" in self.event.keys(), "Illegal trace without tid!"
        return self.event["tid"]
    def device(self):
        if "args" not in self.event.keys():
            return None
        if ("device" in self.event["args"].keys()):
            return self.event["args"]["device"]
        raise TypeError("Device lost!")
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
#             launcher: - (cudaKernelLaunch)
#             executor: - (device kernel)
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
    streams = set() # GPU streams in this trace

    # Remove all events without a duration and sort the event lists by start time (increasing order) and duration (decreasing order)
    sorted_events = [Event(e) for e in raw_trace if "dur" in e.keys()]
    # Workaround: filter out some runtime events like cudaEventQuery and cudaEventDestroy that appear in a thread with huge tid. TODO: Fix this.
    sorted_events = [e for e in sorted_events if not (e.is_runtime() and abnormal_tid(e))]
    sorted_events = sorted(sorted_events, key=lambda x: (x.start_time(), -x.duration()))

    # Skip data-loading ops and distribution of emb data
    skipped_intervals = []
    skipped_intervals_loader = []
    skipped_intervals_forward = []
    def should_skip(x, skipped_intervals):
        event_start = x.start_time()
        for s, e in skipped_intervals:
            if event_start >= s and event_start < e:
                return True
        return False
    for x in sorted_events:
        event_start = x.start_time()
        event_duration = x.duration()
        # Remove all events between data loading and forward
        if 'DataLoader' in x.name() and event_duration > 100:
            skipped_intervals_loader.append((event_start, event_start+event_duration))
        if '## Forward ##' in x.name():
            skipped_intervals_forward.append((event_start, event_start+event_duration))
        # Find the thread ID of the main thread btw
        if main_tid == -1 and x.name().startswith(module_marker):
            main_tid = x.tid()
    assert main_tid != -1, "Main tid (data loading) not found!"
    assert len(skipped_intervals_loader) == len(skipped_intervals_forward) or \
            len(skipped_intervals_loader) == 0, \
            "DataLoader and forward pass tag counts wrong ({} vs. {})!".format(
                len(skipped_intervals_loader),
                len(skipped_intervals_forward)
            )
    if skipped_intervals_loader:
        for x, y in zip(skipped_intervals_loader, skipped_intervals_forward):
            skipped_intervals.append((x[0], y[0]))
    corrected_start_time = sorted_events[0].start_time()
    corrected_end_time = sorted_events[-1].start_time()

    # Skip all events in skipped intervals
    sorted_events = [s for s in sorted_events if not should_skip(s, skipped_intervals)]

    # Start the analysis from the first module detected, if module is not to be skipped
    start_idx = 0
    end_idx = len(sorted_events) - 1
    for idx, x in enumerate(sorted_events):
        ######## IMPORTANT ######## (Probably out-of-date. TODO: Fix this.)
        # Find the start of an iteration started with "##" without ":".
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

    # # For debugging purpose
    # with open("sorted_events.json", 'w') as f:
    #     json.dump([e.event for e in sorted_events], f)

    for x in sorted_events:
        # Get start, duration and end time of the current event
        event_start = x.start_time()
        event_duration = x.duration()
        external_id = x.external_id()
        stream = x.stream()
        if stream is not None:
            streams.add(stream)

        # Runtime events e.g. cudaLaunchKernel counted as host events
        if  x.is_cpu_op() or x.is_runtime():
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

                # The current event has no overlap with the leaf
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
                    import analysis.extend_distributed as ext_dist
                    raise ValueError("\tCrossover happens to {} and {} in {}!".format(str(x), str(l), ext_dist.my_local_rank))
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
            if x.is_runtime():
                if external_id != -1: # Not consider some events without ex_id, e.g. cudaEventCreateWithFlags
                    if external_id not in cc.keys():
                        cc[external_id] = {}
                        cc[external_id]["caller"] = x.parent # Probably None
                        cc[external_id]["callees"] = {
                            "launcher": None,
                            "executor": None,
                        }
                    cc[external_id]["callees"]["launcher"] = x
        else:
            # Skip modules if needed
            if x.is_annotation() or (skip_module and x.name().startswith(module_marker)):
                continue
            # "cat" = "Memcpy" or "Kernel", i.e. callee
            if external_id != -1: # Doesn't consider some events without ex_id, e.g. cudaEventCreateWithFlags
                cc[external_id]["callees"]["executor"] = x

    # Update 'has_device_calls' for all events in the tree
    def update_has_device_calls(roots):
        for r in roots:
            ex_id = r.external_id()
            if ex_id in cc.keys() and cc[ex_id]["callees"]["executor"]:
                r.has_device_calls = True
            else:
                update_has_device_calls(r.children)
                for c in r.children:
                    if c.has_device_calls:
                        r.has_device_calls = True
    update_has_device_calls(roots)
    
    sum_skipped_intervals = sum([e-s for s, e in skipped_intervals])

    return roots, cc, tuple(streams), corrected_start_time, corrected_end_time, sum_skipped_intervals


def get_event_all_kernel_launches(event):
    count = 0 # Count of leading events of a launch
    def get_launches(event, lst):
        nonlocal count
        if len(event.children) == 0:
            if event.is_runtime() and event.name() not in SKIP_RUNTIME_EVENTS:
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

    for _, vv in cc.items():
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


# Get and flatten root operators, not including modules
def get_operators(roots, ops):
    for r in roots:
        # Is an operator, and
        # Not a module or submodule, and
        # (Parent is a module, or, is simply a root operator)
        if (r.is_cpu_op()) and \
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
            v = cc[external_id]["callees"]
            lc, ex = v["launcher"], v["executor"]
            if ex is None:
                # Dummy: no executor runtime
                pass
            elif lc is not None:
                ex_dur = ex.duration()

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
                print("here!")
                to_be_deleted.append(ex_id)
        for ex_id in to_be_deleted:
            del result[ex_id]
    return result


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


def get_gpu_stream_stats(cc):
    # All GPU kernels in start time order
    all_kernels = sorted([
        d['callees']['executor'] \
            for _, d in cc.items() \
            if d['callees']['executor'] is not None
        ], key=lambda x: x.start_time()
    )

    gpu_time = defaultdict(int)
    idx = 0
    while 1:
        k = all_kernels[idx]
        k_start, k_end = k.start_time(), k.start_time() + k.duration()
        k_stream = k.stream()
        gpu_time[k_stream] += k.duration()

        front = k_end # The time front of a group of overlapped kernels
        idx_incr = 1 # How many kernels are already processed in the following loop
        for tmp_idx, kk in enumerate(all_kernels[(idx+1):]):
            kk_stream = kk.stream()
            if kk.start_time() >= front: # No overlaps
                break
            kk_duration = kk.duration()
            gpu_time[kk_stream] += kk_duration
            # Overlapped
            front = max(front, kk.start_time() + kk_duration)
            idx_incr = tmp_idx + 1

        gpu_time['total'] += front - k_start
        idx += idx_incr

        if idx >= len(all_kernels):
            break

    return gpu_time


def save_raw_overhead_data(overheads, file_name, model_name, batch_size):
    header = ["model_name", "batch_size", "op_name", "shapes", "type", "time"]
    df = pd.DataFrame(None, columns=header)
    # T1
    tmp = pd.DataFrame(None, columns=header)
    tmp.loc[0] = [model_name, batch_size, '', '((-1,),)', 't1', 0]
    tmp = pd.DataFrame(np.repeat(tmp.values, len(overheads['independent']['t1']), axis=0),
        columns=header)
    tmp = tmp.assign(time=overheads['independent']['t1'])
    df = pd.concat([df, tmp], ignore_index=True)
    # T4
    for k, v in overheads['independent']['t4'].items():
        tmp = pd.DataFrame(None, columns=header)
        tmp.loc[0] = [model_name, batch_size, k, '((-1,),)', 't4', 0]
        tmp = pd.DataFrame(np.repeat(tmp.values, len(v), axis=0),
            columns=header)
        tmp = tmp.assign(time=v)
        df = pd.concat([df, tmp], ignore_index=True)
    # T2, T3, T5
    for op_name, v in overheads.items():
        if op_name != 'independent':
            for shapes, vv in v.items():
                for t in ['t2', 't3', 't5']:
                    if len(vv[t]) > 0:
                        tmp = pd.DataFrame(None, columns=header)
                        tmp.loc[0] = [model_name, batch_size, op_name, shapes, t, 0]
                        tmp = pd.DataFrame(np.repeat(tmp.values, len(vv[t]), axis=0),
                            columns=header)
                        tmp = tmp.assign(time=vv[t])
                        df = pd.concat([df, tmp], ignore_index=True)
    df.to_csv(file_name, index=False)


def get_overheads(ops):
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
        input_shape = op.input_shape()
        output_shape = op.output_shape()
        shapes = str((input_shape, output_shape))
        if name not in overheads.keys():
            overheads[name] = {}
        if shapes not in overheads[name].keys():
            overheads[name][shapes] = {}
        if 't2' not in overheads[name][shapes].keys():
            overheads[name][shapes]['t2'] = []
        if 't3' not in overheads[name][shapes].keys():
            overheads[name][shapes]['t3'] = []
        if 't5' not in overheads[name][shapes].keys():
            overheads[name][shapes]['t5'] = []

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
            overheads[name][shapes]['t2'].append(launches[0][0].start_time() - op.start_time() - launches[0][1] * CPU_EVENT_OVERHEAD) # T2 has all overheads before the first launch
            trailing_sub_event_count = sub_event_count - sum([y+1 for _, y in launches]) # And kernel launches themselves
            overheads[name][shapes]['t3'].append(max(op.end_time() - launches[-1][0].end_time() - trailing_sub_event_count * CPU_EVENT_OVERHEAD, 0)) # T3 has all overheads after the last launch
            if len(launches) > 1:
                overheads[name][shapes]['t5'].extend([launches[i][0].start_time() - launches[i-1][0].end_time() - launches[i][1] * CPU_EVENT_OVERHEAD for i in range(1, len(launches))]) # T5 has all overheads between each pair of launches
            else:
                overheads[name][shapes]['t5'].append(0)

            # T4 is launch-type-dependent
            for x, _ in launches:
                # Skip T4 for synchronization
                if x.name() in ["cudaStreamSynchronize"]:
                    continue
                if x.name() not in overheads['independent']['t4']:
                    overheads['independent']['t4'][x.name()] = []
                overheads['independent']['t4'][x.name()].append(KERNEL_LAUNCH_LENGTH - CPU_EVENT_OVERHEAD - GPU_EVENT_OVERHEAD) # T4 has 1 overhead

            if name not in launches_dict.keys():
                launches_dict[name] = []
                for x, _ in launches:
                    # Don't record synchronization
                    if x.name() not in ["cudaStreamSynchronize"]:
                        launches_dict[name].append(x.name())
        else:
            if name not in overheads.keys():
                overheads[name] = {}
            if shapes not in overheads[name].keys():
                overheads[name][shapes] = {}
            # If an op doesn't have kernel calls it has only one T5 overhead representing its CPU duration
            if 't5' not in overheads[name][shapes].keys():
                overheads[name][shapes]['t5'] = []
            if name == "aten::to":
                continue # Some aten::to doesn't have children
            else:
                overheads[name][shapes]['t5'].append(op.duration() - sub_event_count * CPU_EVENT_OVERHEAD) # Remove cpu overhead for all sub events

        if i > 0:
            prev_op = ops[i-1]
            
            # Only consider adjacent ops under the SAME MODULE
            if prev_op.parent != op.parent:
                continue

            gap = op.start_time() - prev_op.end_time()
            if gap < 50 and gap > CPU_EVENT_OVERHEAD and is_backward(op): # Skip dataloading gaps and FW ops (too much manual interference)
                overheads['independent']['t1'].append(gap - CPU_EVENT_OVERHEAD) # Some pairs of ops are actually inserted by a runtime call which has been filtered from ops. TODO: fix it.

    # Remove outliers (skip t4 as we set them to constants)
    for k, v in overheads.items():
        if k != 'independent':
            for shapes, vv in v.items():
                for t in ['t2', 't3', 't5']:
                    if len(vv[t]) > 0:
                        vv[t] = remove_outliers(vv[t])
        else:
            v['t1'] = remove_outliers(v['t1'])

    # # T1: mean ~= 21, std ~= 20
    # from analysis.utils import histogram
    # histogram(overheads['independent']['t1'], perc=False, bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 100000])
    # print(np.mean(overheads['independent']['t1']), np.std(overheads['independent']['t1']))

    # T2, T3, T5
    tmp_list = []
    for t in ['t2', 't3', 't5']:
        tmp_dict = {}
        for name, v in overheads.items():
            if name != 'independent':
                for shapes, vv in v.items():
                    if len(vv[t]) > 0:
                        if name not in tmp_dict.keys():
                            tmp_dict[name] = {}
                        tmp_dict[name][shapes] = (np.mean(vv[t]), np.std(vv[t]), len(vv[t]))
        tmp_list.append(tmp_dict)

    # # T4
    # for t, l in overheads['independent']['t4'].items():
        # print(t, np.mean(l), np.std(l), len(l))

    return {
        "t1": (np.mean(overheads['independent']['t1']), np.std(overheads['independent']['t1']), len(overheads['independent']['t1'])),
        "t2": tmp_list[0],
        "t3": tmp_list[1],
        "t4": {
            t: (np.mean(l), np.std(l), len(l)) for t, l in overheads['independent']['t4'].items()
        },
        "t5": tmp_list[2],
        "launches": launches_dict
    }, overheads


def create_shared_overhead(overhead_raw_files, overhead_stats_files, return_df=False):
    # Read raw data
    df = gather_overhead_raw(overhead_raw_files)

    # Read stats data for kernel launches details
    launches = gather_overhead_stats(overhead_stats_files)

    overhead = {}
    # T1
    t1 = df[df['type'] == 't1']
    overhead['t1'] = (np.mean(t1['time']), np.std(t1['time']), len(t1['time']))
    # T4
    t4 = df[df['type'] == 't4']
    overhead['t4'] = {}
    runtime_funcs = t4['op_name'].unique()
    for rf in runtime_funcs:
        rf_data = t4[t4['op_name'] == rf]
        overhead['t4'][rf] = (np.mean(rf_data['time']), np.std(rf_data['time']), len(rf_data['time']))
    # T2, T3, T5
    for t in ['t2', 't3', 't5']:
        data = df[df['type'] == t]
        overhead[t] = {}
        op_names = data['op_name'].unique()
        for op in op_names:
            overhead[t][op] = {}
            op_data = data[data['op_name'] == op]
            overhead[t][op][str(DUMMY_SHAPES)] = (np.mean(op_data['time']), np.std(op_data['time']), len(op_data['time']))
    # Launches
    overhead['launches'] = launches

    if return_df:
        return overhead, df
    return overhead


def gather_overhead_raw(overhead_raw_files):
    """
        overhead_raw_files: a (regex) glob of all raw overhead files
    """
    df = None
    for file in overhead_raw_files:
        model_name = file.split('/')[-2]
        batch_size = file.split('/')[-1].split('_')[1]
        tmp = pd.read_csv(file)
        tmp['model_name'] = [model_name] * len(tmp)
        tmp['batch_size'] = [batch_size] * len(tmp)
        if df is None:
            df = tmp
        else:
            df = pd.concat([df, tmp], ignore_index=True)
    return df


def gather_overhead_stats(overhead_stats_files):
    """
        overhead_stats_files: a (regex) glob of all overhead stats files
    """
    launches = {}
    for file in overhead_stats_files:
        with open(file) as f:
            overhead = json.load(f)
        for k, v in overhead['launches'].items():
            # Optimizer.step#SGD.step and Optimizer.zero_grad#SGD.zero_grad: handle during inference
            if 'Optimizer' in k:
                continue
            if k not in launches.keys() or len(v) > len(launches[k]): # Take the longest
                launches[k] = v
    return launches
