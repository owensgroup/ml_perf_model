import json
from pprint import pprint
from analysis.utils import *
from analysis.inference import get_kernel_time
from param_bench.train.compute.python.lib.pytorch.exec_graph_utils import ExecutionGraph

SKIPPED_TID = []

def get_dependency(graph, module_marker="##"):
    nodes = graph.get_nodes(clean=True)
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])

    forward_found = False
    branch = False # comm op detected, start to collect comp ops
    comm_op = None
    comp_ops = []
    overlaps = []
    prev_node = None
    for _, node in sorted_nodes:
        if module_marker in node.name:
            forward_found = True
        if not forward_found or not node.is_op():
            continue
        # print("   " if branch else "", node.id, node.name)
        if branch and node.tid not in SKIPPED_TID: # TODO: match with some trace tid
            # print(node.id, node.name, node.search_input_tensor(comm_op[1]))
            # Stop tracking compute ops when
            # 1. the current node is dependent on the current comm op
            # 2. meet wait ops
            # 3. meet another collective
            if (comm_op and depends_on_collective_output(node, comm_op[1])) or \
                is_wait_collective(node) or \
                has_comm_collective(node):
                branch = False
                overlaps.append({
                    'comp': comp_ops,
                    'comm': comm_op
                })
                comp_ops = []
                comm_op = None
            elif not (is_all2all(node) or is_allreduce(node)): # Don't include nccl:all_to_all/nccl:all_reduce for 3rd case downwards
                comp_ops.append((node.name, node.input_shapes[0] if node.input_shapes else None, node))
        if has_comm_collective(node) and not is_wait_collective(node):
            if is_all2all_parent(node): # Has a2a call
                branch = True
                tmp = node.get_child_by_name(["aten::new_empty", "aten::empty"])
                comm_op = ("all_to_all", tmp.outputs[0], tmp.output_shapes[0], node)
            elif is_allreduce_parent(node): # Has all_reduce call
                branch = True
                tmp = node.get_child_by_name("nccl:all_reduce")
                comm_op = ("all_reduce", tmp.inputs[0], tmp.input_shapes[0], node)
            # Some cases that nccl:all_to_all/nccl:all_reduce comes with trailing record_param_comms
            elif prev_node and (is_all2all(prev_node) or is_allreduce(prev_node)):
                branch = True
                comm_op = ("all_to_all", node.outputs[0], node.output_shapes[0], node) \
                            if is_all2all(prev_node) \
                            else ("all_reduce", node.outputs[0], node.output_shapes[0], node)
        prev_node = node

    # pprint(overlaps)
    print("Number of communication ops {}".format(len(overlaps)))
    return overlaps


def predict_overlaps(overlaps):
    for o in overlaps:
        comm_op = o["comm"][-1]
        comm_time = get_kernel_time(comm_op, ndevices=4)[0]
        comp_time = 0
        for x in o["comp"]:
            comp_op = x[-1]
            comp_time += get_kernel_time(comp_op, ndevices=4)[0]
        print("Communication time: {:.2f}".format(comm_time), "Computation time: {:.2f}".format(comp_time))


if __name__ == "__main__":
    eg_file = "./data/V100/e2e/DLRM_default/f/barrier_bucketed_allreduce/25/4_8192_distributed_0_graph.json"
    with open(eg_file) as f:
        graph = ExecutionGraph(json.load(f))
    overlaps = get_dependency(graph)
    predict_overlaps(overlaps)


### Overlaps:
# Lower bound: with another branch that has no data dependency
# Upper bound: also with ops previous to the scheduling of comm op (a common case)

### Detection:
# Lower bound: found comm OUTPUT tensor and wait input tensor (same), and extract everything between them
# Upper bound: found (the creation of)comm INPUT tensor and extract
