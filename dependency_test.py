import json
from analysis.utils import *
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
            if depends_on_collective_output(node, comm_op[1]) or \
                is_wait_collective(node) or \
                has_comm_collective(node):
                branch = False
                overlaps.append({
                    'comp': comp_ops,
                    'comm': comm_op
                })
                comp_ops = []
                comm_op = None
            elif not (is_all2all(node) or is_allreduce(node)):
                comp_ops.append((node.name, node.input_shapes[0] if node.input_shapes else None))
        if has_comm_collective(node) and not is_wait_collective(node):
            if is_all2all_parent(node): # Has a2a call
                branch = True
                tmp = node.get_child_by_name(["aten::new_empty", "aten::empty"])
                comm_op = ("all_to_all", tmp.outputs[0], tmp.output_shapes[0])
            elif is_allreduce_parent(node): # Has all_reduce call
                branch = True
                tmp = node.get_child_by_name("nccl:all_reduce")
                comm_op = ("all_reduce", tmp.inputs[0], tmp.input_shapes[0])
            # Some cases that nccl:all_to_all/nccl:all_reduce comes with trailing record_param_comms
            elif prev_node and (is_all2all(prev_node) or is_allreduce(prev_node)):
                branch = True
                comm_op = ("all_to_all" if is_all2all(prev_node) else "all_reduce", tmp.outputs[0], tmp.output_shapes[0])


    from pprint import pprint
    pprint(overlaps)
    print("Number of communication ops {}".format(len(overlaps)))
    return

if __name__ == "__main__":
    eg_file = "./data/V100/e2e/DLRM_default/f/bucketed/4_8192_distributed_0_graph.json"
    with open(eg_file) as f:
        graph = ExecutionGraph(json.load(f))
    get_dependency(graph)


### Overlaps:
# Lower bound: with another branch that has no data dependency
# Upper bound: also with ops previous to the scheduling of comm op

### Detection:
# Lower bound: found comm OUTPUT tensor and wait input tensor (same), and extract everything between them
# Upper bound: found (the creation of)comm INPUT tensor and extract