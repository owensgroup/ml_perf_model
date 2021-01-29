from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, json, sys
from pprint import pprint
import pydot
from enum import Enum

NodeType = Enum("NodeType", "OPERATOR LABEL")
class TensorNode:
    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.sources = set()
        self.sinks = set()
        self.shapes = set()

    def add_source(self, id):
        self.sources.add(id)

    def add_sink(self, id):
        self.sinks.add(id)

    def add_shape(self, shape):
        self.shapes.add(tuple(shape))

class Node:
    def __init__(self, name, id, parent_id, thread_id, inputs, input_types, input_shapes, outputs, output_types, output_shapes):
        self.name = name
        self.parent_id = parent_id
        self.parent = None
        self.children = []
        self.id = id
        self.thread_id = thread_id
        self.type = self.detect_type(name, inputs, outputs)
        self.inputs = inputs
        self.input_types = input_types
        self.input_shapes = input_shapes
        self.outputs = outputs
        self.output_types = output_types
        self.output_shapes = output_shapes

    def get_inputs(self):
        return zip(self.inputs, self.input_types, self.input_shapes)

    def get_outputs(self):
        return zip(self.outputs, self.output_types, self.output_shapes)

    def set_parent(self, parent):
        assert parent.id == self.parent_id
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

    def is_leaf_op(self):
        return not self.children

    def _get_base_op(self, node):
        if node.parent.type == NodeType.LABEL:
            return self
        return self._get_base_op(node.parent)

    def get_base_op(self):
        return self._get_base_op(self)

    def detect_type(self, name, inputs, outputs):
        if name.startswith("##"):
          return NodeType.LABEL
        else:
          return NodeType.OPERATOR

    def get_tensors(self, param_list):
        tensors = []
        for (id, type, shape) in param_list:
            if type.startswith('Tensor'):
                tensors.append((id, type, shape))
        return tensors

    def get_input_tensors(self):
        return self.get_tensors(self.get_inputs())

    def get_output_tensors(self):
        return self.get_tensors(self.get_outputs())

class ModelGraph:
    def __init__(self, json):
        self.nodes = {}
        self.tensors = {}
        for x in json:
            if x:
              self.nodes[x['id']] = Node(x['name'], x['id'], x['parent'], x['tid'], x['inputs'], x['input_types'],
                                    x['input_shapes'], x['outputs'], x['output_types'], x['output_shapes'])
              input_tensors = self.nodes[x['id']].get_input_tensors()
              output_tensors = self.nodes[x['id']].get_output_tensors()

              # build tensor refernece table
              for (t_id, type, shape) in input_tensors:
                  if t_id not in self.tensors:
                      self.tensors[t_id] = TensorNode(t_id, type)
                  self.tensors[t_id].add_sink(x["id"])
                  self.tensors[t_id].add_shape(shape)

              for (t_id, type, shape) in output_tensors:
                  if t_id not in self.tensors:
                      self.tensors[t_id] = TensorNode(t_id, type)
                  self.tensors[t_id].add_source(x["id"])
                  self.tensors[t_id].add_shape(shape)

        # populate parent and children nodes
        for _, n in self.nodes.items():
            self.nodes[n.parent_id].add_child(n)
            n.set_parent(self.nodes[n.parent_id])

    def op_stats(self):
        ops = {}
        for id, n in self.nodes.items():
            if n.type == NodeType.OPERATOR:
                if n.name in ops.keys():
                    ops[n.name]["count"] += 1
                else:
                    ops[n.name] = {"count" : 1}
        print("Operator, Count")
        for key, val in ops.items():
            print(f"{key}, {val['count']}")

    def gen_graph(self, file_name):
        dot = pydot.Dot(graph_type='digraph')
        for id, n in self.nodes.items():
            dot.add_node(pydot.Node(id, label=f"{n.name} ({n.id})", shape="box", style='filled', fillcolor="#fffbed"))
        for id, _ in self.tensors.items():
            dot.add_node(pydot.Node(id, label=f"T{id}", style = 'filled', fillcolor="#e8faff"))
        for id, n in self.nodes.items():
            dot.add_edge(pydot.Edge(n.parent_id, id, arrowhead="odiamond"))
            for (input, _, _) in n.get_input_tensors():
                dot.add_edge(pydot.Edge(input, id))
            for (output, _, _) in n.get_output_tensors():
                dot.add_edge(pydot.Edge(id, output))
        dot.write_svg(file_name, prog = 'dot')


def main():
    # set_simple_logging(escape_newlines=False)
    parser = argparse.ArgumentParser(description="Model graph building and analysis")
    parser.add_argument(
        "--input", type=str, required=True, help="The input trace file."
    )
    parser.add_argument(
        "--iter-start", type=str, default="### zero_grad ###", help="The iteration start operator, default: '### zero_grad ###'"
    )
    args = parser.parse_args()

    model_json = args.input

    with open(model_json) as model_data:
        model_graph = ModelGraph(json.load(model_data))
        model_graph.op_stats()
        for id, t in model_graph.tensors.items():
            print(id, t.type, t.sources, t.sinks, t.shapes)
        model_graph.gen_graph("model_graph.svg")


if __name__ == "__main__":
    main()
