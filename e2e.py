import argparse, json
from analysis.exec_graph_utils import ExecutionGraph
from analysis.utils import PM_HOME, GPU_NAME
from analysis.inference import get_e2e_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict end-to-end training time of DLRM models.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    model_name = "{}_{}".format(args.model_name, args.num_gpus)
    exec_graph_file = "{}/data/{}/e2e/{}_graph.json".format(PM_HOME, GPU_NAME, model_name)
    with open(exec_graph_file) as f:
        graph = ExecutionGraph(json.load(f))
    overheads_file = "{}/data/{}/e2e/{}_overheads.json".format(PM_HOME, GPU_NAME, model_name)
    with open(overheads_file) as f:
        overheads = json.load(f)

    total_time, gpu_active_time = get_e2e_time(graph, overheads)
    print("Total time is: {:.2f}".format(total_time))
    print("GPU time is: {:.2f}".format(gpu_active_time))
