import argparse, json, re, os
from analysis.exec_graph_utils import ExecutionGraph
from analysis.utils import PM_HOME, GPU_NAME
from analysis.inference import get_e2e_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict end-to-end training time of DLRM models.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    print("======= {}, {} GPU(s), batch size: {} =======".format(args.model_name, args.num_gpus, args.batch_size))

    exec_graph_file = "{}/data/{}/e2e/{}/{}_{}_graph.json".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size)
    with open(exec_graph_file) as f:
        graph = ExecutionGraph(json.load(f))
    overheads_file = "{}/data/{}/e2e/{}/{}_{}_overheads.json".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size)
    with open(overheads_file) as f:
        overheads = json.load(f)

    real_e2e_time = -1
    log_file = "{}/data/{}/e2e/{}/{}_{}.log".format(PM_HOME, GPU_NAME, args.model_name, args.num_gpus, args.batch_size)
    if os.path.exists(log_file):
        for line in open(log_file, 'r'):
            if re.search("Overall per-batch", line):
                real_e2e_time = float(line.split(' ')[4])

    total_time, gpu_active_time = get_e2e_time(graph, overheads)
    print("Total time: {:.2f}, GPU time: {:.2f}".format(total_time, gpu_active_time))
    if real_e2e_time != -1:
        print("Reference time: {:.2f}".format(real_e2e_time * 1000))
        print("Prediction error: {:.2f}%".format(abs(total_time / real_e2e_time / 1000 - 1) * 100))
